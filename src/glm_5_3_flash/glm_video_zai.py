"""Video understanding with GLM-5.3 (Flash) through Z.ai's official Chat Completion API.

Endpoint: https://api.z.ai/api/paas/v4/chat/completions
Model docs: https://docs.z.ai/guides/vlm/glm-5.3-flash
API reference: https://docs.z.ai/api-reference/llm/chat-completion

The official Z.ai examples pass a public direct video URL via
{"type": "video_url", "video_url": {"url": ...}} (<= 200 MB, mp4/mkv/mov).
For local files this script sends a base64 data URL instead — undocumented for
video, but images accept base64 data URLs on Z.ai, so it is worth trying. If
the API rejects the local video payload, the script retries against the
official demo URL used in Z.ai's docs:
https://cdn.bigmodel.cn/agent-demos/lark/113123.mov
"""

import argparse
import base64
import mimetypes
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


def encode_video_data_url(video_path: str) -> str:
    """Return a base64 data URL for a local video file."""
    with open(video_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    mime_type, _ = mimetypes.guess_type(video_path)
    if mime_type is None:
        mime_type = "video/mp4"  # reasonable default for unknown extensions

    return f"data:{mime_type};base64,{encoded}"


def build_video_content(video_url: str | None = None, video: str | None = None) -> dict:
    """Build the {"type": "video_url", ...} content block for Z.ai.

    Prefers a public direct URL when given; encodes a local file as a base64
    data URL otherwise. Exactly one source must be provided.
    """
    if video_url:
        return {"type": "video_url", "video_url": {"url": video_url}}

    if video is not None:
        if not os.path.isfile(video):
            raise FileNotFoundError(f"video file not found: {video}")
        return {
            "type": "video_url",
            "video_url": {"url": encode_video_data_url(video)},
        }

    raise ValueError("provide either --video-url or --video")


FALLBACK_VIDEO_URL = "https://cdn.bigmodel.cn/agent-demos/lark/113123.mov"


_FALLBACK_FALSE_MARKERS = (
    "api key",
    "invalid api",
    "unauthorized",
    "401",
    "arrearage",
    "balance",
    "quota",
    "internal error",
)

_FALLBACK_TRUE_MARKERS = (
    "base64",
    "not supported",
    "unsupported",
    "content type",
    "payload",
    "too large",
    "413",
    "request size",
    "maximum size",
)


def is_payload_rejection(error_message: str) -> bool:
    """Return True when an API error looks like a rejected video payload.

    The demo-URL fallback only makes sense when the delivery method
    (base64/video block) itself was rejected — not for auth, quota, or
    server-side problems, which retrying with another video cannot fix.
    """
    lowered = error_message.lower()
    if any(marker in lowered for marker in _FALLBACK_FALSE_MARKERS):
        return False
    return any(marker in lowered for marker in _FALLBACK_TRUE_MARKERS)


def build_messages(content_blocks: list[dict], prompt: str) -> list[dict]:
    """Wrap the video content blocks plus the text prompt in a user message."""
    return [
        {"role": "user", "content": content_blocks + [{"type": "text", "text": prompt}]}
    ]


DEFAULT_MAX_TOKENS = 4096


def run_analysis(
    client,
    video_block: dict,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str = "low",
    stream: bool = True,
    fallback_url: str | None = None,
) -> str:
    """Run the video-understanding request; return the model's answer.

    When ``fallback_url`` is set (local-file mode) and the response looks like
    a payload/content-type rejection, retry once against that URL — Z.ai's
    official demo video — and announce it on stderr so stdout stays clean.
    """
    try:
        return _send_chat(
            client, video_block, prompt,
            model=model, reasoning_effort=reasoning_effort, stream=stream,
        )
    except Exception as exc:  # noqa: BLE001 — dispatches on the error text
        if fallback_url and is_payload_rejection(str(exc)):
            print(
                f"Z.ai rejected the local video payload ({str(exc).strip()}) — "
                f"retrying with the official demo video used in Z.ai's docs: {fallback_url}",
                file=sys.stderr,
            )
            return _send_chat(
                client,
                {"type": "video_url", "video_url": {"url": fallback_url}},
                prompt,
                model=model, reasoning_effort=reasoning_effort, stream=stream,
            )
        raise


def _send_chat(
    client,
    video_block: dict,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    stream: bool,
) -> str:
    """Send one chat completion request; stream output live when enabled."""
    kwargs: dict = {
        "model": model,
        "messages": build_messages([video_block], prompt),
        "temperature": 1.0,  # Z.ai recommended settings for GLM-5.3-Flash
        "top_p": 0.95,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
    }

    if not stream:
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        return content

    pieces: list[str] = []
    for chunk in client.chat.completions.create(**kwargs):
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and getattr(delta, "content", None):
            pieces.append(delta.content)
            print(delta.content, end="", flush=True)
    if pieces:
        print()
    return "".join(pieces)


def write_analysis(path: str, text: str) -> None:
    """Save the analysis text, creating parent directories as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# Z.ai's official OpenAI-compatible base URL for the PAAS v4 API.
# OpenAI SDK usage is documented at https://docs.z.ai/guides/develop/openai/python
BASE_URL = "https://api.z.ai/api/paas/v4/"


def main() -> None:
    load_dotenv()
    args = parse_cli()

    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        sys.exit(
            "Error: set the ZAI_API_KEY environment variable (e.g. in .env) "
            "with your Z.ai API key."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        default_headers={"Accept-Language": "en-US,en"},
    )

    try:
        video_block = build_video_content(args.video_url, args.video)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"Error: {exc}")

    # Demo-URL retry only makes sense for a locally encoded video, whose
    # delivery method may be rejected; a user-supplied URL already follows
    # the documented path.
    fallback_url = None if args.video_url else FALLBACK_VIDEO_URL

    answer = run_analysis(
        client,
        video_block,
        args.prompt,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        stream=args.stream,
        fallback_url=fallback_url,
    )

    if not args.stream:
        print(answer)

    if args.save:
        write_analysis(args.save, answer)
        print(f"Analysis saved to {args.save}", file=sys.stderr)


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments with sensible defaults for this repo."""
    parser = argparse.ArgumentParser(
        description="Ask GLM-5.3 (Flash) about a video through Z.ai's official API."
    )
    parser.add_argument(
        "--video",
        default="data/video/Ray_Kaguya_480p.mp4",
        help=(
            "Path to a local video file (sent as a base64 data URL). "
            "Defaults to the Ray Kaguya clip in data/video/."
        ),
    )
    parser.add_argument(
        "--video-url",
        help=(
            "Public direct URL of the video instead of a local file — the "
            "officially documented mode (<= 200 MB, mp4/mkv/mov)."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Describe and summarize what happens in this video in detail by "
            "timestamp. Identify speakers, on-screen text, and notable events."
        ),
        help="The question or instruction to send alongside the video.",
    )
    parser.add_argument(
        "--model",
        default="glm-5.3-flash",
        help="Z.ai model code (glm-5.3-flash or glm-5.3-flashx).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "high", "max"],
        default="low",
        help="GLM-5.3-Flash thinking depth (default low for cheap quick runs).",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Wait for the full reply instead of streaming it.",
    )
    parser.add_argument(
        "--save",
        help="Optionally write the final analysis to this markdown file.",
    )
    args = parser.parse_args(argv)
    if args.video_url:
        args.video = None  # URL mode replaces the default local file
    return args


if __name__ == "__main__":
    main()
