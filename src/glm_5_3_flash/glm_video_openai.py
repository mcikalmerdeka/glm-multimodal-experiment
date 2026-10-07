"""
GLM-5.3-Flash video understanding via your OpenCode Go subscription.

OpenCode Go exposes an OpenAI-compatible Chat Completions endpoint that
proxies to GLM-5.3-Flash (and other models), so this uses the standard
`openai` Python SDK pointed at OpenCode Go's base URL instead of a direct
Z.ai API key.

Setup:
    pip install openai
    export OPENCODE_API_KEY="your-opencode-go-key-here"   # from opencode.ai/auth

Usage:
    # Video from a public URL
    python glm_video_understanding.py --video-url "https://example.com/clip.mp4" \
        --prompt "What happens in this video?"

    # Local video file (auto base64-encoded as a data URL)
    python glm_video_understanding.py --video-path "./my_clip.mp4" \
        --prompt "Describe the key events and who is speaking."

Notes:
    - Model id on OpenCode Go: glm-5.3-flash
    - Base URL: https://opencode.ai/zen/go/v1
    - Prefer a public URL over base64 for large video files; base64 payloads
      bloat the request body and are slower to upload.
    - IMPORTANT: OpenCode Go's own published model config for glm-5.3-flash
      currently lists its input types as text and image only (not video),
      even though the underlying Z.ai model supports video input directly.
      This script sends the standard `video_url` content block since that's
      the correct OpenAI-compatible shape for it, but whether OpenCode Go's
      proxy actually forwards/accepts it for this model isn't guaranteed.
      If you get an error mentioning an unsupported content type, that's
      why — in that case you'd need to go through Z.ai directly (or another
      provider that has confirmed video support) instead of OpenCode Go.
"""

import argparse
import base64
import mimetypes
import os
import subprocess
import sys
import tempfile
import uuid

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://opencode.ai/zen/go/v1"
MODEL_ID = "glm-5.3-flash"

# Stable per-run session ID (regenerated each process invocation, but constant
# across all requests within that conversation/run, which is what Go requires).
SESSION_ID = str(uuid.uuid4())


def build_video_content_block(video_url: str | None, video_path: str | None) -> dict:
    """Build the {"type": "video_url", ...} content block from a URL or local file."""
    if video_url:
        return {"type": "video_url", "video_url": {"url": video_url}}

    if video_path:
        if not os.path.isfile(video_path):
            sys.exit(f"Error: video file not found: {video_path}")

        mime_type, _ = mimetypes.guess_type(video_path)
        if mime_type is None:
            mime_type = "video/mp4"  # reasonable default

        with open(video_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        data_url = f"data:{mime_type};base64,{encoded}"
        return {"type": "video_url", "video_url": {"url": data_url}}

    sys.exit("Error: you must provide either --video-url or --video-path.")


def extract_frames(video_path: str, num_frames: int) -> list[dict]:
    """Extract evenly-spaced frames from a local video with ffmpeg.

    Returns image_url content blocks, roughly ordered, with the frame's
    timestamp embedded in a preceding text marker for each frame.
    """
    import json

    if not os.path.isfile(video_path):
        sys.exit(f"Error: video file not found: {video_path}")

    # Get duration via ffprobe.
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])

    stamps = [(idx + 0.5) * duration / num_frames for idx in range(num_frames)]

    with tempfile.TemporaryDirectory() as tmp:
        blocks: list[dict] = []
        for idx, stamp in enumerate(stamps):
            out_file = os.path.join(tmp, f"frame_{idx:03d}.jpg")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", f"{stamp:.3f}", "-i", video_path,
                    "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "5",
                    out_file,
                ],
                capture_output=True, check=True,
            )
            with open(out_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            minutes, seconds = divmod(int(stamp), 60)
            blocks.append(
                {"type": "text", "text": f"[Frame at {minutes:02d}:{seconds:02d}]"}
            )
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask GLM-5.3-Flash about a video.")
    parser.add_argument("--video-url", help="Public URL of the video to analyze.")
    parser.add_argument("--video-path", help="Path to a local video file to analyze.")
    parser.add_argument(
        "--prompt",
        default="Describe and summarize what happens in this video in detail by timestamp.",
        help="The question or instruction to send alongside the video.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response as it's generated instead of waiting for the full reply.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Number of frames to extract from a local video (frames mode).",
    )
    args = parser.parse_args()

    if not args.video_url and not args.video_path:
        parser.error("provide either --video-url or --video-path")

    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        sys.exit("Error: set the OPENCODE_API_KEY environment variable with your OpenCode Go API key.")

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        # OpenCode Go requires a stable session ID header per conversation
        # for routing/prompt caching; omitting it returns MissingSessionID.
        default_headers={"x-opencode-session": SESSION_ID},
    )

    if args.video_url:
        content_blocks = [build_video_content_block(args.video_url, None)]
    else:
        # OpenCode Go rejects video_url content; fall back to extracted frames
        # sent as image_url blocks, each preceded by its timestamp marker.
        content_blocks = extract_frames(args.video_path, max(1, args.max_frames))

    messages = [
        {
            "role": "user",
            "content": content_blocks + [{"type": "text", "text": args.prompt}],
        }
    ]

    if args.stream:
        response_stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=1,
            top_p=0.95,
            stream=True,
        )
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
        print()
    else:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=1,
            top_p=0.95,
        )
        print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
