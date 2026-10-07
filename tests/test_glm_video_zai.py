"""Tests for the Z.ai direct video-understanding CLI (glm_video_zai)."""

import base64

import pytest

from glm_5_3_flash import glm_video_zai


def test_encode_video_data_url_builds_mp4_data_url(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42FAKE_VIDEO_BYTES")

    url = glm_video_zai.encode_video_data_url(str(video))

    scheme, encoded = url.split(",", 1)
    assert scheme == "data:video/mp4;base64"
    assert base64.b64decode(encoded) == video.read_bytes()


def test_encode_video_data_url_defaults_to_mp4_for_unknown_extension(tmp_path):
    video = tmp_path / "clip"
    video.write_bytes(b"FAKE")

    url = glm_video_zai.encode_video_data_url(str(video))

    assert url.startswith("data:video/mp4;base64,")


def test_build_video_content_uses_public_url_unchanged():
    official_demo = glm_video_zai.FALLBACK_VIDEO_URL
    block = glm_video_zai.build_video_content(video_url=official_demo, video=None)
    assert block == {
        "type": "video_url",
        "video_url": {"url": official_demo},
    }


def test_build_video_content_encodes_local_file_as_data_url(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"FAKE_VIDEO_BYTES")

    block = glm_video_zai.build_video_content(video_url=None, video=str(video))
    assert block["type"] == "video_url"
    assert block["video_url"]["url"].startswith("data:video/mp4;base64,")


def test_build_video_content_requires_a_source():
    with pytest.raises(ValueError, match="--video-url"):
        glm_video_zai.build_video_content(video_url=None, video=None)


def test_build_video_content_rejects_missing_file(tmp_path):
    missing = str(tmp_path / "nope.mp4")
    with pytest.raises(FileNotFoundError):
        glm_video_zai.build_video_content(video=str(missing))


@pytest.mark.parametrize(
    "message",
    [
        'content part type "video_url" is not supported; only text and image_url are',
        "video_url base64 data url is not supported for this model",
        "request body exceeds the maximum request size limit",
        "HTTP 413 Payload Too Large",
        "invalid base64 video content",
        "em..too large video payload",
    ],
)
def test_is_payload_rejection_true_for_payload_rejections(message):
    assert glm_video_zai.is_payload_rejection(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "无效的API密钥: Settings.ZAI_API_KEY is invalid",
        "HTTP 401 Unauthorized",
        "account arrearage: balance is insufficient",
        "1113 Internal error - please contact developer",
    ],
)
def test_is_payload_rejection_false_for_non_payload_errors(message):
    assert glm_video_zai.is_payload_rejection(message) is False


def test_fallback_video_url_is_the_official_docs_demo_video():
    assert (
        glm_video_zai.FALLBACK_VIDEO_URL
        == "https://cdn.bigmodel.cn/agent-demos/lark/113123.mov"
    )


def test_build_messages_appends_prompt_after_video_block():
    blocks = [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAA="}}]

    messages = glm_video_zai.build_messages(blocks, "What happens here?")

    assert messages == [
        {
            "role": "user",
            "content": blocks + [{"type": "text", "text": "What happens here?"}],
        }
    ]


def test_parse_cli_defaults_target_the_local_video():
    args = glm_video_zai.parse_cli([])

    assert args.video == "data/video/Ray_Kaguya_480p.mp4"
    assert args.video_url is None
    assert args.model == "glm-5.3-flash"
    assert args.reasoning_effort == "low"
    assert args.stream is True
    assert "timestamp" in args.prompt.lower()
    assert args.save is None


def test_parse_cli_video_url_overrides_the_default_local_video():
    args = glm_video_zai.parse_cli([f"--video-url={glm_video_zai.FALLBACK_VIDEO_URL}"])

    assert args.video_url == glm_video_zai.FALLBACK_VIDEO_URL
    assert args.video is None


def test_parse_cli_no_stream_disables_streaming():
    args = glm_video_zai.parse_cli(["--no-stream"])
    assert args.stream is False


def test_parse_cli_video_path_override():
    args = glm_video_zai.parse_cli(["--video", "D:/somewhere/clip.mp4"])
    assert args.video == "D:/somewhere/clip.mp4"


# --- run_analysis (client injected so no network is touched) -----------------

from types import SimpleNamespace  # noqa: E402

VIDEO_BLOCK = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAA="}}


def _fake_response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content="hmm"))
        ]
    )


def _chunks():
    def stream():
        for piece in ("Hello", ", ", "world"):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))]
            )

    return stream


class FakeCompletions:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome()
        return outcome


def make_client(outcomes):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(outcomes)))


_RUN_KW = {"model": "glm-5.3-flash", "reasoning_effort": "low", "stream": False}


def test_run_analysis_non_stream_sends_video_block_and_prompt():
    client = make_client([_fake_response("the answer")])

    out = glm_video_zai.run_analysis(
        client, VIDEO_BLOCK, "What happens?", fallback_url=None, **_RUN_KW
    )

    assert out == "the answer"
    assert len(client.chat.completions.calls) == 1
    call = client.chat.completions.calls[0]
    assert call["model"] == "glm-5.3-flash"
    assert call["reasoning_effort"] == "low"
    assert call["temperature"] == 1.0
    assert call["top_p"] == 0.95
    assert call["max_tokens"] == 4096
    content = call["messages"][0]["content"]
    assert content[0] == VIDEO_BLOCK
    assert content[-1] == {"type": "text", "text": "What happens?"}


def test_run_analysis_retries_with_demo_url_on_payload_rejection(capsys):
    client = make_client(
        [
            RuntimeError("Error code 400: base64 video_url payload not supported"),
            _fake_response("demo analysis"),
        ]
    )

    out = glm_video_zai.run_analysis(
        client,
        VIDEO_BLOCK,
        "What happens?",
        fallback_url=glm_video_zai.FALLBACK_VIDEO_URL,
        **_RUN_KW,
    )

    assert out == "demo analysis"
    retry_call = client.chat.completions.calls[1]
    assert retry_call["messages"][0]["content"][0]["video_url"]["url"] == (
        glm_video_zai.FALLBACK_VIDEO_URL
    )
    err = capsys.readouterr().err
    assert "demo video" in err.lower()


def test_run_analysis_reraises_non_payload_errors_same_call_count(capsys):
    client = make_client([RuntimeError("Error 401: invalid api key")])

    with pytest.raises(RuntimeError, match="invalid api key"):
        glm_video_zai.run_analysis(
            client, VIDEO_BLOCK, "What happens?",
            fallback_url=glm_video_zai.FALLBACK_VIDEO_URL, **_RUN_KW,
        )

    assert len(client.chat.completions.calls) == 1
    assert "demo" not in capsys.readouterr().err


def test_run_analysis_streams_chunks_and_returns_full_text():
    client = make_client([_chunks()])

    out = glm_video_zai.run_analysis(
        client, VIDEO_BLOCK, "What happens?",
        fallback_url=None, model="glm-5.3-flash",
        reasoning_effort="low", stream=True,
    )

    assert out == "Hello, world"
    call = client.chat.completions.calls[0]
    assert call["stream"] is True


def test_write_analysis_saves_text_to_markdown(tmp_path):
    destination = tmp_path / "nested" / "analysis.md"

    glm_video_zai.write_analysis(str(destination), "RESULT TEXT")

    assert destination.read_text(encoding="utf-8") == "RESULT TEXT"


def test_module_is_runnable_as_main_exits_before_network_for_missing_file(
    tmp_path, monkeypatch, capsys
):
    """Run python -m glm_5_3_flash.glm_video_zai for real (no network).

    With a nonexistent --video, main() must bail out with a clear message
    before making any request — this exercises the module's top-level
    definition order, which a plain import does not.
    """
    import runpy

    monkeypatch.setattr(
        "sys.argv",
        ["glm_video_zai", "--video", str(tmp_path / "definitely-missing.mp4")],
    )

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("glm_5_3_flash.glm_video_zai", run_name="__main__")

    assert "video file not found" in str(excinfo.value)
    assert "parse_cli" not in str(excinfo.value)  # no NameError pollution
