# GLM Video Understanding (Z.ai + OpenCode Go)

Send real videos — local files or URLs — to Zhipu/Z.ai's **GLM-5.3-Flash** for
timestamped, speaker-aware analysis, straight from Python. This repo compares two
delivery routes and ships a working CLI for the one that actually supports video:

| Route | Script | Video support | Status |
| --- | --- | --- | --- |
| **Z.ai official API** | `src/glm_5_3_flash/glm_video_zai.py` | ✅ native `video_url` | **Working** (live-verified, tested suite) |
| OpenCode Go proxy | `src/glm_5_3_flash/glm_video_openai.py` | ⚠️ `video_url` rejected → ffmpeg frames workaround | Fallback only |

## What makes this work

- **GLM-5.3-Flash / GLM-5.3-FlashX are Z.ai's first native multimodal models** —
  official input modalities: *Video / Image / Text / File*, with a 1M-token
  context window.
- The documented video input is a **direct public URL** (`≤ 200 MB`, `mp4`/`mkv`/`mov`).
- **Local files are undocumented territory — and they work anyway.** This repo
  verified live (2026-10-07) that a base64 data URL in `video_url` is accepted,
  first with a 13 MB clip, then with a 66 MB / 15-minute video (~88 MB request
  body). Nothing is uploaded to any third party.
- If Z.ai ever rejects the local payload, the CLI retries automatically against
  Z.ai's own official demo video so you always see a result.

## Project layout

```
src/glm_5_3_flash/
├── glm_video_zai.py            # Z.ai direct CLI (the main pipeline)
├── glm_video_openai.py         # OpenCode Go + ffmpeg frames workaround
├── video_url_support_zai_vs_opencode_go.md   # research: why two routes exist
├── analysis_ray_kaguya.md      # example output: anime montage analysis
├── analysis_bisakah_ai_membuat_uang.md       # example output: 15-min essay video
└── analysis_louisiana_purchase.md            # example output
tests/
└── test_glm_video_zai.py       # 28 tests (pytest), incl. a "run as main" regression
data/video/                     # local video files (gitignored)
```

## Quick start

Requires Python **≥ 3.12** (repo is pinned to 3.14.3) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create .venv + install dependencies

# .env — at least one key, none are committed (see .gitignore)
ZAI_API_KEY=<your-z.ai-key>        # https://z.ai/manage-apikey/apikey-list
# OPENCODE_API_KEY=<your-opencode-go-key>   # only needed for the Go route
```

## Usage — Z.ai route (`glm_video_zai.py`)

```bash
# Default: analyze data/video/Ray_Kaguya_480p.mp4, streamed live
uv run python -m glm_5_3_flash.glm_video_zai

# Any local video file (sent as base64 — nothing leaves your machine)
uv run python -m glm_5_3_flash.glm_video_zai --video path/to/clip.mp4

# Documented mode: a direct public video URL
uv run python -m glm_5_3_flash.glm_video_zai --video-url https://cdn.example.com/clip.mp4

# Full control
uv run python -m glm_5_3_flash.glm_video_zai \
    --video my_clip.mp4 \
    --prompt "List every on-screen product and when it appears" \
    --model glm-5.3-flash \
    --reasoning-effort max \
    --save analysis.md
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--video` | `data/video/Ray_Kaguya_480p.mp4` | Local file → base64 `video_url` |
| `--video-url` | — | Public direct URL; overrides `--video` |
| `--prompt` | detailed timestamped summary | Anything you'd ask about the video |
| `--model` | `glm-5.3-flash` | `glm-5.3-flashx` also valid (not on the Coding Plan yet) |
| `--reasoning-effort` | `low` | `low` / `high` / `max` (flash only supports these) |
| `--no-stream` | off | Wait for the full reply instead of streaming |
| `--save` | — | Write the final analysis to a markdown file |

Settings follow Z.ai's recommendations for the flash family: `temperature: 1.0`,
`top_p: 0.95`, `thinking: enabled` (the only supported value), optional
`Accept-Language: en-US,en` header.

## Usage — OpenCode Go route (`glm_video_openai.py`)

The OpenCode Go proxy (`https://opencode.ai/zen/go/v1`) rejects `video_url`
content blocks — its published config lists `glm-5.3-flash` as text + image
only — so this script extracts evenly spaced frames with ffmpeg and sends them
as timestamped base64 `image_url` blocks:

```bash
python src/glm_5_3_flash/glm_video_openai.py --video-path ./my_clip.mp4 \
    --prompt "What happens in this video?" --max-frames 12 --stream
```

Requires `ffmpeg`/`ffprobe` on `PATH` and `OPENCODE_API_KEY`.

## Testing

```bash
uv run python -m pytest        # 28 tests, no network calls
```

The suite covers the data-URL encoder, content-block builder, payload-rejection
classification (auth/quota errors must *not* trigger the fallback), CLI defaults,
the streaming/non-streaming request paths (with an injected fake client), the
markdown saver, and a `runpy`-based test that executes the module exactly the
way `python -m` does — added after catching a definition-order `NameError` that
plain imports never expose.

## Notes & limits

- Video limits (Z.ai docs): **≤ 200 MB**, `mp4` / `mkv` / `mov`, up to 2 videos
  per message on the 5V/4.6V series.
- **YouTube watch URLs are not direct file URLs** — they won't work in either
  route. Practical flow: `yt-dlp` → local file → base64. Extracted
  `googlevideo.com` CDN links are IP-bound and expire within hours, so they
  can't be handed to Z.ai's fetcher either.
- Re-hosting a video publicly (catbox/tmpfiles/own bucket/tunnel) is an
  alternative for very large files, but redundant below ~150 MB given the
  verified base64 path.
- A live run consumes Z.ai credit (video tokens are priced; flash-tier rates).

## Sources

- [Chat Completion API reference](https://docs.z.ai/api-reference/llm/chat-completion) —
  `video_url` schema, vision model enum, official video example, limits
- [GLM-5.3-Flash/FlashX guide](https://docs.z.ai/guides/vlm/glm-5.3-flash) —
  native multimodality, 1M context, recommended settings
- [File Upload API](https://docs.z.ai/api-reference/agents/file-upload) —
  documents/images only (no video upload exists)
- [OpenCode Go docs](https://opencode.ai/docs/go/) — proxy behavior & session header
- Session research & live verification details:
  [`video_url_support_zai_vs_opencode_go.md`](src/glm_5_3_flash/video_url_support_zai_vs_opencode_go.md)
