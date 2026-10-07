# Video Understanding: OpenCode Go vs Official Z.ai Endpoint

> **Status as of 2026-10-07.** The previous version of this document was written
> against older Z.ai docs and is now fully obsolete; this version reflects the
> official docs read today plus a **successful live test** from this session.
> What follows separates "read in official docs" from "verified live" where it matters.

## Why the Current OpenCode Go Script Cannot Send Real Video

The OpenCode Go script (`glm_video_openai.py`) talks to the **OpenCode Go proxy**:

- Endpoint: `https://opencode.ai/zen/go/v1/chat/completions`
- Model available there: `glm-5.3-flash`

When we send a `{"type": "video_url", ...}` content block, the proxy rejects it:

> `content part type "video_url" is not supported; only text and image_url are`

OpenCode's own published model config lists `glm-5.3-flash` as text + image input
only. Even though the underlying Z.ai model now supports video natively, **the
OpenCode Go proxy strips/rejects video content**. That is why `glm_video_openai.py`
had to fall back to extracting frames with ffmpeg and sending timestamped
`image_url` blocks.

## What the Official Z.ai Endpoint Supports (updated 2026-10-07)

Yes — the official Z.ai API accepts real video input, and (larger finding) the
model we're already paying for does it directly:

- Endpoint: `https://api.z.ai/api/paas/v4/chat/completions`
  (same shape for `https://open.bigmodel.cn/api/paas/v4/chat/completions`)
- It is OpenAI-SDK compatible: point the `openai` Python client at
  `https://api.z.ai/api/paas/v4/` (optional `Accept-Language: en-US,en` header).
- **Router model note (corrected):** the old claim that Z.ai video was handled
  only by `glm-5v-turbo` / `glm-4.6v` / `glm-4.5v` is **outdated**. Per the
  current Chat Completion API reference, the vision request model enum is now:
  `glm-5.3-flashx`, **`glm-5.3-flash`**, `glm-4.6v`, `autoglm-phone-multilingual`,
  `glm-4.6v-flash`, `glm-4.6v-flashx`, `glm-4.5v`.
- **`GLM-5.3-Flash` / `GLM-5.3-FlashX` are Z.ai's first native multimodal models**:
  official input modality is **Video / Image / Text / File**, 1M-token context,
  up to 128K output tokens. The official video example in the API docs itself
  uses `glm-5.3-flashx`. `glm-5.3-flash` is fully on the GLM Coding Plan
  (FlashX is not yet), so a coding-plan-style key may only serve the plain
  flash model.
- Content block shape (unchanged from before):

  ```json
  {"type": "video_url", "video_url": {"url": "<direct video URL>"}}
  ```

  Official example payload pairs this with `glm-5.3-flashx` and the demo video
  `https://cdn.bigmodel.cn/agent-demos/lark/113123.mov`.

### Constraints (from Z.ai's official docs)

| Constraint | Value |
| --- | --- |
| Video size | ≤ 200 MB per video |
| Formats | `mp4`, `mkv`, `mov` |
| Videos per message | GLM-5V / GLM-4.6V series: up to 2; GLM-4.5V: 1 |
| URL type (documented) | A direct, publicly accessible URL to the video file |
| Base64 video | **Not documented** — schema only says "Video URL address" (images alone officially allow base64 data URLs, ≤ 5 MB each) |

### New Finding: Local Files DO Work via Base64 Data URLs (verified live)

The docs document no local-file route for video (and the official File Upload
API `POST /paas/v4/files` only admits document/image formats — pdf, doc, xlsx,
ppt, txt, jpg, png — never video). Nevertheless, in this session:

- We sent our local 13.1 MB `data/video/Ray_Kaguya_480p.mp4` as
  `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`
  (~18 MB body) to `glm-5.3-flash` through the official endpoint.
- **Z.ai accepted it and returned a correct, timestamped analysis on the first
  call** — speakers, on-screen text (including Japanese captions), and a
  00:00:00–00:02:29 timeline. No demo-URL fallback was needed.

So: **base64-for-video is undocumented but empirically supported** with
GLM-5.3-Flash today. Treat it as an undocumented capability that could change;
the script keeps a documented-path fallback ready. For very large videos the
documented public-URL route remains the safe choice (and avoids an 18 MB+ JSON body).

### Z.ai vs YouTube

- A direct `.mp4`/`.mov`/`.mkv` file URL anywhere public: works fine — that's
  the documented path (and fallback target in our script).
- A YouTube watch page is still NOT a direct video file URL; Z.ai's fetcher can't
  parse it. Practically that still means `yt-dlp` → local file → base64 (which we
  now know works) or `yt-dlp` → upload → direct URL.

## Side-by-Side Summary

| Capability | OpenCode Go (`opencode.ai/zen/go/v1`) | Official Z.ai (`api.z.ai`) |
| --- | --- | --- |
| `text` input | ✅ | ✅ |
| `image_url` input | ✅ (base64 data URLs too) | ✅ (base64 data URLs too, documented) |
| `video_url` input | ❌ rejected | ✅ native support on GLM-5.3-Flash(-X), GLM-4.6V, GLM-4.5V, … |
| Local video via base64 data URL | — (no video at all) | ✅ **verified live 2026-10-07** (undocumented, ≤ ~200 MB video / ~18 MB body for the tested clip) |
| Official local-video upload | ❌ | ❌ (File Upload API: documents/images only) |
| Public direct video URL, no local download | ❌ | ✅ |
| YouTube watch-page URL as input | ❌ | ❌ (not a direct file) |
| Model for video | — (`glm-5.3-flash` = text+image via proxy) | GLM-5.3-Flash(-X) native multimodal + GLM-4.6V/GLM-4.5V family |
| Context window | varies | 1M for GLM-5.3-Flash series |
| Auth | `OPENCODE_API_KEY` (Go subscription) | Z.ai API key (per-token pricing; `glm-5.3-flash` on the GLM Coding Plan) |
| Extra routing header | Needs `x-opencode-session` | Not required (optional `Accept-Language`) |

## What's in the Repo Now

| File | Route | What it does |
| --- | --- | --- |
| `glm_video_zai.py` | Z.ai official endpoint | Single-module CLI with a real test suite (28 tests). Local file → base64 `video_url` (verified working); `--video-url` for the documented path; auto-retries with Z.ai's official demo video if a local payload is ever rejected; streams by default; `--save` for markdown output; hard-coded default target `data/video/Ray_Kaguya_480p.mp4`. |
| `glm_video_openai.py` | OpenCode Go proxy | Frames-only workaround (ffmpeg → 12–10 base64 `image_url` blocks) because the proxy rejects `video_url`. Still the only way to use a video through the Go subscription itself. |

Recommended settings observed in the Z.ai docs for GLM-5.3-Flash:
`temperature: 1`, `top_p: 0.95`, `thinking.type: enabled` (only value supported),
`reasoning_effort` ∈ {`low`, `high`, `max`} — our script defaults to `low` for
cheap quick video analysis, `max` is what the docs recommend for coding.

## Sources (read/verified 2026-10-07)

- Chat Completion API reference — `video_url` item, vision model enum including
  `glm-5.3-flash`/`glm-5.3-flashx`, official video example, 200 MB/2-video
  limits, `file` content type with `file_id`/`file_url`/`file_data`:
  https://docs.z.ai/api-reference/llm/chat-completion
- GLM-5.3-Flash/FlashX guide — native multimodal (Video/Image/Text/File input),
  1M context, recommended settings, GLM Coding Plan availability:
  https://docs.z.ai/guides/vlm/glm-5.3-flash
- File Upload API reference — allowed formats exclude every video format:
  https://docs.z.ai/api-reference/agents/file-upload
- OpenCode Go docs — client requirements & session header + current rejection of
  video content blocks: https://opencode.ai/docs/go/
- **Live verification (2026-10-07):** 13.1 MB local mp4 → base64 `video_url` →
  `glm-5.3-flash` returned a full timestamped analysis (saved to
  `src/glm_5_3_flash/analysis_ray_kaguya.md`), zero fallback retries.
