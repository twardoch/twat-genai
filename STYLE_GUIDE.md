# twat-genai Style Guide

A reference for contributors and users that explains the generative-AI concepts,
provider choices, and code conventions used in this library.

---

## What is generative AI (for image work)?

Generative AI models are neural networks trained on billions of image–text pairs.
Given a text description (a *prompt*), the model synthesises a new image from
scratch — not by retrieving or copying an existing picture, but by learning the
statistical patterns that connect words to visual features.

The same family of models can also *transform* an existing image: change its
style, extend its borders, sharpen its resolution, or blend it with another
image guided by structural cues.

**twat-genai** wraps several cloud providers that host these models so you don't
need a GPU, a model file, or deep ML knowledge — just an API key and a prompt.

---

## Providers

Each provider is a cloud service that runs GPU-accelerated AI models.
`twat-genai` exposes a thin, testable client for each one.

### FAL (fal.ai) — primary provider

**What it is:** A serverless GPU platform hosting hundreds of open-source models
(mainly the FLUX family) via a queue-based job API.

**What it can do:** Text-to-image, image-to-image, ControlNet-style conditioning
(edge/depth maps), upscaling (8 specialist models), outpainting (2 models), and
LoRA application.

**Client:** `FALEngine` (async context manager, used by the main CLI commands).
`FalApiClient` is the lower-level async HTTP wrapper.

**Auth:** `FAL_KEY` environment variable.

**When to use:** This is the default for all image-generation CLI commands
(`text`, `image`, `canny`, `depth`, `upscale`, `outpaint`).

---

### Chutes — video and specialist image work

**What it is:** An inference-as-a-service platform with a simple REST API,
focusing on video generation (SkyReels) and specialist image models (HiDream,
background removal).

**What it can do:** Image generation, image editing (HiDream style), background
removal (`rembg`), text-to-video and image-to-video (SkyReels), and
first-last-frame-to-video (WAN).

**Client:** `ChutesClient` (sync + async methods with injectable transport).

**Auth:** `CHUTES_API_KEY` or `CHUTES_API_TOKEN`.

**When to use:** When you need video output or the HiDream editing model.

---

### OpenAI-compatible — quote-card and general image generation

**What it is:** Any HTTP endpoint that speaks the OpenAI `/images/generations`
format.  The default points at `api.openai.com`; set `CLIPROXY_API_ENDPOINT`
to point at any compatible service.

**What it can do:** Text-to-image (and basic image editing via the generations
endpoint), with configurable size and output format.

**Client:** `OpenAIImageClient` (sync).

**Auth:** `OPENAI_API_KEY` or `CLIPROXY_API_KEY`.

**When to use:** When you need DALL·E 3 / GPT Image-2 output, or you are
routing through a cost-management proxy.

---

### Gemini / Nano Banana — style-transfer and creative transforms

**What it is:** Google's Gemini multimodal model (via the `generateContent`
API), which understands both text instructions and inline image data.

**What it can do:** Creative image transformations described in natural language
— e.g. "match the lighting of image A while preserving the identity from image B".

**Client:** `GeminiImageClient` (sync).

**Auth:** `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

**Supported model aliases:** `flash25`, `pro3`, `flash31` / `nano` (default).

**When to use:** Open-ended creative editing that doesn't fit fixed operation
categories (upscale, canny, etc.).

---

## Operations (what each does)

| Operation | CLI command | Description |
|-----------|-------------|-------------|
| Text-to-image (TTI) | `text` | Generate a brand-new image from a text prompt only. |
| Image-to-image (I2I) | `image` | Restyle or vary an existing image, guided by a prompt. The `--strength` parameter controls how much the original is changed (0 = keep original, 1 = ignore it). |
| Canny edge | `canny` | Extract the edge lines from an input image and use them as a structural guide while generating new content. Good for turning sketches into photos. |
| Depth | `depth` | Extract a depth map and use it as a 3-D layout guide. Keeps spatial composition while changing style. |
| Upscale | `upscale` | Enlarge a low-resolution image while adding realistic detail. Eight specialist models are available via `--tool`. |
| Outpaint | `outpaint` | Extend an image's canvas in any direction. The model invents new content that blends with the original. |

---

## LoRA (Low-Rank Adaptation)

A LoRA is a small set of additional weights (~50–200 MB) that *steer* a base
model toward a specific style or subject without retraining the full model.
Think of it as a style preset layered on top of the model.

**Spec format:** `<name_or_url>:<scale>`

- Name from built-in library: `gstdrw style:0.8`
- HuggingFace URL with scale: `https://huggingface.co/path/to/lora.safetensors:0.7`
- Multiple LoRAs (semicolon-separated): `name1:0.5; url2:0.8`

Scale is a float in `[0.0, 1.0]` controlling LoRA influence.

---

## Prompt syntax

`twat-genai` inherits the twat prompt expander, which supports:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `{a,b,c}` | Permutation — generates one image per option | `"a {red,blue} car"` → 2 images |
| `a; b` | Semicolon-separated prompts | generates one image per segment |
| Plain string | Single prompt | one image |

---

## Image size presets

| Preset | Dimensions |
|--------|-----------|
| `SQ` | 1024 × 1024 (square) |
| `HD` | 1280 × 720 (landscape 16:9) |
| `P` | 720 × 1280 (portrait 9:16) |
| `WIDE` | 1440 × 768 (ultra-wide) |

Custom sizes: `--image_size 1920,1080` (comma-separated, no spaces).

---

## Code conventions

### Imports and type hints

- Use `from __future__ import annotations` at the top of every module.
- Prefer union syntax: `str | None` not `Optional[str]`.
- All public functions and methods must have type annotations and a one-line
  docstring minimum.

### Error handling

- Wrap provider failures in `ProviderError` (subclass of `RuntimeError`).
- Wrap FAL job failures in `RuntimeError("Processing {model} job failed: ...")`.
- Never swallow exceptions silently; always log with `logger.error(...)`.

### Async conventions

- FAL operations are `async`; use `asyncio.run()` only at CLI entry points.
- Provider clients (`ChutesClient`, `OpenAIImageClient`, `GeminiImageClient`)
  offer synchronous methods for simpler integration; `ChutesClient` also
  exposes `async` variants (`_apost_json`).

### Testing

- Mock all network calls; tests must pass fully offline.
- Async tests use `pytest-asyncio` with `asyncio_mode = "auto"` (set in
  `pyproject.toml`).
- Fixtures are the single source of truth for mocks — never patch the same
  symbol in both a fixture and a test function.
- Test command: `python -m pytest tests/`

### Adding a new provider

1. Create `src/twat_genai/engines/<provider>/client.py` with a class that
   accepts an injectable `httpx.Client` (for testability).
2. Re-export from `src/twat_genai/engines/<provider>/__init__.py`.
3. Add the class to `src/twat_genai/__init__.py` exports and `__all__`.
4. Add an `EngineConfig`-compatible method or integrate into `TwatGenAiCLI`.
5. Write tests with fully mocked HTTP transport.
