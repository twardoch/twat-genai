# this_file: src/twat_genai/__main__.py
"""Fire CLI entry point for twat-genai.

Provides the canonical COMMANDS dict pattern per SPEC §4.2.
Top-level dispatcher: twat-genai
Dashed leaf entries: twat-genai-image, twat-genai-video, etc.
"""

from __future__ import annotations


def _version() -> str:
    """Print the installed package version."""
    from twat_genai.__version__ import __version__

    return __version__


def _image(
    input_image: str,
    prompts: str = "",
    strength: float = 0.75,
    output: str | None = None,
    output_dir: str = "generated_images",
    verbose: bool = False,
    image_size: str = "SQ",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    negative_prompt: str = "",
    lora: str | None = None,
) -> None:
    """Generate images using an existing image as a reference (image-to-image via fal.ai)."""
    import asyncio

    from twat_genai.cli import TwatGenAiCLI

    cli = TwatGenAiCLI(
        output_dir=output_dir,
        verbose=verbose,
        image_size=image_size,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        negative_prompt=negative_prompt,
        lora=lora,
    )
    asyncio.run(cli.image(input_image=input_image, prompts=prompts, strength=strength, output=output))


def _video(
    prompt: str,
    first_frame: str | None = None,
    last_frame: str | None = None,
    resolution: str | None = None,
    fps: int | None = None,
    num_frames: int | None = None,
) -> None:
    """Generate video through Chutes SkyReels or WAN endpoints."""
    from twat_genai.cli import TwatGenAiCLI

    cli = TwatGenAiCLI()
    cli.chutes_video(
        prompt=prompt,
        first_frame=first_frame,
        last_frame=last_frame,
        resolution=resolution,
        fps=fps,
        num_frames=num_frames,
    )


def _list_providers() -> list[str]:
    """List the available AI provider names."""
    from twat_genai.models import ProviderName

    providers = [p.value for p in ProviderName]
    for name in providers:
        print(name)
    return providers


def _list_models() -> list[str]:
    """List all available FAL model identifiers."""
    from twat_genai.engines.fal.config import ModelTypes

    models = [f"{m.name}: {m.value}" for m in ModelTypes]
    for m in models:
        print(m)
    return models


# Explicit allow-list per SPEC §6 — no private internals exposed.
COMMANDS: dict[str, object] = {
    "version": _version,
    "image": _image,
    "video": _video,
    "list-providers": _list_providers,
    "list-models": _list_models,
}


def main() -> None:
    """Top-level Fire dispatcher for twat-genai."""
    import fire  # type: ignore[import-untyped]

    fire.Fire(COMMANDS, name="twat-genai")


# Dashed entry-point helpers — one per leaf.
def cmd_image() -> None:
    """Entry point for twat-genai-image."""
    import fire  # type: ignore[import-untyped]

    fire.Fire(_image, name="twat-genai-image")


def cmd_video() -> None:
    """Entry point for twat-genai-video."""
    import fire  # type: ignore[import-untyped]

    fire.Fire(_video, name="twat-genai-video")


def cmd_list_providers() -> None:
    """Entry point for twat-genai-list-providers."""
    import fire  # type: ignore[import-untyped]

    fire.Fire(_list_providers, name="twat-genai-list-providers")


def cmd_list_models() -> None:
    """Entry point for twat-genai-list-models."""
    import fire  # type: ignore[import-untyped]

    fire.Fire(_list_models, name="twat-genai-list-models")


def cmd_version() -> None:
    """Entry point for twat-genai-version."""
    import fire  # type: ignore[import-untyped]

    fire.Fire(_version, name="twat-genai-version")


if __name__ == "__main__":
    main()
