#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fal-client", "fire", "python-dotenv", "httpx", "Pillow", "pydantic", "python-slugify", "rich", "loguru"]
# ///
"""
Script for asynchronous image generation using fal-ai.
Dependencies: fal-client, fire, python-dotenv, httpx, Pillow, pydantic, python-slugify, rich, loguru
"""

import asyncio
import sys
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import fal_client
import fire
import httpx
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from pydantic import BaseModel, RootModel
from slugify import slugify

from .cli import cli

load_dotenv()

# --- Enums and Models ---


class ModelTypes(str, Enum):
    """Available model types."""

    TEXT = "fal-ai/flux-lora"
    IMAGE = "fal-ai/flux-lora/image-to-image"
    CANNY = "fal-ai/flux-lora-canny"
    DEPTH = "fal-ai/flux-lora-depth"


class ImageInput(BaseModel):
    """Represents an image input that can be a URL, file path, or PIL Image."""

    url: Optional[str] = None
    path: Optional[Path] = None
    pil_image: Optional[Image.Image] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_valid(self) -> bool:
        """Check if exactly one input type is provided."""
        return (
            sum(1 for x in (self.url, self.path, self.pil_image) if x is not None) == 1
        )

    async def to_url(self) -> str:
        """Convert the image input to a URL that can be used by fal.ai."""
        if not self.is_valid:
            raise ValueError("Exactly one of url, path, or pil_image must be provided")
        if self.url:
            return self.url
        if self.path:
            return await fal_client.upload_file_async(Path(str(self.path)))
        if self.pil_image:
            with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                tmp_path = Path(tmp.name)
                self.pil_image.save(tmp_path, format="JPEG", quality=95)
                return await fal_client.upload_file_async(tmp_path)
        raise ValueError("No valid image input provided")


class ImageToImageConfig(BaseModel):
    """Configuration for image-to-image operations."""

    model_type: ModelTypes
    input_image: ImageInput
    strength: float = 0.75  # Only used for standard image-to-image
    negative_prompt: str = ""


class ImageSizes(str, Enum):
    SQ = "square_hd"
    SQL = "square"
    SDV = "portrait_4_3"
    HDV = "portrait_16_9"
    SD = "landscape_4_3"
    HD = "landscape_16_9"


class ImageFormats(str, Enum):
    JPG = "jpeg"
    PNG = "png"
    PIL = "pil"


class ImageSizeWH(BaseModel):
    """Width and height for a custom image size."""

    width: int
    height: int


# Type aliases
Prompts = list[str]
FALModel = str
FALLoras = Optional[list[str]]
ImageSize = Union[ImageSizes, ImageSizeWH]
OutputDir = Optional[Path]
GuidanceScale = float
NumInferenceSteps = int
ImageFormat = ImageFormats
URLStr = str
RequestID = str
JsonDict = dict[str, Any]


# --- New Lora Models ---


class LoraRecord(BaseModel):
    """Represents a Lora record with its URL and scale."""

    url: str
    scale: float = 1.0


class LoraRecordList(RootModel[list[LoraRecord]]):
    """A list of LoraRecord models."""

    pass


class LoraLib(RootModel[dict[str, LoraRecordList]]):
    """A dictionary where the key is the prompt portion and the value is a LoraRecordList."""

    pass


# --- Global Lora Library ---
LORA_LIB_PATH = Path(__file__).parent / f"{Path(__file__).stem}_loras.json"
LORA_LIB = LoraLib.model_validate_json(LORA_LIB_PATH.read_text())


# --- Other Models ---


class ImageResult(BaseModel):
    """
    The result of a single image generation.
    Contains FAL request information, the raw API response, image metadata, and the original job parameters.
    """

    request_id: str
    timestamp: str
    result: JsonDict
    image_info: dict[str, Any]
    image: Optional[Image.Image] = None
    original_prompt: Optional[str] = None
    job_params: Optional[dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}


# --- Lora Spec Models and Helpers ---


class LoraSpecEntry(BaseModel):
    """Single Lora specification for inference."""

    path: str
    scale: float = 1.0
    prompt: str = ""


class CombinedLoraSpecEntry(BaseModel):
    """Combined specification composed of multiple LoraSpecEntry items."""

    entries: list[Union[LoraSpecEntry, "CombinedLoraSpecEntry"]]
    factory_key: Optional[str] = None  # Store the factory key if applicable


def parse_lora_phrase(phrase: str) -> Union[LoraSpecEntry, CombinedLoraSpecEntry]:
    """
    Parse a Lora phrase which may include an optional scale.
    A phrase is either:
      - A key from LORA_LIB (the prompt portion) or
      - A URL/path optionally followed by a colon and a numeric scale.
    """
    phrase = phrase.strip()
    if phrase in LORA_LIB.root:
        entries = []
        for record in LORA_LIB.root[phrase].root:
            entries.append(
                LoraSpecEntry(path=record.url, scale=record.scale, prompt=phrase + ",")
            )
        return CombinedLoraSpecEntry(entries=entries, factory_key=phrase)
    if ":" in phrase:
        identifier, scale_str = phrase.split(":", 1)
        identifier = identifier.strip()
        try:
            scale = float(scale_str.strip())
        except ValueError:
            raise ValueError(f"Invalid scale value in Lora phrase: {phrase}")
    else:
        identifier = phrase
        scale = 1.0
    return LoraSpecEntry(path=identifier, scale=scale, prompt="")


def normalize_lora_spec(
    spec: Union[str, list, tuple, None],
) -> list[Union[LoraSpecEntry, CombinedLoraSpecEntry]]:
    """
    Normalize various Lora specification formats into a unified list.
    Supported formats:
      - None: returns an empty list.
      - A string:
        - If it's a key in LORA_LIB, treated as a single predefined configuration
        - If it contains semicolons, interpreted as multiple "url:scale" specifications
      - A list or tuple: each element can be a dict, a string, or a list of strings (combined spec).
    """
    if spec is None:
        return []
    normalized: list[Union[LoraSpecEntry, CombinedLoraSpecEntry]] = []
    match spec:
        case list() | tuple() as items:
            for item in items:
                match item:
                    case dict(path=path, scale=scale, prompt=prompt):
                        normalized.append(
                            LoraSpecEntry(path=path, scale=float(scale), prompt=prompt)
                        )
                    case dict() as d:
                        if "path" not in d:
                            raise ValueError("Lora spec dictionary must have a 'path'.")
                        normalized.append(
                            LoraSpecEntry(
                                path=d["path"],
                                scale=float(d.get("scale", 1.0)),
                                prompt=d.get("prompt", ""),
                            )
                        )
                    case str() as phrase:
                        normalized.append(parse_lora_phrase(phrase))
                    case list() as sublist:
                        combined = [
                            parse_lora_phrase(sub)
                            for sub in sublist
                            if isinstance(sub, str)
                        ]
                        normalized.append(CombinedLoraSpecEntry(entries=combined))
                    case _:
                        raise ValueError(
                            f"Unsupported Lora spec item type: {type(item)}"
                        )
        case str() as s:
            if s in LORA_LIB.root:
                return [parse_lora_phrase(s)]
            phrases = [phrase.strip() for phrase in s.split(";") if phrase.strip()]
            return [parse_lora_phrase(phrase) for phrase in phrases]
        case _:
            raise ValueError(f"Unsupported Lora spec type: {type(spec)}")
    return normalized


def build_lora_arguments(
    lora_spec: Union[str, list, tuple, None], prompt: str
) -> tuple[list[dict[str, Any]], str]:
    """
    Build the list of inference Lora dictionaries and a final prompt.
    If any entry contains a prompt prefix, it is prepended.
    Combined specs are flattened.
    """
    entries = normalize_lora_spec(lora_spec)
    lora_list: list[dict[str, Any]] = []
    prompt_prefixes: list[str] = []

    def process_entry(entry: Union[LoraSpecEntry, CombinedLoraSpecEntry]) -> None:
        if isinstance(entry, LoraSpecEntry):
            lora_list.append({"path": entry.path, "scale": entry.scale})
            if entry.prompt:
                prompt_prefixes.append(entry.prompt.rstrip(","))
        else:
            for sub_entry in entry.entries:
                process_entry(sub_entry)

    for entry in entries:
        process_entry(entry)

    final_prompt = (
        f"{', '.join(prompt_prefixes)}, {prompt}".strip() if prompt_prefixes else prompt
    )
    return lora_list, final_prompt


# --- Prompt Expansion Helpers ---


def split_top_level(s: str, delimiter: str = ";") -> list[str]:
    """
    Split a string by the given delimiter only at the top level (i.e. not inside braces).
    """
    parts = []
    buf = []
    depth = 0
    for char in s:
        if char == "{":
            depth += 1
        elif char == "}" and depth:
            depth -= 1
        if char == delimiter and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return parts


def expand_prompts(s: str) -> list[str]:
    """
    Recursively expand a prompt string with alternatives inside braces.
    """
    open_index = s.find("{")
    if open_index == -1:
        return [s]
    depth = 0
    close_index = -1
    for i, char in enumerate(s[open_index:], start=open_index):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                close_index = i
                break
    if close_index == -1:
        raise ValueError("Unbalanced braces in prompt string.")
    prefix = s[:open_index]
    brace_content = s[open_index + 1 : close_index]
    suffix = s[close_index + 1 :]
    alternatives = split_top_level(brace_content, delimiter=";")
    expanded = []
    for alt in alternatives:
        for alt_exp in expand_prompts(alt):
            for suffix_exp in expand_prompts(suffix):
                expanded.append(f"{prefix}{alt_exp}{suffix_exp}")
    return expanded


# --- Job Configuration and Submission ---


class TTIJobConfig(BaseModel):
    """
    Configuration for an image generation job.
    Supports both text-to-image and image-to-image operations.
    """

    prompt: str
    original_prompt: str  # Store the original user prompt before Lora modifications
    model: ModelTypes = ModelTypes.TEXT
    lora_spec: Union[str, list, tuple, None] = None
    output_dir: OutputDir
    image_size: ImageSize
    guidance_scale: GuidanceScale
    num_inference_steps: NumInferenceSteps
    filename_suffix: Optional[str] = None
    filename_prefix: Optional[str] = None
    image_config: Optional[ImageToImageConfig] = (
        None  # Only for image-to-image operations
    )

    async def to_fal_arguments(self) -> dict[str, Any]:
        """Convert the job config into a dictionary for fal-client."""
        size_value = (
            self.image_size.value
            if isinstance(self.image_size, ImageSizes)
            else self.image_size
        )
        lora_list, final_prompt = build_lora_arguments(self.lora_spec, self.prompt)
        logger.debug(f"Using Lora configuration: {lora_list}")
        # Adjust guidance_scale for CANNY model
        guidance_scale = self.guidance_scale
        if self.model == ModelTypes.CANNY:
            guidance_scale = max(self.guidance_scale, 20.0)
        args = {
            "loras": lora_list,
            "prompt": final_prompt,
            "image_size": size_value,
            "num_images": 1,
            "output_format": "jpeg",
            "guidance_scale": guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": False,
        }
        if self.model != ModelTypes.TEXT and self.image_config:
            image_url = await self.image_config.input_image.to_url()
            args["image_url"] = image_url
            if self.model == ModelTypes.IMAGE:
                args["strength"] = self.image_config.strength
            if self.image_config.negative_prompt:
                args["negative_prompt"] = self.image_config.negative_prompt
        return args


async def submit_image_job(job: TTIJobConfig) -> RequestID:
    """
    Submit an asynchronous image generation job using the provided configuration.
    Returns:
        The FAL request ID.
    """
    args = await job.to_fal_arguments()
    logger.debug(f"Submitting job with lora_list: {args.get('loras')}")
    handler = await fal_client.submit_async(job.model.value, arguments=args)
    return handler.request_id


async def download_image(url: URLStr, output_path: Path) -> None:
    """
    Download an image from a URL and save it to disk.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        logger.info(f"Saved image to: {output_path}")


async def get_result(
    request_id: RequestID,
    output_dir: OutputDir = None,
    filename_suffix: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    original_prompt: Optional[str] = None,
    job_params: Optional[dict[str, Any]] = None,
) -> ImageResult:
    """
    Retrieve and process the result of a submitted job.
    Saves the image and metadata as soon as they're available.
    """
    status = await fal_client.status_async(
        "fal-ai/flux-lora", request_id, with_logs=True
    )
    while isinstance(status, fal_client.InProgress):
        await asyncio.sleep(1)
        status = await fal_client.status_async(
            "fal-ai/flux-lora", request_id, with_logs=True
        )
    result = await fal_client.result_async("fal-ai/flux-lora", request_id)
    timestamp = result.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    image_info = result["images"][0]
    content_type = image_info.get("content_type", "image/jpeg")
    extension = "jpg" if content_type == "image/jpeg" else "png"
    image_url = image_info["url"]
    image_path = None
    metadata_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        request_id_prefix = request_id.split("-")[0] if request_id else ""
        if not filename_prefix and original_prompt:
            words = original_prompt.split()
            filename_prefix = (
                "_".join(words[:2]).lower() + "_"
                if len(words) >= 2
                else (words[0] + "_")
                if words
                else "image_"
            )
        stem = f"{filename_prefix or 'image_'}{timestamp}-{request_id_prefix}"
        if filename_suffix:
            stem = f"{stem}-{filename_suffix}"
        filename = slugify(stem) + f".{extension}"
        image_path = output_dir / filename
        try:
            await download_image(image_url, image_path)
        except Exception as e:
            logger.error(f"Failed to save image {image_path}: {e}")
            image_path = None
        try:
            metadata_path = output_dir / f"{slugify(stem)}.json"
            metadata = ImageResult(
                request_id=request_id,
                timestamp=timestamp,
                result=result,
                image_info={
                    "path": str(image_path) if image_path else None,
                    "url": image_url,
                    "index": 0,
                    "suffix": filename_suffix,
                    "metadata_path": str(metadata_path) if metadata_path else None,
                    "original_prompt": original_prompt,
                },
                image=None,
                original_prompt=original_prompt,
                job_params=job_params,
            )
            metadata_path.write_text(metadata.model_dump_json(indent=2))
            logger.info(f"Saved metadata to: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata {metadata_path}: {e}")
            metadata_path = None
    return ImageResult(
        request_id=request_id,
        timestamp=timestamp,
        result=result,
        image_info={
            "path": str(image_path) if image_path else None,
            "url": image_url,
            "index": 0,
            "suffix": filename_suffix,
            "metadata_path": str(metadata_path) if metadata_path else None,
            "original_prompt": original_prompt,
        },
        image=None,
        original_prompt=original_prompt,
        job_params=job_params,
    )


async def process_single_job(job: TTIJobConfig) -> ImageResult:
    """
    Process a single image generation job.
    Handles submission, waiting for completion, and saving results.
    """
    request_id = await submit_image_job(job)
    logger.info(f"Submitted job with ID: {request_id}")
    suffix_parts = []
    if job.filename_suffix:
        suffix_parts.append(job.filename_suffix)
    else:
        if isinstance(job.lora_spec, str):
            suffix_parts.append(job.lora_spec)
        elif (
            isinstance(job.lora_spec, CombinedLoraSpecEntry)
            and job.lora_spec.factory_key
        ):
            suffix_parts.append(slugify(job.lora_spec.factory_key)[:8])
    combined_suffix = "_".join(suffix_parts) if suffix_parts else None

    # Collect job parameters for metadata
    job_params = {
        "prompt": job.prompt,
        "original_prompt": job.original_prompt,
        "model": job.model.value,
        "lora_spec": job.lora_spec,
        "image_size": job.image_size.value
        if isinstance(job.image_size, Enum)
        else job.image_size,
        "guidance_scale": job.guidance_scale,
        "num_inference_steps": job.num_inference_steps,
    }

    return await get_result(
        request_id,
        job.output_dir,
        combined_suffix,
        job.filename_prefix,
        job.original_prompt,
        job_params,
    )


async def async_main(
    prompts: Union[str, list[str]],
    output_dir: Union[str, Path] = "generated_images",
    filename_suffix: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    model: ModelTypes = ModelTypes.TEXT,
    image_config: Optional[ImageToImageConfig] = None,
    lora: Union[str, list, None] = None,
    image_size: str = "SQ",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
) -> list[ImageResult]:
    """
    Generate images concurrently using fal-ai models.
    Supports both text-to-image and image-to-image operations.
    Args:
        prompts: Text prompts for generation
        output_dir: Directory to save generated images
        filename_suffix: Optional suffix for generated filenames
        filename_prefix: Optional prefix for generated filenames
        model: Model type to use
        image_config: Configuration for image-to-image operations
        lora: LoRA configuration
        image_size: Size of the output image
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
    """
    if model != ModelTypes.TEXT:
        if not image_config:
            raise ValueError("image_config is required for image-to-image operations")
        if image_config.model_type != model:
            raise ValueError("image_config.model_type must match the model parameter")
    if isinstance(prompts, str):
        raw_prompts = split_top_level(prompts, delimiter=";")
    else:
        raw_prompts = prompts
    final_prompts: list[str] = []
    for raw in raw_prompts:
        final_prompts.extend(expand_prompts(raw.strip()))
    logger.debug(f"Expanded prompts: {final_prompts}")
    default_suffix = None
    if lora and isinstance(lora, str) and lora in LORA_LIB.root and not filename_suffix:
        default_suffix = slugify(lora)[:8]
    expanded_lora_specs = [lora] if lora is not None else [None]
    try:
        size: ImageSize = ImageSizes[image_size.upper()]
    except KeyError:
        if "," in image_size:
            try:
                w, h = (int(x.strip()) for x in image_size.split(",", 1))
                size = ImageSizeWH(width=w, height=h)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    "For custom image sizes use 'width,height' with integers."
                ) from err
        else:
            valid_names = ", ".join(s.name for s in ImageSizes)
            raise ValueError(
                f"image_size must be one of: {valid_names} or in 'width,height' format."
            )
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    jobs: list[TTIJobConfig] = []
    for prompt in final_prompts:
        for spec in expanded_lora_specs:
            jobs.append(
                TTIJobConfig(
                    prompt=prompt,
                    original_prompt=prompt,
                    model=model,
                    lora_spec=spec,
                    output_dir=output_dir_path,
                    filename_suffix=filename_suffix or default_suffix,
                    filename_prefix=filename_prefix,
                    image_size=size,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    image_config=image_config,
                )
            )
    return await asyncio.gather(*(process_single_job(job) for job in jobs))


if __name__ == "__main__":
    fire.Fire(cli)
