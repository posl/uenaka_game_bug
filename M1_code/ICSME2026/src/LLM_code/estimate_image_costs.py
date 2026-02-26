#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

PATCH_MODELS = {
    "gpt-5-mini": 1.62,
    "gpt-5-nano": 2.46,
    "gpt-4.1-mini": 1.62,
    "gpt-4.1-nano": 2.46,
    "o4-mini": 1.72,
}

DETAIL_MODELS = {
    "gpt-5": (70, 140),
    "gpt-5-chat-latest": (70, 140),
    "gpt-4o": (85, 170),
    "gpt-4.1": (85, 170),
    "gpt-4.5": (85, 170),
    "gpt-4o-mini": (2833, 5667),
    "o1": (75, 150),
    "o1-pro": (75, 150),
    "o3": (75, 150),
    "computer-use-preview": (65, 129),
}

TEXT_INPUT_TOKEN_PRICES_PER_1M = {
    "batch": {
        "gpt-5-mini": 0.125,
        "gpt-5-nano": 0.025,
        "gpt-4.1": 1.00,
        "gpt-4.1-mini": 0.20,
        "gpt-4.1-nano": 0.05,
        "gpt-4o": 1.25,
        "gpt-4o-mini": 0.075,
        "o1": 7.50,
        "o1-pro": 75.00,
        "o3": 1.00,
        "o4-mini": 0.55,
        "computer-use-preview": 1.50,
    },
    "standard": {
        "gpt-5-mini": 0.25,
        "gpt-5-nano": 0.05,
        "gpt-4.1": 2.00,
        "gpt-4.1-mini": 0.40,
        "gpt-4.1-nano": 0.10,
        "gpt-4o": 2.50,
        "gpt-4o-mini": 0.15,
        "o1": 15.00,
        "o1-pro": 150.00,
        "o3": 2.00,
        "o4-mini": 1.10,
        "computer-use-preview": 3.00,
    },
}

TEXT_OUTPUT_TOKEN_PRICES_PER_1M = {
    "batch": {
        "gpt-5-mini": 0.50,
        "gpt-5-nano": 0.10,
        "gpt-4.1": 4.00,
        "gpt-4.1-mini": 0.80,
        "gpt-4.1-nano": 0.20,
        "gpt-4o": 5.00,
        "gpt-4o-mini": 0.30,
        "o1": 30.00,
        "o1-pro": 300.00,
        "o3": 4.00,
        "o4-mini": 2.20,
        "computer-use-preview": 6.00,
    },
    "standard": {
        "gpt-5-mini": 1.00,
        "gpt-5-nano": 0.20,
        "gpt-4.1": 8.00,
        "gpt-4.1-mini": 1.60,
        "gpt-4.1-nano": 0.40,
        "gpt-4o": 10.00,
        "gpt-4o-mini": 0.60,
        "o1": 60.00,
        "o1-pro": 600.00,
        "o3": 8.00,
        "o4-mini": 4.40,
        "computer-use-preview": 12.00,
    },
}

IMAGE_INPUT_TOKEN_PRICES_PER_1M = {
    "batch": {
        "gpt-image-1": 5.00,
    },
    "standard": {
        "gpt-image-1": 10.00,
    },
}


def calculate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    tier: str = "standard",
) -> dict[str, float]:
    model_key = model.lower()
    for suffix in ("", "-2025-04-14", "-2025-06-01"):
        candidate = model_key.removesuffix(suffix)
        if candidate in TEXT_INPUT_TOKEN_PRICES_PER_1M.get(tier, {}):
            model_key = candidate
            break

    input_price = TEXT_INPUT_TOKEN_PRICES_PER_1M.get(tier, {}).get(model_key)
    output_price = TEXT_OUTPUT_TOKEN_PRICES_PER_1M.get(tier, {}).get(model_key)

    input_cost = input_tokens / 1_000_000 * input_price if input_price else 0.0
    output_cost = output_tokens / 1_000_000 * output_price if output_price else 0.0

    return {
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(input_cost + output_cost, 8),
    }


def _patch_token_count(width: int, height: int, max_patches: int = 1536) -> int:
    patch = 32
    raw_patches = math.ceil(width / patch) * math.ceil(height / patch)
    if raw_patches <= max_patches:
        return raw_patches

    r = math.sqrt((patch * patch * max_patches) / (width * height))
    scaled_w = width * r
    scaled_h = height * r
    scaled_w_patches = scaled_w / patch
    scaled_h_patches = scaled_h / patch
    r = r * min(
        math.floor(scaled_w_patches) / scaled_w_patches,
        math.floor(scaled_h_patches) / scaled_h_patches,
    )
    resized_w = math.floor(width * r)
    resized_h = math.floor(height * r)
    return math.ceil(resized_w / patch) * math.ceil(resized_h / patch)


def _scale_to_fit(width: int, height: int, max_side: int) -> tuple[int, int]:
    if width <= max_side and height <= max_side:
        return width, height
    scale = min(max_side / width, max_side / height)
    return max(1, int(math.floor(width * scale))), max(1, int(math.floor(height * scale)))


def _scale_shortest_side(width: int, height: int, target: int) -> tuple[int, int]:
    shortest = min(width, height)
    if shortest == 0:
        return width, height
    scale = target / shortest
    return max(1, int(math.floor(width * scale))), max(1, int(math.floor(height * scale)))


def calculate_image_tokens(
    width: int, height: int, model: str, detail: str = "auto"
) -> int:
    model_key = model.lower()
    if model_key in PATCH_MODELS:
        patches = _patch_token_count(width, height)
        return int(math.ceil(patches * PATCH_MODELS[model_key]))

    if model_key == "gpt-image-1":
        fidelity = "low" if detail == "low" else "high"
        base = 65
        tile = 129
        resized_w, resized_h = _scale_to_fit(width, height, 2048)
        resized_w, resized_h = _scale_shortest_side(resized_w, resized_h, 512)
        tiles = math.ceil(resized_w / 512) * math.ceil(resized_h / 512)
        total = base + tiles * tile
        if fidelity == "high":
            total += 4160 if resized_w == resized_h else 6240
        return total

    if model_key not in DETAIL_MODELS:
        raise ValueError(f"Unsupported model for image token calc: {model}")

    base, tile = DETAIL_MODELS[model_key]
    if detail == "low":
        return base

    resized_w, resized_h = _scale_to_fit(width, height, 2048)
    resized_w, resized_h = _scale_shortest_side(resized_w, resized_h, 768)
    tiles = math.ceil(resized_w / 512) * math.ceil(resized_h / 512)
    return base + tiles * tile


def resolve_detail(detail: str) -> Optional[str]:
    if detail == "auto":
        return "high"
    return detail


def get_image_token_price_per_1m(model: str, tier: str) -> Optional[float]:
    model_key = model.lower()
    if model_key in IMAGE_INPUT_TOKEN_PRICES_PER_1M.get(tier, {}):
        return IMAGE_INPUT_TOKEN_PRICES_PER_1M[tier][model_key]
    return TEXT_INPUT_TOKEN_PRICES_PER_1M.get(tier, {}).get(model_key)


def make_context_frames(
    frames_dir: Path, frame_name: str, window: int = 10
) -> List[str]:
    stem = Path(frame_name).stem
    suffix = Path(frame_name).suffix or ".jpg"
    try:
        center_id = int(stem)
    except ValueError:
        raise ValueError(f"Frame name is not numeric: {frame_name}")

    width = len(stem)
    context = []
    for frame_id in range(center_id - window, center_id + window + 1):
        candidate = frames_dir / f"{frame_id:0{width}d}{suffix}"
        if candidate.exists():
            context.append(str(candidate))
    return context


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate image token costs for TotK frames."
    )
    parser.add_argument("--frames-dir", default="frames", help="Frames directory")
    parser.add_argument(
        "--scenario",
        choices=["all", "oob", "fg"],
        default="all",
        help="Label scenario",
    )
    parser.add_argument("--window", type=int, default=10, help="Context window size")
    parser.add_argument(
        "--image-token-model",
        default="gpt-4.1-mini",
        help="Model name for image token calculation (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default="auto",
        help="Detail level for image token calculation",
    )
    parser.add_argument(
        "--pricing-tier",
        choices=["batch", "standard"],
        default="standard",
        help="Pricing tier for cost estimation",
    )
    args = parser.parse_args()

    print(f"Frames directory: {args.frames_dir}")
    print(f"Scenario: {args.scenario}")
    print(f"Window: {args.window}")
    print(f"Image token model: {args.image_token_model}")
    print(f"Image detail: {args.image_detail}")
    print(f"Pricing tier: {args.pricing_tier}")

    price_per_1m = get_image_token_price_per_1m(
        args.image_token_model, args.pricing_tier
    )
    if price_per_1m is None:
        print(
            f"Warning: No price found for {args.image_token_model} "
            f"({args.pricing_tier} tier). Cost will be omitted."
        )

    input_path = f"balanced_datasets/{args.scenario}_balanced.csv.gz"
    df = pd.read_csv(input_path)
    frames_dir = Path(args.frames_dir)

    if "frame" not in df.columns or "label" not in df.columns:
        raise ValueError("Input dataset must contain 'frame' and 'label' columns.")

    detail = resolve_detail(args.image_detail) or "auto"
    per_image_tokens = calculate_image_tokens(
        FRAME_WIDTH, FRAME_HEIGHT, args.image_token_model, detail
    )

    total_images = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Estimating costs"):
        context_frames = make_context_frames(frames_dir, row["frame"], args.window)
        total_images += len(context_frames)

    total_tokens = per_image_tokens * total_images
    print(f"Records: {len(df)}")
    print(f"Total images: {total_images}")
    print(f"Image tokens per image: {per_image_tokens}")
    print(f"Image tokens total: {total_tokens}")
    if price_per_1m is not None:
        total_cost = round(total_tokens / 1_000_000 * price_per_1m, 8)
        print(f"Image tokens price per 1M: {price_per_1m}")
        print(f"Image tokens cost USD: {total_cost}")


if __name__ == "__main__":
    main()
