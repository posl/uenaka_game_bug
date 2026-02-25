#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from estimate_image_costs import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    calculate_image_tokens,
    get_image_token_price_per_1m,
    make_context_frames,
    resolve_detail,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize total image costs per scenario."
    )
    parser.add_argument("--frames-dir", default="frames", help="Frames directory")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all", "oob", "fg"],
        help="Scenarios to include",
    )
    parser.add_argument(
        "--out",
        default="prompts/costs_summary.csv",
        help="Output CSV path",
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

    detail = resolve_detail(args.image_detail) or "auto"
    per_image_tokens = calculate_image_tokens(
        FRAME_WIDTH, FRAME_HEIGHT, args.image_token_model, detail
    )
    price_per_1m = get_image_token_price_per_1m(
        args.image_token_model, args.pricing_tier
    )

    frames_dir = Path(args.frames_dir)
    rows = []

    for scenario in args.scenarios:
        input_path = Path("balanced_datasets") / f"{scenario}_balanced.csv.gz"
        df = pd.read_csv(input_path)
        if "frame" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"Input dataset for {scenario} must contain 'frame' and 'label'."
            )

        total_images = 0
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Counting frames ({scenario})",
        ):
            context_frames = make_context_frames(frames_dir, row["frame"], args.window)
            total_images += len(context_frames)

        total_tokens = per_image_tokens * total_images
        total_cost = None
        if price_per_1m is not None:
            total_cost = round(total_tokens / 1_000_000 * price_per_1m, 8)

        rows.append(
            {
                "scenario": scenario,
                "records": len(df),
                "total_images": total_images,
                "image_tokens_per_image": per_image_tokens,
                "image_tokens_total": total_tokens,
                "image_tokens_price_per_1m": price_per_1m,
                "image_tokens_cost_usd": total_cost,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
