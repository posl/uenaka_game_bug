#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from estimate_image_costs import make_context_frames


def load_file_id_mapping(mapping_path: Path) -> dict[str, str]:
    mapping = {}
    with mapping_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["frame"]] = row["file_id"]
    return mapping


def build_system_prompt(scenario: str) -> str:
    scenario_text = {
        "all": "any bug (regardless of category)",
        "oob": "OoB bug",
        "fg": "FG bug",
    }[scenario]
    return (
        "You are a helpful assistant analyzing video game images and screenshots for glitches.\n"
        "You are given a target frame and its surrounding context frames \n"
        "(from -10 to +10 relative to the target).\n"
        f"Decide whether the target frame contains {scenario_text}.\n"
        "Answer with 1 for bug, 0 for non-bug."
    )


def build_content(
    center_frame: str,
    context_frames: list[str],
    frames_dir: Path,
    window: int,
    file_id_map: dict[str, str],
) -> list[dict]:
    stem = Path(center_frame).stem
    suffix = Path(center_frame).suffix or ".jpg"
    center_id = int(stem)
    digit_width = len(stem)

    content: list[dict] = []

    content.append({"type": "input_text", "text": f"Target frame (0): {center_frame}"})
    content.append({"type": "input_image", "file_id": file_id_map[center_frame]})

    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        frame_id = center_id + offset
        candidate = f"{frame_id:0{digit_width}d}{suffix}"
        candidate_path = str(frames_dir / candidate)
        if candidate_path in context_frames:
            sign = "+" if offset > 0 else ""
            content.append(
                {"type": "input_text", "text": f"Context frame ({sign}{offset}): {candidate}"}
            )
            content.append({"type": "input_image", "file_id": file_id_map[candidate]})

    return content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-image prompts in Responses API format with context frames."
    )
    parser.add_argument("--frames-dir", default="frames", help="Frames directory")
    parser.add_argument(
        "--scenario",
        choices=["all", "oob", "fg"],
        default="all",
        help="Label scenario",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: prompts/<scenario>_vision.jsonl)",
    )
    parser.add_argument("--window", type=int, default=10, help="Context window size")
    parser.add_argument(
        "--mapping",
        default="frame_file_ids.csv",
        help="Path to frame->file_id mapping CSV",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model name for prompt body (default: gpt-4.1-mini)",
    )
    args = parser.parse_args()
    # argsをまとめて表示
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_path}\n"
            "Run upload_frames.py first to upload frames and generate the mapping."
        )
    file_id_map = load_file_id_mapping(mapping_path)
    print(f"Loaded {len(file_id_map)} file_id mappings from {mapping_path}")

    input_path = Path("balanced_datasets") / f"{args.scenario}_balanced.csv.gz"
    df = pd.read_csv(input_path)
    frames_dir = Path(args.frames_dir)

    if "frame" not in df.columns or "label" not in df.columns:
        raise ValueError("Input dataset must contain 'frame' and 'label' columns.")

    out_path = (
        Path(args.out)
        if args.out
        else Path("prompts") / f"{args.scenario}_vision.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = build_system_prompt(args.scenario)

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building prompts"):
            frame_name = row["frame"]
            label = int(row["label"])
            context_frames = make_context_frames(frames_dir, frame_name, args.window)

            content = build_content(
                frame_name, context_frames, frames_dir, args.window, file_id_map
            )
            print(f"content: {content}")

            record = {
                "frame": frame_name,
                "label": label,
                "scenario": args.scenario,
                "context_frames_count": len(context_frames),
                "request": {
                    "model": args.model,
                    "input": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": content,
                        },
                    ],
                },
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(df)} records to {out_path}")


if __name__ == "__main__":
    main()
