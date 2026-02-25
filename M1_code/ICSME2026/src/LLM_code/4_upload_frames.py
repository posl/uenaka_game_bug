#!/usr/bin/env python3
import argparse
import csv
import time
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# .env を読み込む
load_dotenv()


def load_existing_mapping(mapping_path: Path) -> dict[str, str]:
    if not mapping_path.exists():
        return {}
    mapping = {}
    with mapping_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["frame"]] = row["file_id"]
    return mapping


def save_mapping(mapping: dict[str, str], mapping_path: Path) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "file_id"])
        writer.writeheader()
        for frame, file_id in sorted(mapping.items()):
            writer.writerow({"frame": frame, "file_id": file_id})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload frames to OpenAI Files API and save frame->file_id mapping."
    )
    parser.add_argument("--frames-dir", default="frames", help="Frames directory")
    parser.add_argument(
        "--mapping",
        default="frame_file_ids.csv",
        help="Output mapping CSV path (default: frame_file_ids.csv)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep between uploads to avoid rate limits (seconds)",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    mapping_path = Path(args.mapping)

    all_frames = sorted(p.name for p in frames_dir.iterdir() if p.suffix == ".jpg")
    print(f"Found {len(all_frames)} frames in {frames_dir}")

    existing = load_existing_mapping(mapping_path)
    print(f"Already uploaded: {len(existing)} frames")

    to_upload = [f for f in all_frames if f not in existing]
    print(f"Remaining to upload: {len(to_upload)} frames")

    if not to_upload:
        print("Nothing to upload.")
        return

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    mapping = dict(existing)

    for frame_name in tqdm(to_upload, desc="Uploading frames"):
        frame_path = frames_dir / frame_name
        with frame_path.open("rb") as f:
            result = client.files.create(file=f, purpose="vision")
        mapping[frame_name] = result.id

        if len(mapping) % 100 == 0:
            save_mapping(mapping, mapping_path)

        if args.sleep > 0:
            time.sleep(args.sleep)

    save_mapping(mapping, mapping_path)
    print(f"Wrote {len(mapping)} entries to {mapping_path}")


if __name__ == "__main__":
    main()
