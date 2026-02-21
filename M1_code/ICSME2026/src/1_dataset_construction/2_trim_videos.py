import csv
from pathlib import Path
import ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
CSV_PATH = DATA_DIR / "1_dataset_construction" / "sec_info.csv"


def parse_multi_value(cell: str) -> list[str]:
    """
    "a, b, c" -> ["a", "b", "c"]
    """
    return [x.strip() for x in cell.split(",")]


def trim_video(
    input_path: Path,
    start: int,
    end: int,
    output_path: Path,
):
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid time range: {start} - {end}")

    (
        ffmpeg
        .input(str(input_path), ss=start)
        .output(str(output_path), t=duration, c="copy")
        .run(overwrite_output=True)
    )


def main():
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            data_name = row["Data_name"].strip()

            paths = parse_multi_value(row["Path"])
            starts = parse_multi_value(row["Start_second"])
            ends = parse_multi_value(row["End_second"])

            if not (len(paths) == len(starts) == len(ends)):
                raise ValueError(
                    f"Length mismatch in row {data_name}: "
                    f"{len(paths)}, {len(starts)}, {len(ends)}"
                )

            output_dir = DATASET_DIR / data_name
            output_dir.mkdir(parents=True, exist_ok=True)

            for idx, (path, start, end) in enumerate(zip(paths, starts, ends)):
                input_video = DATASET_DIR / path
                if not input_video.exists():
                    raise FileNotFoundError(input_video)

                output_video = output_dir / f"video{idx}.mp4"

                trim_video(
                    input_path=input_video,
                    start=int(start),
                    end=int(end),
                    output_path=output_video,
                )

                print(
                    f"[OK] {data_name}: {path} "
                    f"({start}-{end}) -> {output_video}"
                )

if __name__ == "__main__":
    main()
