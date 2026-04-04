import csv
from pathlib import Path
import ffmpeg
from module.module import parse_multi_value

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "code" / "urls.csv"

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
            category = row["Category"].strip()

            starts = parse_multi_value(row["Start_sec"])
            ends = parse_multi_value(row["End_sec"])

            if not (len(starts) == len(ends)):
                raise ValueError(
                    f"Length mismatch in row {category}: "
                    f"{len(starts)}, {len(ends)}"
                )

            output_dir = PROJECT_ROOT / category
            output_dir.mkdir(parents=True, exist_ok=True)

            for idx, (start, end) in enumerate(zip(starts, ends), start=1):
                input_video = output_dir / f"video{idx}.mp4"
                if not input_video.exists():
                    raise FileNotFoundError(input_video)

                output_video = output_dir / f"trimmed_video{idx}.mp4"

                trim_video(
                    input_path=input_video,
                    start=int(start),
                    end=int(end),
                    output_path=output_video,
                )

                print(
                    f"[OK] {category}: {input_video} "
                    f"({start}-{end}) -> {output_video}"
                )

if __name__ == "__main__":
    main()
