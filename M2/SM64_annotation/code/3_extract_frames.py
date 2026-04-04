from pathlib import Path
import pandas as pd
import cv2
from module.module import parse_multi_value

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "code" / "urls.csv"

def extract_frames(video_path: Path, output_dir: Path, start: int,
                   duration_sec: int, fps: int = 30):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    saved = 0
    target_times = [i / fps for i in range(duration_sec * fps)]

    for t in target_times:
        frame_no = int(t * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = output_dir / f"{start + saved}.jpg"
        cv2.imwrite(str(frame_file), frame)
        saved += 1

    cap.release()
    print(f"[INFO] Extracted {saved} frames -> {output_dir}")

def main():

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found")

    df = pd.read_csv(CSV_PATH)

    for _, row in df.iterrows():

        category = row["Category"].strip()
        data_dir = PROJECT_ROOT / category

        if not data_dir.exists():
            print(f"[WARN] {data_dir} not found, skip")
            continue

        starts = parse_multi_value(row["Start_sec"])
        ends = parse_multi_value(row["End_sec"])

        if len(starts) != len(ends):
            raise ValueError(
                f"[{category}] len(starts)={len(starts)} != len(ends)={len(ends)}"
            )

        for i, (start, end) in enumerate(zip(starts, ends), start=1):

            video_path = data_dir / f"trimmed_video{i}.mp4"

            if not video_path.exists():
                print(f"[WARN] {video_path} not found, skip")
                continue

            output_dir = data_dir / f"frames{i}"
            output_dir.mkdir(parents=True, exist_ok=True)

            duration_sec = int(end) - int(start)
            print(f"\n[INFO] Processing {video_path}")
            print(f"       First frame number = {start}, duration = {duration_sec} sec")

            extract_frames(
                video_path=video_path,
                output_dir=output_dir,
                start=0,
                duration_sec=duration_sec,
                fps=30
            )

if __name__ == "__main__":
    main()
