from pathlib import Path
import pandas as pd
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
SEC_INFO_PATH = DATA_DIR / "1_dataset_construction" / "sec_info.csv"

def extract_frames(video_path: Path,
                   output_dir: Path,
                   first_frame_number: int,
                   fps: int = 30):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))

    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            frame_id = first_frame_number + saved
            frame_file = output_dir / f"{frame_id}.jpg"

            cv2.imwrite(str(frame_file), frame)
            saved += 1

        idx += 1

    cap.release()

    print(f"[INFO] Extracted {saved} frames -> {output_dir}")

def main():

    if not SEC_INFO_PATH.exists():
        raise FileNotFoundError(f"{SEC_INFO_PATH} not found")

    df = pd.read_csv(SEC_INFO_PATH)

    for _, row in df.iterrows():

        data_name = row["Data_name"].strip()
        data_dir = DATASET_DIR / data_name

        if not data_dir.exists():
            print(f"[WARN] {data_dir} not found, skip")
            continue

        first_frames = [
            int(x.strip())
            for x in str(row["First_frame_number"]).split(",")
            if x.strip()
        ]

        for i, first_frame_number in enumerate(first_frames, start=1):

            video_path = data_dir / f"video{i}.mp4"

            if not video_path.exists():
                print(f"[WARN] {video_path} not found, skip")
                continue

            output_dir = data_dir / f"frames{i}"
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[INFO] Processing {video_path}")
            print(f"       First frame number = {first_frame_number}")

            extract_frames(
                video_path=video_path,
                output_dir=output_dir,
                first_frame_number=first_frame_number,
                fps=30
            )

if __name__ == "__main__":
    main()
