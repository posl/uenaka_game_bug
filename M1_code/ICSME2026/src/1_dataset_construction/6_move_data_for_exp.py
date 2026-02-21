import shutil
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"

MOVE_CSV = DATA_DIR / "move_data.csv"

def get_label_csv_path(frame_dir: str) -> Path:
    data_name, frame_name = frame_dir.split("/")
    label_name = frame_name.replace("frame", "frame_label") + ".csv"
    return DATASET_DIR / data_name / label_name

def main():
    df = pd.read_csv(MOVE_CSV)

    for _, row in df.iterrows():
        from_dir = row["From"]
        to_dir = row["To"]

        start_frame = int(row["Start_frame_number"])
        end_frame = int(row["End_frame_number"])

        print(f"\n[Process] {from_dir} -> {to_dir} ({start_frame} - {end_frame})")

        from_path = DATASET_DIR / from_dir
        to_path = DATASET_DIR / to_dir

        to_path.mkdir(parents=True, exist_ok=True)

        for frame_num in range(start_frame, end_frame + 1):
            img_name = f"{frame_num}.jpg"
            src_img = from_path / img_name
            dst_img = to_path / img_name

            if not src_img.exists():
                print(f"[Warning] Missing image: {src_img}")
                continue

            shutil.move(str(src_img), str(dst_img))

        print("[OK] Images moved.")

        from_label_csv = get_label_csv_path(from_dir)
        to_label_csv = get_label_csv_path(to_dir)

        if not from_label_csv.exists():
            print(f"[Warning] Label CSV not found: {from_label_csv}")
            continue

        label_df = pd.read_csv(from_label_csv)

        start_name = f"{start_frame}.jpg"
        end_name = f"{end_frame}.jpg"

        mask = label_df["frame"].between(start_name, end_name)

        move_rows = label_df[mask].copy()

        label_df = label_df[~mask]

        label_df.to_csv(from_label_csv, index=False)

        if to_label_csv.exists():
            to_df = pd.read_csv(to_label_csv)
            combined = pd.concat([to_df, move_rows], ignore_index=True)
        else:
            combined = move_rows

        combined.to_csv(to_label_csv, index=False)

        print("[OK] Label CSV rows moved.")
        print(f"    Source updated: {from_label_csv}")
        print(f"    Destination updated: {to_label_csv}")

    print("\nDone! All data moved successfully.")


if __name__ == "__main__":
    main()
