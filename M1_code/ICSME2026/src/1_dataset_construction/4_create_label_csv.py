from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
SEC_INFO_PATH = DATA_DIR / "1_dataset_construction" / "sec_info.csv"

def main():

    if not SEC_INFO_PATH.exists():
        raise FileNotFoundError(f"{SEC_INFO_PATH} not found")

    df = pd.read_csv(SEC_INFO_PATH)

    label_columns = [
        "All_label",
        "OoB_label",
        "SG_label",
        "FG_label",
        "IG_label",
        "DB_label",
        "TG_label",
        "WD_label",
    ]

    for _, row in df.iterrows():

        data_name = row["Data_name"].strip()
        data_dir = DATASET_DIR / data_name

        if not data_dir.exists():
            print(f"[WARN] {data_dir} not found, skip")
            continue

        start_seconds = [
            float(x.strip())
            for x in str(row["Start_second"]).split(",")
            if x.strip()
        ]

        end_seconds = [
            float(x.strip())
            for x in str(row["End_second"]).split(",")
            if x.strip()
        ]

        first_frames = [
            int(x.strip())
            for x in str(row["First_frame_number"]).split(",")
            if x.strip()
        ]

        # 個数チェック
        n = len(first_frames)

        if not (len(start_seconds) == len(end_seconds) == n):
            print(f"[ERROR] Mismatch in counts for {data_name}")
            continue

        for idx in range(n):

            start_sec = start_seconds[idx]
            end_sec = end_seconds[idx]
            first_frame = first_frames[idx]

            num_frames = int((end_sec - start_sec) * 30)

            frames = [
                f"{first_frame + i}.jpg"
                for i in range(num_frames)
            ]

            out_df = pd.DataFrame({"frame": frames})

            for col in label_columns:
                out_df[col] = 0

            out_path = data_dir / f"frame_label{idx+1}.csv"
            out_df.to_csv(out_path, index=False)

            print(f"[INFO] Created label CSV: {out_path}")

if __name__ == "__main__":
    main()
