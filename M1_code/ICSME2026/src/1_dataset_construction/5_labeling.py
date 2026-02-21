import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
AUTHOR_CSV = DATA_DIR / "author_annotation_results.csv"

BUG_TYPES = ["OoB", "SG", "FG", "IG", "DB", "TG", "WD"]

def main():
    author_df = pd.read_csv(AUTHOR_CSV)

    # Process each row
    for _, row in author_df.iterrows():
        bug_file = row["Bug_file"]
        start_frame = int(row["Start_frame_number"])
        end_frame = int(row["End_frame_number"])

        csv_path = DATASET_DIR / bug_file

        if not csv_path.exists():
            print(f"[Warning] File not found: {csv_path}")
            continue

        label_df = pd.read_csv(csv_path)

        start_name = f"{start_frame}.jpg"
        end_name = f"{end_frame}.jpg"

        mask = label_df["frame"].between(start_name, end_name)

        label_df.loc[mask, "All_label"] = 1

        for bug in BUG_TYPES:
            if int(row[bug]) == 1:
                col_name = f"{bug}_label"
                label_df.loc[mask, col_name] = 1

        label_df.to_csv(csv_path, index=False)
        print(f"[Updated] {csv_path} ({start_name} - {end_name})")

    print("\nDone! All labeling completed.")


if __name__ == "__main__":
    main()
