from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import re
from glob import glob
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_SPLIT_CSV = DATA_DIR / "2_model_training_and_evaluation" / "data_split.csv"

def frame_number_to_random_value(frame_name):
    frame_num = re.sub(r"\D", "", str(frame_name))
    frame_num = int(frame_num)

    h = hashlib.md5(str(frame_num).encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)

    rng = np.random.default_rng(seed)
    return rng.random()


def add_random_output_for_test_data(game):
    df_data_split = pd.read_csv(DATA_SPLIT_CSV)

    key = f"within-title_{game}"

    if key not in df_data_split.columns:
        raise ValueError(f"Column '{key}' not found in data_split.csv")

    test_names = df_data_split.loc[
        df_data_split[key] == "test", "Data_name"
    ].tolist()

    if len(test_names) == 0:
        print("No test data found.")
        return

    for name in test_names:
        if not os.path.isdir(DATASET_DIR / name):
            print(f"[Skip] Directory not found: {name}")
            continue

        csv_paths = glob(os.path.join(DATASET_DIR, name, "frame_label*.csv"))

        if len(csv_paths) == 0:
            print(f"[Skip] No frame_label CSV in {name}")
            continue

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)

            if "frame" not in df.columns:
                print(f"[Skip] 'frame' column not found: {csv_path}")
                continue

            df["random_output"] = df["frame"].apply(
                frame_number_to_random_value
            )

            df.to_csv(csv_path, index=False)
            print(f"[Updated] {csv_path}")

if __name__ == "__main__":
    games = ["SM64", "BotW", "TotK"]

    for game in games:
        add_random_output_for_test_data(game)
