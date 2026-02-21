from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

def delete_game_dirs():
    delete_dirs = ["SM64", "BotW", "TotK"]

    for d in delete_dirs:
        target_dir = DATASET_DIR / d
        if target_dir.exists() and target_dir.is_dir():
            print(f"Deleting directory: {target_dir}")
            shutil.rmtree(target_dir)
        else:
            print(f"Directory not found (skip): {target_dir}")

def delete_mp4_in_data_folders():
    for i in range(1, 70):
        folder_name = f"Data{i:02d}"
        data_dir = DATASET_DIR / folder_name

        if not data_dir.exists():
            print(f"Folder not found (skip): {data_dir}")
            continue

        mp4_files = list(data_dir.rglob("*.mp4"))

        for mp4 in mp4_files:
            print(f"Deleting mp4: {mp4}")
            mp4.unlink()

def main():
    delete_game_dirs()
    delete_mp4_in_data_folders()

    print("Done.")

if __name__ == "__main__":
    main()
