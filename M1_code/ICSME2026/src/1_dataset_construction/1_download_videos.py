from pathlib import Path
import csv
from pytubefix import YouTube

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"

URL_CSV_PATH = DATA_DIR / "1_dataset_construction" / "urls.csv"

def download_video(url: str, save_path: Path):
    yt = YouTube(url)

    stream = (
        yt.streams
        .filter(progressive=True, file_extension="mp4")
        .get_highest_resolution()
    )

    if stream is None:
        raise RuntimeError(f"No downloadable stream found for {url}")

    stream.download(
        output_path=save_path.parent,
        filename=save_path.name
    )


def main():
    if not URL_CSV_PATH.exists():
        raise FileNotFoundError(f"{URL_CSV_PATH} not found")

    with open(URL_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game = row["Game"].strip()
            category = row["Category"].strip()
            urls = [u.strip() for u in row["URL"].split(",") if u.strip()]

            save_dir = DATASET_DIR / game / category
            save_dir.mkdir(parents=True, exist_ok=True)

            for idx, url in enumerate(urls, start=1):
                video_path = save_dir / f"video{idx}.mp4"
                print(f"[INFO] Downloading: {url}")
                print(f"       -> {video_path}")

                try:
                    download_video(url, video_path)
                except Exception as e:
                    print(f"[ERROR] Failed to download {url}")
                    print(f"        {e}")

if __name__ == "__main__":
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    main()
