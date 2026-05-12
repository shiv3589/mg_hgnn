import os
import requests
import zipfile
import pathlib

OULAD_URL = "https://analyse.kmi.open.ac.uk/open-dataset/download"
SAVE_DIR  = pathlib.Path("data/raw/oulad")


def download_oulad():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SAVE_DIR / "oulad.zip"

    if not zip_path.exists():
        print("Downloading OULAD (~40 MB)...")
        r = requests.get(OULAD_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("Zip already present, skipping download.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(SAVE_DIR)

    csvs = list(SAVE_DIR.glob("*.csv"))
    print(f"Extracted {len(csvs)} CSV files:")
    for c in sorted(csvs):
        print(f"  {c.name}  ({c.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    download_oulad()
