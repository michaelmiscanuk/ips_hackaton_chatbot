"""Unzip utility script for CZSU multi-agent project."""

import os
import shutil
import zipfile
import gdown
from pathlib import Path

# Get the base directory
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

# Configuration of paths to unzip
PATHS_TO_UNZIP = [
    BASE_DIR / "backend" / "data" / "chroma_db.zip",  # ChromaDB for chatbot
]


def download_from_gdrive(gdrive_url: str, destination_path: Path) -> bool:
    """Simple Google Drive download using gdown."""
    print(f"Downloading to: {destination_path}")

    # Extract file ID
    file_id = gdrive_url.split("/file/d/")[-1].split("/")[0]
    print(f"File ID: {file_id}")

    # Make sure directory exists
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download the file
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            str(destination_path),
            quiet=False,
        )

        if destination_path.exists():
            print("✅ Downloaded successfully")
            return True
        print("❌ Download failed")
        return False

    except Exception as exc:
        print(f"❌ Download failed: {exc}")
        return False


def safe_remove_directory(target_path: Path):
    """Safely remove directory, handling Windows permissions."""
    if not target_path.exists():
        return

    print(f"Removing existing: {target_path}")
    try:
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
    except PermissionError:
        # Force remove on Windows
        if os.name == "nt":
            os.system(f'rmdir /s /q "{target_path}"')
        else:
            raise


def format_size(size_bytes):
    """Format size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"


def unzip_path(path_to_unzip: Path):
    """Unzip a file at the specified path."""
    if not path_to_unzip.exists():
        print(f"Warning: Zip file does not exist: {path_to_unzip}")
        return
    if not path_to_unzip.suffix == ".zip":
        print(f"Warning: Not a zip file: {path_to_unzip}")
        return

    target_path = path_to_unzip.with_suffix("")
    print(f"Unzipping: {path_to_unzip}")
    print(f"Output: {target_path}")

    # Remove existing target
    safe_remove_directory(target_path)

    try:
        # Extract zip file
        with zipfile.ZipFile(path_to_unzip, "r") as zipf:
            zipf.extractall(target_path.parent)
        print(f"Successfully unzipped: {path_to_unzip}")
    except Exception as exc:
        print(f"Error extracting {path_to_unzip}: {exc}")


def main():
    """Main function to download and unzip files."""
    print(f"Base directory: {BASE_DIR}")

    # Step 1: Download from Google Drive
    # print("Step 1: Downloading files from Google Drive...")
    # gdrive_url = "https://drive.google.com/file/d/1zjS6tsTmUaYy63E4Cq8NqvXgJpS2TQl1/view?usp=sharing"
    # destination_path = BASE_DIR / "data" / "pdf_chromadb_llamaparse.zip"

    # success = download_from_gdrive(gdrive_url, destination_path)
    # if not success:
    #     print("❌ Download failed. Cannot proceed.")
    #     return

    # Step 2: Unzip all files
    print("\nStep 2: Starting unzip process...")
    for path in PATHS_TO_UNZIP:
        print(f"\n{'='*60}")
        unzip_path(path)

    print("\n" + "=" * 60)
    print("Unzip process completed!")


if __name__ == "__main__":
    main()
