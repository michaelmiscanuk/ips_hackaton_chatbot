"""Decompression utility script for CZSU multi-agent project using XZ (LZMA) format."""

import os
import shutil
import lzma
import tarfile
from pathlib import Path
from dotenv import load_dotenv

# Get the base directory
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

load_dotenv(BASE_DIR / "backend" / ".env")


# Get embedding model name for ChromaDB path
def get_chromadb_compressed_path():
    """Get ChromaDB compressed file path with embedding model suffix."""
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
    if provider == "azure":
        model_name = os.getenv(
            "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small_mimi"
        )
    else:
        model_name = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    # Sanitize model name for use in paths
    model_name = model_name.replace("/", "-").replace("\\", "-").replace(":", "-")
    return BASE_DIR / "backend" / "data" / f"chroma_db_{model_name}.tar.xz"


# Configuration of paths to decompress
PATHS_TO_DECOMPRESS = [
    get_chromadb_compressed_path(),  # ChromaDB for chatbot (with embedding model suffix)
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


def decompress_path(path_to_decompress: Path):
    """Decompress a tar.xz file at the specified path."""
    if not path_to_decompress.exists():
        print(f"Warning: Compressed file does not exist: {path_to_decompress}")
        return
    if not str(path_to_decompress).endswith(".tar.xz"):
        print(f"Warning: Not a tar.xz file: {path_to_decompress}")
        return

    # Remove .tar.xz to get target path
    target_path = Path(str(path_to_decompress).replace(".tar.xz", ""))
    print(f"Decompressing: {path_to_decompress}")
    print(f"Output: {target_path}")

    # Remove existing target
    safe_remove_directory(target_path)

    try:
        # Extract tar.xz file
        with lzma.open(path_to_decompress, "rb") as xz_file:
            with tarfile.open(fileobj=xz_file, mode="r") as tar:
                tar.extractall(target_path.parent)
        print(f"Successfully decompressed: {path_to_decompress}")
    except Exception as exc:
        print(f"Error extracting {path_to_decompress}: {exc}")


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

    # Step 2: Decompress all files
    print("\nStep 2: Starting decompression process...")
    for path in PATHS_TO_DECOMPRESS:
        print(f"\n{'='*60}")
        decompress_path(path)

    print("\n" + "=" * 60)
    print("Decompression process completed!")


if __name__ == "__main__":
    main()
