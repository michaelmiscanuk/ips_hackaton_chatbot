"""Zip utility script for CZSU multi-agent project."""

import os
import zipfile
from pathlib import Path

# Get the base directory
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

# Configuration of paths to zip
PATHS_TO_ZIP = [
    BASE_DIR / "backend" / "data" / "chroma_db",  # ChromaDB for chatbot
]


def zip_path(path_to_zip: Path):
    """Zip a file or folder at the specified path with better compression."""
    abs_path = path_to_zip
    if not abs_path.exists():
        print(f"Warning: Path does not exist: {abs_path}")
        return
    # Create zip file path (same location as original)
    zip_path = abs_path.with_suffix(".zip")
    print(f"Zipping: {abs_path}")
    print(f"Output: {zip_path}")
    print("Using LZMA compression for better compression ratio...")

    # Create zip file with better compression
    # ZIP_LZMA provides the best compression ratio (but is slower)
    # compresslevel=9 gives maximum compression for DEFLATE
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_LZMA, compresslevel=9) as zipf:
            if abs_path.is_file():
                # If it's a file, just add it
                zipf.write(abs_path, abs_path.name)
            else:
                # If it's a directory, add all files recursively
                for root, _, files in os.walk(abs_path):
                    for file in files:
                        file_path = Path(root) / file
                        # Calculate relative path for the file in the zip
                        rel_path = file_path.relative_to(abs_path.parent)
                        zipf.write(file_path, rel_path)
    except (OSError, RuntimeError) as exc:
        # Fallback to DEFLATE with maximum compression if LZMA fails
        print(
            f"LZMA compression failed ({exc}), "
            f"falling back to DEFLATE with max compression..."
        )
        with zipfile.ZipFile(
            zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zipf:
            if abs_path.is_file():
                zipf.write(abs_path, abs_path.name)
            else:
                for root, _, files in os.walk(abs_path):
                    for file in files:
                        file_path = Path(root) / file
                        rel_path = file_path.relative_to(abs_path.parent)
                        zipf.write(file_path, rel_path)

    print(f"Successfully zipped: {abs_path}")
    # Show file size info
    original_size = get_size(abs_path)
    compressed_size = zip_path.stat().st_size
    if original_size > 0:
        compression_ratio = ((original_size - compressed_size) / original_size) * 100
        print(f"Original size: {format_size(original_size)}")
        print(f"Compressed size: {format_size(compressed_size)}")
        print(f"Compression ratio: {compression_ratio:.1f}%")


def get_size(path: Path):
    """Get total size of a file or directory."""
    if path.is_file():
        return path.stat().st_size
    else:
        total_size = 0
        for root, _, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    pass
        return total_size


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


def upload_to_gdrive(file_path: Path, gdrive_folder_link: str) -> bool:
    """
    Prepare file for Google Drive upload and provide easy access.
    Since Google Drive API requires authentication, this function will:
    1. Copy the file to an easily accessible location
    2. Open the Google Drive folder in browser
    3. Provide clear instructions for manual upload

    Args:
        file_path: Path to the file to upload
        gdrive_folder_link: Google Drive shared folder link

    Returns:
        bool: True if file preparation successful, False otherwise
    """
    if not file_path.exists():
        print(f"Error: File does not exist: {file_path}")
        return False

    try:
        # Create a convenient upload folder on desktop
        desktop_path = Path.home() / "Desktop"
        upload_folder = desktop_path / "GDRIVE_UPLOAD"
        upload_folder.mkdir(exist_ok=True)

        # Copy file to desktop upload folder
        destination_file = upload_folder / file_path.name

        print(f"Preparing file for Google Drive upload...")
        print(f"File: {file_path.name}")
        print(f"Size: {format_size(file_path.stat().st_size)}")

        # Show progress bar while copying
        print("Copying file to desktop folder...")
        print_progress_bar(0, 100, prefix="Progress:", suffix="Complete", length=50)

        import shutil

        shutil.copy2(file_path, destination_file)

        print_progress_bar(100, 100, prefix="Progress:", suffix="Complete", length=50)

        print(f"\n✓ File copied to: {destination_file}")
        print(f"\n{'='*60}")
        print("MANUAL UPLOAD INSTRUCTIONS")
        print(f"{'='*60}")
        print(f"1. File location: {destination_file}")
        print(f"2. Google Drive folder: {gdrive_folder_link}")
        print(f"3. Open the Google Drive folder in your browser")
        print(f"4. Drag and drop the file from your desktop folder")
        print(f"{'='*60}")

        # Try to open the Google Drive folder in browser
        try:
            import webbrowser

            print(f"Opening Google Drive folder in browser...")
            webbrowser.open(gdrive_folder_link)
        except Exception as exc:
            print(f"Could not open browser automatically: {exc}")
            print(f"Please manually open: {gdrive_folder_link}")

        # Try to open the desktop folder
        try:
            import subprocess

            print(f"Opening desktop upload folder...")
            subprocess.run(["explorer", str(upload_folder)], check=False)
        except Exception as exc:
            print(f"Could not open folder automatically: {exc}")
            print(f"Please manually navigate to: {upload_folder}")

        return True

    except Exception as exc:
        print(f"\n✗ Error preparing file: {exc}")
        print(f"Manual upload required:")
        print(f"File: {file_path}")
        print(f"Google Drive folder: {gdrive_folder_link}")
        return False


def print_progress_bar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    print_end="\r",
):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def upload_pdf_chromadb_to_gdrive():
    """Prepare the pdf_chromadb_llamaparse.zip file for Google Drive upload."""
    pdf_chromadb_zip = BASE_DIR / "data" / "pdf_chromadb_llamaparse.zip"
    # Google Drive shared folder link
    gdrive_folder_link = (
        "https://drive.google.com/drive/folders/"
        "1TZWxURgYoYHgKMji4OV333ftEDCyJRgD?usp=sharing"
    )

    print(f"\n{'='*50}")
    print("PREPARING FOR GOOGLE DRIVE UPLOAD")
    print(f"{'='*50}")

    if pdf_chromadb_zip.exists():
        success = upload_to_gdrive(pdf_chromadb_zip, gdrive_folder_link)
        if success:
            print("✓ File preparation completed successfully!")
            print("Please follow the instructions above to complete the upload.")
        else:
            print("✗ File preparation failed.")
    else:
        print(f"Warning: File not found: {pdf_chromadb_zip}")
        print("Make sure the pdf_chromadb_llamaparse folder was zipped successfully.")


def main():
    """Main function to zip files and upload to Google Drive."""
    print(f"Base directory: {BASE_DIR}")
    print("Starting zip process with improved compression...")

    for path in PATHS_TO_ZIP:
        zip_path(path)

    print("\nZip process completed!")

    # Prepare the pdf_chromadb_llamaparse.zip for Google Drive upload
    upload_pdf_chromadb_to_gdrive()


if __name__ == "__main__":
    main()
