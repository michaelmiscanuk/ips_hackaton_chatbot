import os
import sys
import asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.models import get_langchain_azure_embedding_model

# Load environment variables
load_dotenv()

# Constants
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
    print(f"🔍 BASE_DIR calculated from __file__: {BASE_DIR}")
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
    print(f"🔍 BASE_DIR calculated from cwd: {BASE_DIR}")

CSV_PATH = BASE_DIR / "backend" / "data" / "sample0.csv"
CHROMA_DB_DIR = BASE_DIR / "backend" / "data" / "chroma_db"


async def process_row(row):
    """Process a single row: format and create document."""
    # Combine fields into a single text chunk
    text_content = (
        f"Subject: {row['subject']}\nBody: {row['body']}\nAnswer: {row['answer']}"
    )

    print(f"Processing row: {row['subject'][:30]}...")

    # Create metadata
    metadata = {
        "subject": row["subject"],
        "type": row["type"],
        "language": row["language"],
        "original_text": text_content,
    }

    return Document(page_content=text_content, metadata=metadata)


async def main():
    print(f"Reading data from {CSV_PATH}...")
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} records.")

    documents = []
    for index, row in df.iterrows():
        doc = await process_row(row)
        documents.append(doc)

    print(f"Prepared {len(documents)} documents.")

    print("Initializing Embedding Model...")
    embedding_model = get_langchain_azure_embedding_model()

    print(f"Creating/Overwriting ChromaDB at {CHROMA_DB_DIR}...")
    # Initialize Chroma and add documents
    # persist_directory will save data to disk
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DB_DIR),
    )

    print("Data ingestion complete!")


if __name__ == "__main__":
    asyncio.run(main())
