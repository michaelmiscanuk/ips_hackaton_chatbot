#!/bin/bash
set -e

# Unzip ChromaDB if needed
if [ -f "data/chroma_db.zip" ] && [ ! -d "data/chroma_db" ]; then
    echo "Extracting ChromaDB..."
    python3 -c "import zipfile; zipfile.ZipFile('data/chroma_db.zip', 'r').extractall('data')"
    echo "ChromaDB extracted successfully."
fi

# Use the PORT environment variable provided by Railway, or default to 8000
PORT="${PORT:-8000}"

echo "Starting uvicorn on port $PORT"

# Start uvicorn
exec uvicorn api:app --host 0.0.0.0 --port "$PORT" --log-level info
