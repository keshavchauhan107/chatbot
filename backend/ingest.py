# ingest.py
import os
import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import lancedb
import numpy as np
import sys

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL, EMBED_DIM, DB_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE
)

DATA_DIR = "data"

# Initialize free embedding model
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path):
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        try:
            text.append(p.extract_text() or "")
        except:
            pass
    return "\n".join(text)


def chunk_text(text):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + CHUNK_SIZE]
        chunks.append(chunk.strip())
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 20]


def embed_texts(texts):
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = embedding_model.encode(batch)
        embeddings.extend(batch_embeddings.tolist())
    return embeddings


def main():
    os.makedirs(DB_DIR, exist_ok=True)

    print("Loading documents...")
    files = glob.glob(f"{DATA_DIR}/**/*.*", recursive=True)

    ids, docs, metas = [], [], []
    for path in files:
        ext = os.path.splitext(path)[1].lower()

        if ext in [".txt", ".md"]:
            text = read_text_file(path)
        elif ext == ".pdf":
            text = read_pdf(path)
        else:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{os.path.basename(path)}__{i}")
            docs.append(chunk)
            metas.append({
                "source": os.path.basename(path),
                "path": path,
                "chunk": i
            })

    if not docs:
        print("No documents found in /data.")
        return

    print("Embedding", len(docs), "chunks...")
    embeddings = embed_texts(docs)

    db = lancedb.connect(DB_DIR)

    if COLLECTION_NAME in db.table_names():
        table = db.open_table(COLLECTION_NAME)
        # Clear existing data
        table.delete("id IS NOT NULL")
    else:
        # Create schema for new table
        import pyarrow as pa
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.struct([
                pa.field("source", pa.string()),
                pa.field("path", pa.string()),
                pa.field("chunk", pa.int64())
            ])),
            pa.field("vector", pa.list_(pa.float32(), EMBED_DIM))
        ])
        table = db.create_table(COLLECTION_NAME, schema=schema)

    print("Upserting into LanceDB...")
    batch_rows = []
    for i in range(len(docs)):
        batch_rows.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "vector": embeddings[i],
        })

    table.add(batch_rows)

    print("Ingest completed successfully.")


if __name__ == "__main__":
    main()