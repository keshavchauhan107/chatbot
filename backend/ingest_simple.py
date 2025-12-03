# ingest_simple.py - Offline version using TF-IDF
import os
import glob
from tqdm import tqdm
from PyPDF2 import PdfReader
import lancedb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

DATA_DIR = "data"
DB_DIR = "lancedb_store"
COLLECTION_NAME = "docs"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBED_DIM = 300  # TF-IDF dimension


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


def create_tfidf_embeddings(texts):
    print(f"Creating TF-IDF embeddings for {len(texts)} texts...")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Convert to dense arrays
    embeddings = tfidf_matrix.toarray().tolist()
    
    # Get actual dimensions
    actual_dim = len(embeddings[0]) if embeddings else 100
    print(f"TF-IDF vectors have {actual_dim} dimensions")
    
    # Save vectorizer and documents for later use
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('documents.pkl', 'wb') as f:
        pickle.dump(texts, f)
    
    return embeddings, actual_dim


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

    print("Creating embeddings with TF-IDF...")
    embeddings, actual_dim = create_tfidf_embeddings(docs)

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
            pa.field("embedding", pa.list_(pa.float32(), actual_dim))
        ])
        table = db.create_table(COLLECTION_NAME, schema=schema)

    print("Adding data to LanceDB...")
    batch_rows = []
    for i in range(len(docs)):
        batch_rows.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "embedding": embeddings[i],
        })

    table.add(batch_rows)

    print("Ingest completed successfully.")
    print("TF-IDF vectorizer and documents saved for search.")


if __name__ == "__main__":
    main()