# ingest.py
import os
import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import lancedb

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free embedding model

DATA_DIR = "data"
DB_DIR = "lancedb_store"
COLLECTION_NAME = "docs"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


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
    """Generate embeddings using Hugging Face model"""
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    return [emb.tolist() for emb in embeddings]


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
            pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM))
        ])
        table = db.create_table(COLLECTION_NAME, schema=schema)

    print("Upserting into LanceDB...")
    batch_rows = []
    for i in range(len(docs)):
        batch_rows.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "embedding": embeddings[i],   # plain list embedding
        })

    table.add(batch_rows)

    print("Ingest completed successfully.")


if __name__ == "__main__":
    main()
