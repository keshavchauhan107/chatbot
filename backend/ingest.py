# ingest.py
import os
import glob
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import lancedb
import numpy as np
import sys
from pathlib import Path
from typing import List

# Try to import config from project root or fallback path
try:
    import config as cfg
except Exception:
    try:
        import backend.config.model_config as cfg
    except Exception as e:
        raise RuntimeError("Could not import config.py. Ensure config.py exists.") from e

EMBEDDING_MODEL = getattr(cfg, "EMBEDDING_MODEL", "all-mpnet-base-v2")
# EMBED_DIM from config is only advisory — we auto-detect real dim below
EMBED_DIM = getattr(cfg, "EMBED_DIM", None)
DB_DIR = getattr(cfg, "DB_DIR", "lancedb_store")
COLLECTION_NAME = getattr(cfg, "COLLECTION_NAME", "docs")
CHUNK_SIZE = getattr(cfg, "CHUNK_SIZE", 900)
CHUNK_OVERLAP = getattr(cfg, "CHUNK_OVERLAP", 150)
BATCH_SIZE = getattr(cfg, "BATCH_SIZE", 32)
DATA_DIR = getattr(cfg, "DATA_DIR", "data") if hasattr(cfg, "DATA_DIR") else "data"

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ingest")

# Load embedding model
log.info(f"Loading embedding model: {EMBEDDING_MODEL} ...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        try:
            page_text = p.extract_text()
            if page_text:
                text.append(page_text)
        except Exception:
            # ignore extraction errors on pages
            pass
    return "\n".join(text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple sliding-window chunker. Returns list of chunks > 20 chars."""
    if not text:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - chunk_overlap)
    while i < len(text):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 20:
            chunks.append(chunk)
        i += step
    return chunks


def embed_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
    """Return list of embeddings as float32 lists (one list per text)."""
    log.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})...")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        # convert to numpy for efficient dtype conversion
        np_emb = embedding_model.encode(batch, convert_to_numpy=True)
        # ensure dtype float32
        np_emb = np_emb.astype(np.float32)
        for row in np_emb:
            embeddings.append(row.tolist())
    return embeddings


def create_table_with_schema(db, collection_name: str, vector_dim: int):
    import pyarrow as pa
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("metadata", pa.struct([
            pa.field("source", pa.string()),
            pa.field("path", pa.string()),
            pa.field("chunk", pa.int64())
        ])),
        pa.field("vector", pa.list_(pa.float32(), vector_dim))
    ])
    table = db.create_table(collection_name, schema=schema)
    log.info(f"Created LanceDB table '{collection_name}' with vector dim {vector_dim}")
    return table


def main(data_dir: str = DATA_DIR, db_dir: str = DB_DIR):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise RuntimeError(f"Data directory not found: {data_dir}")

    os.makedirs(db_dir, exist_ok=True)

    log.info("Scanning documents...")
    files = glob.glob(f"{data_dir}/**/*.*", recursive=True)
    files = [f for f in files if Path(f).is_file()]

    ids, docs, metas = [], [], []
    for path in files:
        ext = Path(path).suffix.lower()
        try:
            if ext in [".txt", ".md"]:
                text = read_text_file(path)
            elif ext == ".pdf":
                text = read_pdf(path)
            else:
                log.debug(f"Skipping unsupported file: {path}")
                continue
        except Exception as e:
            log.warning(f"Failed reading {path}: {e}")
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{Path(path).name}__{i}")
            docs.append(chunk)
            metas.append({
                "source": Path(path).name,
                "path": str(path),
                "chunk": int(i)
            })

    if not docs:
        log.info("No document text chunks found under data/. Nothing to ingest.")
        return

    # Generate embeddings (batched)
    embeddings = embed_texts(docs, batch_size=BATCH_SIZE)

    # detect actual model dimension (reliable)
    model_dim = embedding_model.get_sentence_embedding_dimension()
    log.info(f"Detected model embedding dimension: {model_dim}")
    if EMBED_DIM and EMBED_DIM != model_dim:
        log.warning(f"Config EMBED_DIM={EMBED_DIM} differs from model_dim={model_dim}. Using detected model_dim for schema.")

    # Connect to LanceDB
    db = lancedb.connect(db_dir)

    # If table exists, drop it to avoid schema mismatches (safe for reingest)
    if COLLECTION_NAME in db.table_names():
        log.info(f"Dropping existing table '{COLLECTION_NAME}' to recreate with correct schema...")
        try:
            db.drop_table(COLLECTION_NAME)
        except Exception as e:
            log.warning(f"db.drop_table failed: {e}. You may need to remove the DB directory manually.")
            raise

    # Create table with correct vector dim
    table = create_table_with_schema(db, COLLECTION_NAME, model_dim)

    # Insert rows in batches
    log.info("Inserting rows into LanceDB...")
    batch_size = max(1, BATCH_SIZE)
    for i in tqdm(range(0, len(docs), batch_size), desc="DB batches"):
        rows = []
        for j in range(i, min(i + batch_size, len(docs))):
            rows.append({
                "id": ids[j],
                "text": docs[j],
                "metadata": metas[j],
                "vector": embeddings[j]
            })
        try:
            table.add(rows)
        except Exception as e:
            log.exception("Failed adding batch to LanceDB")
            raise

    log.info("Ingest completed successfully. Total chunks: %d", len(docs))


if __name__ == "__main__":
    main()
