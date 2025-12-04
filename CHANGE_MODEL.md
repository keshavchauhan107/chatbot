# How to Change Embedding Model

Now you can change the embedding model in **ONE place** and everything updates automatically!

## Step 1: Edit `config.py`

Open `/workspaces/chatbot/config.py` and change this line:

```python
EMBEDDING_MODEL = 'paraphrase-MiniLM-L3-v2'  # Change this
```

## Available Models

```python
# Fast, good quality (384 dims)
EMBEDDING_MODEL = 'paraphrase-MiniLM-L3-v2'

# Better quality, same speed (384 dims)
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Best quality but slower (768 dims)
EMBEDDING_MODEL = 'all-mpnet-base-v2'

# Multilingual support (384 dims)
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
```

## Step 2: Delete Old Database

```bash
cd /workspaces/chatbot/backend
rm -rf lancedb_store
```

This is necessary because different models have different embedding dimensions.

## Step 3: Re-ingest Documents

```bash
python ingest.py
```

The script will automatically:
- Load the new model from `config.py`
- Use the correct embedding dimension
- Create new embeddings
- Store them in LanceDB

## Step 4: Restart Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## What Changed Automatically?

When you update `EMBEDDING_MODEL` in `config.py`:

✅ `ingest.py` - Uses new model for embeddings
✅ `app.py` - Uses same model for queries
✅ `EMBED_DIM` - Automatically adjusted based on model
✅ Console output - Shows which model is loading

---

## Example: Switch to Better Model

### Current (Fast)
```python
EMBEDDING_MODEL = 'paraphrase-MiniLM-L3-v2'  # 384 dims
```

### Change to (Better Quality)
```python
EMBEDDING_MODEL = 'all-mpnet-base-v2'  # 768 dims
```

Then:
```bash
cd backend
rm -rf lancedb_store
python ingest.py
```

Done! All files are updated automatically.

---

## Why This Matters

Before: You had to change 2 files + update 2 dimension values = 4 places!
Now: Just change 1 line in `config.py` = Everything works!

**No more inconsistencies!** 🎉
