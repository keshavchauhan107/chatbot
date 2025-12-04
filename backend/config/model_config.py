# config.py
# Central configuration file - change settings here only!

# Embedding Model Configuration
EMBEDDING_MODEL = 'all-mpnet-base-v2'  # Change this to switch models

# Model dimensions (update when changing model)
# paraphrase-MiniLM-L3-v2: 384
# all-MiniLM-L6-v2: 384
# all-mpnet-base-v2: 768
EMBED_DIM = 384

# Database Configuration
DB_DIR = "lancedb_store"
COLLECTION_NAME = "docs"

# Document Processing
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
BATCH_SIZE = 32

# Chat Configuration
DEFAULT_N_RESULTS = 5

# Model Dimension Mapping (automatically set based on model)
MODEL_DIMENSIONS = {
    'paraphrase-MiniLM-L3-v2': 384,
    'all-MiniLM-L6-v2': 384,
    'all-mpnet-base-v2': 768,
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 384,
}

# Auto-update EMBED_DIM based on model
if EMBEDDING_MODEL in MODEL_DIMENSIONS:
    EMBED_DIM = MODEL_DIMENSIONS[EMBEDDING_MODEL]
else:
    print(f"Warning: {EMBEDDING_MODEL} not in MODEL_DIMENSIONS. Using EMBED_DIM={EMBED_DIM}")
