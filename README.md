# RAG Bot - Retrieval Augmented Generation Chatbot

A fast, open-source chatbot powered by Retrieval Augmented Generation (RAG) that answers questions based on your document collection. Uses free, open-source models with no API keys required.

## 🚀 Features

- **Document Ingestion**: Support for `.txt`, `.md`, and `.pdf` files
- **Vector Embeddings**: Uses `sentence-transformers` (free, no API key needed)
- **Semantic Search**: LanceDB vector database for fast similarity search
- **Simple Response Generation**: Context-aware answers based on retrieved documents
- **Web Interface**: Clean, responsive HTML chat UI
- **Offline Capable**: Works completely offline after setup

## 📋 Requirements

- Python 3.8+
- ~2GB free disk space (for models)
- Optional: GPU support (for faster embeddings)

## 🔧 Installation & Setup

### Step 1: Clone or Navigate to Project

```bash
cd backend
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `lancedb` - Vector database
- `sentence-transformers` - Embedding model
- `PyPDF2` - PDF processing
- And other utilities

### Step 4: Prepare Your Documents

1. Create a `data` folder (if not already present):
```bash
mkdir -p data
```

2. Add your documents to the `data` folder:
   - Text files (`.txt`)
   - Markdown files (`.md`)
   - PDF files (`.pdf`)

**Example:**
```
backend/
  data/
    document1.txt
    document2.pdf
    document3.md
```

### Step 5: Ingest Documents

Run the ingestion script to process your documents and create embeddings:

```bash
python ingest.py
```

**What happens:**
- Documents are read and split into chunks (900 characters with 150 character overlap)
- Each chunk is embedded using `sentence-transformers`
- Embeddings and metadata are stored in LanceDB vector database
- Process may take a few minutes depending on document size

**Output:**
```
Loading embedding model...
Loading documents...
Embedding X chunks...
Upserting into LanceDB...
Ingest completed successfully.
```

### Step 6: Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Output:**
```
Uvicorn running on http://0.0.0.0:8000
Press CTRL+C to quit
```

### Step 7: Access the Chat Interface

Open your browser and navigate to:
```
http://localhost:8000
```

You should see the RAG Bot chat interface. Start asking questions!

## 📝 Usage Examples

Once the server is running, try these queries:

- "What is this document about?"
- "Summarize the key features"
- "How does the RAG bot work?"
- "Find information about [topic]"

## 🏗️ Project Structure

```
backend/
├── app.py                      # Main FastAPI application
├── ingest.py                   # Document ingestion script
├── requirements.txt            # Python dependencies
├── data/                       # Your documents go here
│   └── sample_doc.txt
├── lancedb_store/              # Vector database (auto-created)
│   └── docs.lance/
└── static/
    └── index.html              # Web UI
```

## 🔄 Workflow

1. **Ingestion Phase** (`ingest.py`):
   - Read documents from `data/` folder
   - Split into overlapping chunks
   - Generate embeddings using `sentence-transformers`
   - Store in LanceDB

2. **Query Phase** (`app.py`):
   - User enters a question in the web UI
   - Query is embedded using same model
   - Semantic search finds top-5 similar document chunks
   - Response is generated based on context
   - Sources are cited

## 🛠️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xyz'"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Run ingest.py first – no LanceDB collection found!"
**Solution:**
```bash
python ingest.py
```

### Issue: Very slow embedding generation
**Solution:** 
- This is normal for CPU-only. First run downloads the model (~200MB)
- If you have a GPU, uncomment the GPU line in `ingest.py`
- Use smaller document files initially

### Issue: Port 8000 already in use
**Solution:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8001
# Then access http://localhost:8001
```

## 🚀 Advanced Configuration

### Change Embedding Model

Edit `ingest.py` and `app.py`:
```python
# Smaller & faster (384 dims)
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Better quality but slower (384 dims)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Best quality, larger model (768 dims)
embedding_model = SentenceTransformer('all-mpnet-base-v2')
```

### Adjust Chunk Size

Edit `ingest.py`:
```python
CHUNK_SIZE = 900          # Smaller = more granular, larger = more context
CHUNK_OVERLAP = 150       # Overlap between chunks for context continuity
```

### Change Number of Retrieved Documents

In the web UI, the `n_results` parameter controls how many chunks are retrieved:
```javascript
body: JSON.stringify({query, n_results: 5})  // Change 5 to desired number
```

## 📊 Performance Tips

1. **First Run**: Model download takes time (~200MB). Subsequent runs are faster.
2. **Large Documents**: Split into multiple files for better organization
3. **GPU Support**: Install `torch` with GPU support for 5-10x faster embeddings
4. **Batch Processing**: Increase `BATCH_SIZE` in `ingest.py` if you have memory

## 🗑️ Cleaning Up

To start fresh:

```bash
# Delete vector database
rm -rf lancedb_store

# Re-ingest documents
python ingest.py
```

## 📚 Adding More Documents

1. Add new files to `backend/data/`
2. Run ingestion again:
   ```bash
   python ingest.py
   ```
3. New documents are added to the vector database

## 🛡️ API Endpoints

### `GET /`
Returns the HTML chat interface

### `POST /chat`
**Request:**
```json
{
  "query": "Your question here",
  "n_results": 5
}
```

**Response:**
```json
{
  "answer": "Answer based on documents...",
  "sources": ["document1.txt", "document2.pdf"]
}
```

## 📖 How It Works (Technical)

1. **Chunking**: Documents split into ~900 character overlapping chunks
2. **Embedding**: Each chunk converted to 384-dimensional vector
3. **Storage**: Vectors indexed in LanceDB for fast retrieval
4. **Query**: User question embedded to same dimension
5. **Search**: Find top-5 most similar chunks using cosine similarity
6. **Response**: Generate answer by combining query + context chunks

## ⚙️ Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `lancedb` | Vector database |
| `sentence-transformers` | Embedding model |
| `torch` | Deep learning backend |
| `PyPDF2` | PDF parsing |
| `pyarrow` | Data serialization |

## 📝 License

Open source - feel free to modify and use

## 🤝 Contributing

Contributions welcome! Feel free to submit issues and pull requests.

## ❓ FAQ

**Q: Do I need an API key?**
A: No! Everything runs locally with free, open-source models.

**Q: How many documents can I add?**
A: Thousands! Limited only by your disk space.

**Q: Can I use this offline?**
A: Yes, after the initial setup. All processing is local.

**Q: What about privacy?**
A: Complete privacy! Your documents never leave your machine.

---

For issues or questions, please check the troubleshooting section above.