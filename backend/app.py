# app.py (clean, consolidated)
import os
import re
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import lancedb

# Try importing config
try:
    import config as cfg
except Exception:
    try:
        import backend.config.model_config as cfg
    except Exception as e:
        raise RuntimeError("Could not import config.py. Ensure config.py exists or adjust import paths.") from e

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("local-chat")

# Config values
EMBEDDING_MODEL = getattr(cfg, "EMBEDDING_MODEL", "all-mpnet-base-v2")
DB_DIR = getattr(cfg, "DB_DIR", "lancedb_store")
COLLECTION_NAME = getattr(cfg, "COLLECTION_NAME", "docs")
DEFAULT_N_RESULTS = getattr(cfg, "DEFAULT_N_RESULTS", 5)
CHUNK_SIZE = getattr(cfg, "CHUNK_SIZE", 900)

# Prompts directory
PROMPT_DIR = Path("prompts")
if not PROMPT_DIR.exists():
    raise RuntimeError("prompts/ directory not found. Create prompts/*.md files in prompts/")

if not any(PROMPT_DIR.glob("*.md")) and not any(PROMPT_DIR.glob("*.txt")):
    raise RuntimeError("No prompt files found in prompts/. Create qa.md (or qa.txt) and system.md.")

DEFAULT_PROMPT_NAME = "qa.md" if (PROMPT_DIR / "qa.md").exists() else sorted([p.name for p in PROMPT_DIR.iterdir() if p.is_file()])[0]

# Load embedding model
log.info(f"Loading embedding model: {EMBEDDING_MODEL} ...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Connect to LanceDB and open table
db = lancedb.connect(DB_DIR)
if COLLECTION_NAME not in db.table_names():
    raise RuntimeError(f"Run ingest.py first — no LanceDB collection named '{COLLECTION_NAME}' found in {DB_DIR}.")
table = db.open_table(COLLECTION_NAME)

# FastAPI app
app = FastAPI()
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    if Path("static/index.html").exists():
        return FileResponse("static/index.html")
    return {"status": "ok", "info": "Chat API running"}


class QueryIn(BaseModel):
    query: str
    n_results: Optional[int] = DEFAULT_N_RESULTS
    prompt_name: Optional[str] = DEFAULT_PROMPT_NAME


# -----------------------
# Utilities
# -----------------------
def load_prompt(prompt_filename: str) -> str:
    p = PROMPT_DIR / prompt_filename
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


def sanitize_text(text: str) -> str:
    """Sanitize and remove markdown/headers/code/sources lines."""
    if not text:
        return ""
    text = str(text)

    # Remove YAML frontmatter
    text = re.sub(r"(?m)^---\s*$.*?^---\s*$", " ", text, flags=re.DOTALL)

    # Remove "Context 1:" or "Context 2:" artifacts
    text = re.sub(r"(?i)\bcontext\s*\d+\s*:\s*", " ", text)

    # Remove 'Sources:' lines
    text = re.sub(r"(?im)^sources\s*[:\-].*$", " ", text)

    # Remove markdown headings
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+.*$", " ", text)

    # Remove code fences and inline code
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Convert markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove bold/italic markers
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def build_context_text(contexts: List[dict], max_chars: int = 2000) -> str:
    if not contexts:
        return ""
    cleaned = []
    k = max(1, min(len(contexts), 4))
    per_ctx = max(200, max_chars // k)
    for ctx in contexts[:k]:
        raw = ctx.get("text", "") or ""
        text = sanitize_text(raw)
        if len(text) > per_ctx:
            cut = text[:per_ctx]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            text = cut + "..."
        cleaned.append(text)
    merged = "\n\n".join(cleaned)
    if len(merged) > max_chars:
        merged = merged[:max_chars].rsplit(" ", 1)[0] + "..."
    log.debug("Merged context (truncated): %s", merged[:1000])
    return merged


# -----------------------
# Local generation (no external LLM)
# -----------------------
def local_rewrite_fallback(context_text: str, question: str) -> str:
    if not context_text or not context_text.strip():
        return "I don't know based on the available documents."

    clauses = []
    for part in context_text.split("\n\n"):
        for sent in part.split(". "):
            s = sent.strip().rstrip(".")
            if s:
                if len(s) > 600:
                    s = s[:600].rsplit(" ", 1)[0] + "..."
                clauses.append(s)
            if len(clauses) >= 4:
                break
        if len(clauses) >= 4:
            break

    if not clauses:
        return "I don't know based on the available documents."

    first = clauses[0]
    second = clauses[1] if len(clauses) > 1 else ""
    if second:
        answer = f"{first}. {second}."
    else:
        answer = f"{first}."

    if answer and not answer[0].isupper():
        answer = answer[0].upper() + answer[1:]
    return answer


def generate_answer_local(prompt_name: str, merged_context: str, question: str) -> str:
    # load template for reference (not used to call external LLM here)
    try:
        _ = load_prompt(prompt_name)
    except FileNotFoundError:
        pass
    return local_rewrite_fallback(merged_context, question)


# -----------------------
# Main endpoint
# -----------------------
@app.post("/chat")
async def chat(q: QueryIn):
    if not q.query or not q.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    n_results = int(q.n_results or DEFAULT_N_RESULTS)
    prompt_name = q.prompt_name or DEFAULT_PROMPT_NAME

    # embed query
    try:
        emb = embedding_model.encode(q.query).tolist()
    except Exception:
        log.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail="Failed to compute embeddings")

    # retrieval
    try:
        results = (
            table.search(emb)
            .metric("cosine")
            .limit(n_results)
            .select(["id", "text", "metadata"])
            .to_list()
        )
    except Exception:
        log.exception("LanceDB search failed")
        raise HTTPException(status_code=500, detail="Vector search failed")

    contexts = []
    sources = []
    for r in results:
        ctx = {"id": r.get("id"), "text": r.get("text", ""), "metadata": r.get("metadata", {}) or {}}
        contexts.append(ctx)
        src = ctx["metadata"].get("source") or ctx["metadata"].get("filename")
        if src:
            sources.append(src)

    merged_context = build_context_text(contexts, max_chars=2000)

    if not merged_context:
        return {
            "answer": "I don't know based on the available documents.",
            "sources": [],
            "context_used": 0,
            "prompt_name": prompt_name,
            "prompt_version": "local-v1",
        }

    answer = generate_answer_local(prompt_name, merged_context, q.query)

    return {
        "answer": answer.strip(),
        "sources": list(dict.fromkeys(sources)),
        "context_used": len(contexts),
        "prompt_name": prompt_name,
        "prompt_version": "local-v1",
    }


@app.get("/prompts")
async def list_prompts():
    return {"prompts": sorted([p.name for p in PROMPT_DIR.iterdir() if p.is_file()])}


@app.get("/prompts/{name}")
async def read_prompt(name: str):
    p = PROMPT_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="prompt not found")
    return {"name": name, "content": p.read_text(encoding="utf-8")}
