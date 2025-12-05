# Chatbot Vector DB Ingestion & Evaluation Guide

**Purpose:**
This document describes a reproducible pipeline for preparing knowledge for a retrieval-augmented chatbot: how to ingest text into a vector database, what metadata to store, recommended embedding strategies, retrieval and reranking options, plus evaluation and monitoring approaches to iteratively improve answer accuracy.

---

## 1. Goals

* Make text searchable via semantic similarity so the chatbot can answer user queries from the stored data.
* Store rich metadata so retrieved passages can be explained, filtered, and audited.
* Provide instrumentation and evaluation plans to measure and improve accuracy, relevance and safety over time.

## 2. Data sources

* Documentation (markdown, HTML pages, PDFs)
* Transcripts (meetings, calls)
* FAQs and support tickets
* Knowledge base articles and changelogs
* Product/user manuals

For each source capture: source_id, title, author, date, url, content, language, and source_type.

## 3. Preprocessing & chunking

1. Normalize: remove repeated whitespace, fix broken encoding, unify newlines.
2. Extract text from non-text formats (PDF, PPTX). Keep page numbers and other provenance.
3. Split into chunks sized for your embedding model **(recommend 200–800 tokens / 500–1500 characters)**. Use semantic chunking: prefer sentence or paragraph boundaries.
4. Add overlap between chunks (10–30%) to preserve context across chunk boundaries.
5. Deduplicate identical or near-identical chunks.

Store each chunk as an ingestion unit with chunk_id and sequence index within the original document.

## 4. Metadata schema (recommended)

```json
{
  "id": "uuid",
  "source_id": "document-123",
  "chunk_id": "document-123::chunk-0001",
  "text": "...",
  "title": "Installation Guide",
  "section": "2. Setup",
  "url": "https://...",
  "language": "en",
  "author": "Alice",
  "created_at": "2024-10-01T12:34:56Z",
  "embedding_model": "text-embedding-3-small",
  "tokens": 321,
  "checksum": "sha256...",
  "source_type": "pdf",
  "version": "v1.2"
}
```

## 5. Embeddings: model selection & considerations

* Choose a model that balances cost and capability. Evaluate at least two: one high-quality (larger) and one efficient (smaller).
* Consistently record the embedding model and model version in metadata.
* If you upgrade embedding models later, re-embed the corpus or store model-specific embeddings per chunk (recommended if you need to compare performance).
* Normalize vectors if your DB expects unit-length vectors.

## 6. Vector DB design & indexing

* Store vector, metadata, and a human-readable text snippet.
* Index strategy: use ANN (HNSW, IVFPQ) for large corpora; brute force for small (<100k) corpora.
* Keep a deterministic id mapping to the original document and chunk for traceability.
* Consider storing multiple index shards by topic or source to accelerate targeted retrieval.

## 7. Retrieval strategy

* Query embedding -> top-K nearest neighbors (K=5–50). Tune K based on average chunk size and answer composition.
* Use **hybrid retrieval**: combine sparse BM25 scoring with vector similarity for better lexical matches.
* Apply metadata filters (language, product version, date ranges) before or after retrieval when appropriate.

## 8. Reranking & answer synthesis

* Rerank retrieved chunks with a cross-encoder or use the downstream LLM to rank by relevance.
* Build a final prompt that includes: query, top N reranked passages (with short citations), and explicit instructions for the model to cite sources and avoid hallucination.
* Limit context size: either select the best N passages by reranker score or use budgeted token allocation per passage.

## 9. Prompt templates (example)

```
You are a helpful assistant. Use the following passages (labeled [1..N]) to answer the user's question. When you use information from a passage, cite its id in square brackets like [doc-123::chunk-4]. If you don't know, say you don't know.

Question: {user_query}

Passages:
[1] {passage_1_text}
[2] {passage_2_text}
...

Answer concisely and include citations.
```

## 10. Evaluation & metrics

### Automated metrics

* **Precision@k / Recall@k** for retrieval (how often the ground-truth chunk appears in top-K).
* **MRR (Mean Reciprocal Rank)** to measure ranking quality.
* **NDCG** when graded relevance levels exist.
* **Exact Match / F1** when ground-truth answers are extractive and can be computed automatically.

### Human evaluation

* **Accuracy**: whether the answer correctly addresses the question.
* **Helpfulness / Usefulness**: 1–5 rating by raters.
* **Hallucination rate**: percent of answers containing unsupported claims.
* **Citation correctness**: are citations relevant and correctly used.

Collect a representative evaluation set with test questions, expected answers, and the gold source chunk(s). Keep the set split by topic difficulty, ambiguity, freshness, and question type (factual, conversational, multi-hop).

## 11. Testing scenarios & test suite

Create tests for:

* Direct fact lookup (single-chunk)
* Multi-chunk synthesis (answers require combining two or more chunks)
* Out-of-domain questions (should return "I don't know" or fallback)
* Versioned content (e.g., behavior changed across versions)
* Ambiguous query handling (ask clarifying question or provide best-effort answer and caveat)

Record expected outcomes and acceptance criteria for each test.

## 12. Instrumentation & logging

Log for each user query:

* query text, timestamp
* user id (pseudonymized)
* query embedding model and vector
* retrieved ids and similarity scores
* reranker scores (if any)
* final answer + tokens used
* whether answer was judged correct (if feedback exists)

Use logs to compute metrics, discover failure modes, and prioritize re-annotation or re-ingestion.

## 13. Continuous improvement loop

1. Monitor low-scoring/low-feedback queries.
2. Triage: is the fault retrieval (no relevant chunks) or generation (incorrect synthesis)?
3. If retrieval: add or re-chunk content, re-embed with updated model, or tune filters/K.
4. If generation: improve prompt, add reranker, or provide additional grounding text.
5. Use active learning: surface ambiguous or low-confidence queries to human annotators for labeling.

## 14. Human-in-the-loop & feedback collection

* Provide a mechanism for users/agents to flag incorrect answers and select correct source passages.
* Store labeled pairs (query -> correct chunk ids) and use them to train/validate rerankers.

## 15. Privacy & compliance

* Mask or remove PII during ingestion unless explicitly required. Record reasons for keeping sensitive data.
* Enforce access control on the vector DB and logs. Encrypt vectors and metadata at rest as required by policy.

## 16. Operational tips

* Periodically (weekly/monthly) re-evaluate embedding models on a validation dataset.
* Keep a changelog of ingestion runs and model versions.
* Snapshot indices before major reindexing for rollback.

## 17. Example ingestion pseudo-commands

```
# extract text
python ingest/extract_text.py --input ./docs --output ./extracted

# chunk and embed
python ingest/chunk_and_embed.py --input ./extracted --chunk_size 800 --overlap 100 --embed_model text-embedding-3-small --out ./embeddings.jsonl

# upsert into vector DB
python ingest/upsert.py --vectors ./embeddings.jsonl --vector_db pinecone --index "kb_v1"
```

---

## Appendix: Quick checklist before embedding

* [ ] Source catalog completed with metadata
* [ ] Text extraction verified for PDFs/Images
* [ ] Chunk size & overlap decided and documented
* [ ] Deduplication run
* [ ] Embedding model selected and recorded
* [ ] Evaluation dataset prepared
* [ ] Logging/instrumentation enabled

---
