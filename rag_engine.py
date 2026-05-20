"""
FPViber RAG engine.

Pipeline:
  1. Load multi-format documents from `data/` (TXT / PDF / DOCX).
  2. Clean noise (page numbers, recurring headers, hyphenation).
  3. Recursive character chunking that protects tables and math formulas.
  4. IBM Granite embeddings via the Hugging Face Inference API.
  5. FAISS cosine-similarity retrieval (inner product on L2-normalised vectors).
  6. Gemini answer generation with strict context-grounding rules.

State is kept inside a single `RAGEngine` instance that is initialised once
at app startup (the Flask layer calls `initialise()` in a background thread).
"""

from __future__ import annotations

import os
import time
import threading

import faiss
import numpy as np

from google import genai
from google.genai import types
from huggingface_hub import InferenceClient

from document_loader import load_documents
from chunker import split_text


# ==========================================================
# API TOKENS (read from env first; fallback to baked-in tokens)
# ==========================================================

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", ""
)
HF_TOKEN = os.environ.get(
    "HF_TOKEN", ""
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FOLDER = os.environ.get("FPVIBER_DATA_FOLDER", "data")

HF_EMBEDDING_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
GEMINI_MODEL = "gemini-3-flash-preview"

TOP_K = 4
BATCH_SIZE = 8

# Below this best-match cosine similarity, we consider the retrieval
# irrelevant and fall back to the "out of context" disclaimer.
RELEVANCE_THRESHOLD = 0.30

OUT_OF_CONTEXT_PREFIX = (
    "I do not have enough information in the documents, "
    "but based on general knowledge..."
)


def _log(msg: str) -> None:
    print(f"[rag] {msg}", flush=True)


# ==========================================================
# HUGGING FACE EMBEDDINGS - shape helper
# ==========================================================

def _normalize_embedding_output(raw_output, expected_count):
    """Convert raw HF embedding output to a clean 2D float32 numpy array."""
    arr = np.array(raw_output, dtype="float32")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[0] == expected_count:
            pass
        elif expected_count == 1:
            arr = arr.mean(axis=0, keepdims=True)
        else:
            raise ValueError(
                f"Unexpected 2D embedding shape: {arr.shape}, "
                f"expected_count={expected_count}"
            )
    elif arr.ndim == 3:
        arr = arr.mean(axis=1)
    else:
        raise ValueError(f"Unexpected embedding dimensions: {arr.ndim}")

    if arr.shape[0] != expected_count:
        raise ValueError(
            f"Embedding count mismatch. Expected {expected_count}, "
            f"got {arr.shape[0]}"
        )

    return arr.astype("float32")


# ==========================================================
# RAG ENGINE
# ==========================================================

class RAGEngine:
    """Encapsulates the whole RAG pipeline as a single in-memory object."""

    def __init__(self, data_folder: str = DATA_FOLDER):
        self.data_folder = data_folder
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.index = None
        self.ready = False
        self.status = "not_initialised"
        self.progress = {"current": 0, "total": 0}

        self._lock = threading.Lock()

        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        self.hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

    # ------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Load docs, chunk, embed and build FAISS index. Idempotent."""
        with self._lock:
            if self.ready:
                return

            self.status = "loading_documents"
            _log(f"Loading documents from '{self.data_folder}'...")
            documents = load_documents(self.data_folder)
            _log(f"Loaded {len(documents)} document(s).")

            self.status = "chunking_documents"
            _log("Chunking documents (recursive, table & math safe)...")
            self.chunks, self.sources = [], []
            for doc in documents:
                pieces = split_text(doc.text)
                for piece in pieces:
                    self.chunks.append(piece)
                    self.sources.append(doc.source)
            _log(f"Produced {len(self.chunks)} chunks.")

            self.status = "embedding_documents"
            _log("Embedding chunks via Hugging Face (IBM Granite)...")
            embeddings = self._embed_texts(self.chunks)

            self.status = "building_index"
            _log("Building FAISS index...")
            self.index = self._create_faiss_index(embeddings)

            self.ready = True
            self.status = "ready"
            _log(f"Ready. {self.index.ntotal} vectors indexed.")

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------

    def _hf_feature_extraction_with_retries(
        self, inputs, expected_count, max_retries: int = 5
    ):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = self.hf_client.feature_extraction(
                    inputs, model=HF_EMBEDDING_MODEL
                )
                return _normalize_embedding_output(
                    raw_output=result, expected_count=expected_count
                )
            except Exception as exc:
                last_error = exc
                _log(f"Embedding call failed (attempt {attempt}/{max_retries}): {exc}")
                if attempt == max_retries:
                    raise
                wait = attempt * 3
                _log(f"Retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"Embedding failed: {last_error}")

    def _embed_texts(self, texts, batch_size: int = BATCH_SIZE):
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        self.progress = {"current": 0, "total": total_batches}
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_num = start // batch_size + 1
            _log(f"  batch {batch_num}/{total_batches} ({len(batch)} items)...")
            embeddings = self._hf_feature_extraction_with_retries(
                inputs=batch, expected_count=len(batch)
            )
            all_embeddings.append(embeddings)
            self.progress = {"current": batch_num, "total": total_batches}
        return np.vstack(all_embeddings).astype("float32")

    def _embed_query(self, query: str):
        embedding = self._hf_feature_extraction_with_retries(
            inputs=query, expected_count=1
        )
        return embedding.astype("float32")

    # ------------------------------------------------------------------
    # FAISS
    # ------------------------------------------------------------------

    @staticmethod
    def _create_faiss_index(embeddings):
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """Return top-k retrieved chunks with score + source filename."""
        if not self.ready:
            raise RuntimeError("RAG engine is not ready yet.")

        query_embedding = self._embed_query(query)
        faiss.normalize_L2(query_embedding)

        scores, indexes = self.index.search(query_embedding, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indexes[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.sources[idx] if idx < len(self.sources) else None,
                "score": float(score),
            })
        return results

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------

    SYSTEM_PROMPT = (
        "You are FPViber, an FPV (first-person-view) drone technical assistant.\n"
        "You answer questions about building, flying, configuring and troubleshooting "
        "FPV drones, motors, ESCs, flight controllers, props, batteries and related "
        "regulations.\n"
        "\n"
        "STRICT GROUNDING RULES (these override everything else):\n"
        "1. You MUST first attempt to answer ONLY from the <CONTEXT> provided.\n"
        "2. NEVER invent, paraphrase or extrapolate facts that are not in the context "
        "as if they came from the documents.\n"
        "3. If the context does not contain enough information to answer the user's "
        "question, you MUST begin your reply with EXACTLY this sentence and on its "
        "own line:\n"
        f"   {OUT_OF_CONTEXT_PREFIX}\n"
        "   ...and only then add a short, clearly-labelled general-knowledge answer.\n"
        "4. If the question is empty, nonsensical, or completely unrelated to FPV, "
        "answer with the same out-of-context sentence and politely ask the user to "
        "clarify.\n"
        "5. Use the CONVERSATION history only to resolve references like 'it' or "
        "'the previous one'; never invent earlier turns.\n"
        "6. Keep answers concise, technical and step-by-step where appropriate.\n"
    )

    def ask_gemini(
        self,
        context: str,
        question: str,
        history: list[dict] | None = None,
        force_out_of_context: bool = False,
    ) -> str:
        """Call Gemini with the retrieved context, history and the user's question."""
        history_text = ""
        if history:
            lines = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            history_text = "\n".join(lines)

        if force_out_of_context or not context.strip():
            context_block = "(no relevant context retrieved from the documents)"
        else:
            context_block = context

        prompt = f"""{self.SYSTEM_PROMPT}

<CONVERSATION>
{history_text if history_text else "(no previous messages)"}
</CONVERSATION>

<CONTEXT>
{context_block}
</CONTEXT>

<USER_QUESTION>
{question}
</USER_QUESTION>

Answer:
"""

        response = self.gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=700,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (response.text or "").strip()

    # ------------------------------------------------------------------
    # High-level helper
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = TOP_K,
    ) -> dict:
        """
        Run the full retrieve + generate flow and return a structured dict:
        {
            "answer": str,
            "context": [ {text, source, score}, ... ],
            "grounded": bool,           # True if answered from context
            "out_of_context": bool,     # True if model fell back to general knowledge
            "sources": [filename, ...]  # de-duplicated source filenames actually used
        }
        """
        question = (question or "").strip()
        if not question:
            return {
                "answer": (
                    f"{OUT_OF_CONTEXT_PREFIX}\n"
                    "Please ask a question about FPV drones, components or building "
                    "techniques."
                ),
                "context": [],
                "grounded": False,
                "out_of_context": True,
                "sources": [],
            }

        retrieved = self.retrieve(question, k=k)

        # Decide if anything relevant came back.
        best_score = max((r["score"] for r in retrieved), default=0.0)
        force_oos = best_score < RELEVANCE_THRESHOLD

        if force_oos:
            context = ""
            context_for_ui: list[dict] = []
        else:
            context = "\n\n".join(item["text"] for item in retrieved)
            context_for_ui = retrieved

        answer_text = self.ask_gemini(
            context=context,
            question=question,
            history=history or [],
            force_out_of_context=force_oos,
        )

        out_of_context = (
            force_oos or answer_text.startswith(OUT_OF_CONTEXT_PREFIX)
        )

        sources: list[str] = []
        if not out_of_context:
            seen: set[str] = set()
            for item in context_for_ui:
                src = item.get("source")
                if src and src not in seen:
                    seen.add(src)
                    sources.append(src)

        return {
            "answer": answer_text,
            "context": context_for_ui,
            "grounded": not out_of_context,
            "out_of_context": out_of_context,
            "sources": sources,
        }
