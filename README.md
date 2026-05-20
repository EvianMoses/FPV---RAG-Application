# FPViber - FPV Drone Assistant

> Hi-tech, retrieval-augmented chat assistant for everything FPV: building, flying, configuring and troubleshooting first-person-view drones.

FPViber is a small Flask app that loads your FPV technical documentation (PDF / DOCX / TXT), embeds it with the **IBM Granite multilingual embedding model**, indexes it in **FAISS**, and answers questions with **Google Gemini** under strict context-grounding rules.

---

## Features

- **Multi-format ingestion**: `.txt`, `.pdf` (via `pypdf`) and `.docx` (via `python-docx`).
- **Cleaning pipeline**: strips page numbers, recurring headers/footers, soft hyphens and noise lines.
- **Table- and math-aware chunking**: recursive character text splitter that *protects* pipe-separated tables and LaTeX-style math (`$...$`, `\(...\)`, `\[...\]`) from being broken mid-block.
- **IBM Granite embeddings** via the Hugging Face Inference API.
- **FAISS cosine-similarity retrieval** (inner product on L2-normalised vectors).
- **Strict grounding**: Gemini is instructed to answer *only* from retrieved context; if nothing relevant comes back (similarity below threshold) the answer starts with a deterministic *"I do not have enough information in the documents, but based on general knowledge..."* disclaimer.
- **Hi-tech UI** (Orbitron headings, black + yellow palette) with a custom drone-flying loading animation, source chips, and FPV iconography.
- **Per-session SQLite memory** isolated by `session_id` - safe for multi-user use.
- **Dockerised** with explicit data + DB volume mounts.

---

## Architecture

```
┌──────────────┐    HTTP    ┌──────────────────────┐
│   Browser    │ ─────────▶ │  Flask (app.py)      │
│  (FPViber UI)│  /api/...  │  - routes            │
└──────────────┘            │  - session isolation │
                            └──────────┬───────────┘
                                       │
                ┌──────────────────────┼─────────────────────────┐
                ▼                      ▼                         ▼
       ┌────────────────┐    ┌────────────────────┐    ┌──────────────────┐
       │ database.py    │    │ rag_engine.py      │    │ document_loader/ │
       │  SQLite        │    │  - retrieve()      │    │ chunker.py       │
       │  (chat.db)     │    │  - ask_gemini()    │    │  - clean         │
       │  per-session   │    │  - answer()        │    │  - split         │
       └────────────────┘    └────────┬───────────┘    └──────────────────┘
                                      │
                  ┌───────────────────┼──────────────────────┐
                  ▼                   ▼                      ▼
         ┌────────────────┐  ┌─────────────────┐  ┌────────────────────┐
         │ Hugging Face   │  │ FAISS           │  │ Google Gemini      │
         │ (Granite       │  │ IndexFlatIP     │  │ gemini-3-flash     │
         │  embeddings)   │  │ cosine similar. │  │  (text generation) │
         └────────────────┘  └─────────────────┘  └────────────────────┘
```

The app starts the RAG initialisation **in a background thread** so the UI renders immediately. The frontend polls `/api/status` until the engine reports `ready`.

---

## Embedding model

**`ibm-granite/granite-embedding-97m-multilingual-r2`** is a 97-M-parameter dense retrieval encoder from IBM. We chose it because:

- It is **multilingual** - useful since FPV documentation often mixes English and the user's native language.
- It produces **384-dim vectors**, cheap to store and search in FAISS.
- It is **retrieval-tuned** (not just sentence similarity), so technical FPV terminology (ESC, BLHeli, PID, props, betaflight, etc.) is mapped into a space where related concepts cluster well.
- It is **served on the Hugging Face Inference API**, so we don't ship multi-GB model weights inside our Docker image.

We L2-normalise every vector and use `faiss.IndexFlatIP`, which makes inner-product search equivalent to **cosine similarity**. A configurable threshold (`RELEVANCE_THRESHOLD = 0.30`) decides whether the best-matching chunk is *similar enough* - if not, the engine forces the "out of context" branch.

---

## Chunking strategy

Naive sentence-splitters (e.g. NLTK `sent_tokenize`) wreak havoc on technical drone PDFs: they slice tables of motor specs mid-row, cut "1." off numbered build steps, and split a thrust-equation in the middle of a `\[ ... \]` block.

`chunker.py` implements a custom **recursive character text splitter** with a *protect-first* strategy:

1. **Isolate math blocks** matching `$$...$$`, `\[...\]`, `\(...\)`, `$...$`.
2. **Isolate tables** - consecutive lines containing `|` separators are glued and marked atomic.
3. **Recursively split the prose** using a hierarchy of separators:
   `paragraph break ▶ line break ▶ sentence (`. `, `? `, `! `) ▶ clause (`; `, `, `) ▶ word`.
4. **Greedy pack** the pieces back into chunks of ~`CHUNK_SIZE` (900 chars) with `CHUNK_OVERLAP` (150 chars) of overlap so neighbouring context bleeds across boundaries.

Protected blocks larger than `CHUNK_SIZE` are kept whole (capped at `HARD_MAX = 1600` chars). The result: build steps, motor spec tables and thrust formulas stay coherent inside a single chunk.

---

## Project layout

```
.
├── app.py                # Flask routes + background init
├── rag_engine.py         # Embed / retrieve / ask Gemini + grounding rules
├── document_loader.py    # TXT / PDF / DOCX extraction + cleaning
├── chunker.py            # Recursive character splitter (table/math safe)
├── database.py           # SQLite, session-isolated
├── templates/index.html  # FPViber UI
├── static/
│   ├── css/style.css     # Black + yellow hi-tech theme
│   └── js/app.js         # UI logic + drone loader animation
├── data/                 # YOUR knowledge base (.txt / .pdf / .docx)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Quick start (local Python)

```bash
# 1. Clone / cd into the project
# 2. Create a venv (Python 3.11+)
python -m venv venv
. venv/Scripts/activate     # Windows PowerShell:  venv\Scripts\Activate.ps1

# 3. Install deps
pip install -r requirements.txt

# 4. Drop FPV docs into ./data (PDF / DOCX / TXT)

# 5. Run
python app.py
# -> http://localhost:5000
```

The first launch downloads no model files: embeddings are computed by the Hugging Face Inference API.

### Environment variables

| Variable                 | Default            | Purpose                                          |
|--------------------------|--------------------|--------------------------------------------------|
| `GEMINI_API_KEY`         | baked-in fallback  | Override the Gemini API key                      |
| `HF_TOKEN`               | baked-in fallback  | Override the Hugging Face token                  |
| `FPVIBER_DATA_FOLDER`    | `data`             | Path to the knowledge base                       |
| `FPVIBER_DB_PATH`        | `./chat.db`        | SQLite DB file location                          |
| `FPVIBER_HOST`           | `0.0.0.0`          | Bind host                                        |
| `FPVIBER_PORT`           | `5000`             | Bind port                                        |

---

## Docker

### Build

```bash
docker build -t fpviber .
```

### Run with persistent data + DB

```bash
docker run --rm -it \
  -p 5000:5000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/chat.db:/app/chat.db" \
  --name fpviber \
  fpviber
```

PowerShell equivalent:

```powershell
docker run --rm -it `
  -p 5000:5000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/chat.db:/app/chat.db" `
  --name fpviber `
  fpviber
```

Then open <http://localhost:5000>.

- `/app/data` is the knowledge base mount - drop your `.pdf` / `.docx` / `.txt` files in `./data` on the host and they show up inside the container.
- `/app/chat.db` is the SQLite file mount - your conversation history survives container restarts and rebuilds.

> **Note**: on Windows, make sure `chat.db` exists on the host before mounting it (`type nul > chat.db` in cmd, or `New-Item chat.db` in PowerShell) - Docker will otherwise create it as a directory.

---

## Robustness & safety

- **Strict grounding**: see the `SYSTEM_PROMPT` constant in `rag_engine.py`. Gemini is forbidden from inventing facts and must start with a deterministic disclaimer when context is missing.
- **Relevance threshold**: best-match cosine similarity below `0.30` ⇒ "out of context" path.
- **Empty query**: handled before any LLM call - returns the standard disclaimer.
- **Session isolation**: every read/write on the `messages` table is filtered on `session_id`. There is no API that returns messages without a session filter. The schema enforces `FOREIGN KEY (session_id) ON DELETE CASCADE`.
- **Chunking integrity**: tables and math blocks are extracted as atomic units *before* splitting, so the recursive splitter cannot break them. NLTK is no longer in the pipeline.
- **Source attribution**: every assistant reply that was grounded in documents ships a `metadata.sources` list which the UI renders as yellow source chips under the message. The full retrieved chunks (with scores) are available in an expandable "Retrieved chunks" panel.

---

## API surface

| Method | Path                                        | Description                       |
|--------|---------------------------------------------|-----------------------------------|
| GET    | `/api/status`                               | Engine status + chunk count       |
| GET    | `/api/sessions`                             | List sessions                     |
| POST   | `/api/sessions`                             | Create session                    |
| GET    | `/api/sessions/<id>`                        | Session + messages                |
| PATCH  | `/api/sessions/<id>`                        | Rename session                    |
| DELETE | `/api/sessions/<id>`                        | Delete session (cascades)         |
| POST   | `/api/sessions/<id>/messages`               | Send user message, get reply      |

---

## License

This project is provided as-is for educational / portfolio use. FPV documentation in `./data` retains its original license.
