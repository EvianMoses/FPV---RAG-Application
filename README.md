# FPViber - FPV Drone Knowledge Assistant

> A retrieval-augmented generation (RAG) chat assistant for everything FPV: building, configuring, flying, and troubleshooting first-person-view drones.

FPViber is a small Flask app that loads your FPV technical documentation (PDF / DOCX / TXT), embeds it with the **IBM Granite multilingual embedding model**, indexes it in **FAISS**, and answers questions with **Google Gemini** under strict context-grounding rules.
**Created by Aviad Moshe.**

---



https://github.com/user-attachments/assets/4e2ac04e-4723-4f79-bfd7-b1e06c7490ba

<details>
  <summary>📸 Click here to view examples of questions asked in the application (Screenshots)</summary>
  <br>
  <img width="747" height="625" alt="Screenshot 2026-05-21 184941" src="https://github.com/user-attachments/assets/73ca89d4-86da-4ecc-8663-3c7140fe5c50" />
  <br>
  <img width="768" height="707" alt="Screenshot 2026-05-21 184915" src="https://github.com/user-attachments/assets/1d54c044-95aa-41fa-b8d8-ee12940a0c68" />
  <br>
  <img width="834" height="581" alt="Screenshot 2026-05-21 160414" src="https://github.com/user-attachments/assets/f78653ac-a87f-4a17-a54c-026fbc3daeac" />
  <br>
   <img width="690" height="698" alt="Screenshot 2026-05-21 150421" src="https://github.com/user-attachments/assets/80553a1c-b8a7-43cb-99fd-80f7bfb32be2" />
  <br>
   <img width="701" height="842" alt="Screenshot 2026-05-21 150516" src="https://github.com/user-attachments/assets/19526ef4-46a0-457f-8b62-36e9b60d0d53" />
</details>

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation & Knowledge Sources](#2-motivation--knowledge-sources)
3. [Features](#3-features)
4. [Architecture](#4-architecture)
5. [Technical Implementation](#5-technical-implementation)
   - [Supported File Types](#51-supported-file-types)
   - [Document Cleaning Pipeline](#52-document-cleaning-pipeline)
   - [Chunking Strategy](#53-chunking-strategy)
   - [Embedding Model](#54-embedding-model)
   - [Vector Database & Retrieval](#55-vector-database--retrieval)
   - [LLM & System Rules](#56-llm--system-rules)
   - [Frontend](#57-frontend)
6. [Project Layout](#6-project-layout)
7. [Quick Start](#7-quick-start)
8. [Environment Variables](#8-environment-variables)
9. [Docker](#9-docker)
10. [API Reference](#10-api-reference)
11. [Example Questions](#11-example-questions)
12. [Reflection — What Works & What Could Be Improved](#12-reflection--what-works--what-could-be-improved)

---

## 1. Project Overview

FPViber is a local Flask web application that ingests a curated set of FPV drone documentation and builds a searchable vector knowledge base. When a user asks a question, the engine retrieves the most relevant text chunks, feeds them as context to Google Gemini, and returns a grounded, step-by-step answer — rendered with full Markdown and LaTeX math support directly in the browser.

The system is intentionally **strict**: answers are always anchored to the retrieved context. When the knowledge base does not contain enough information, FPViber says so and labels any supplementary general knowledge explicitly.

---

## 2. Motivation & Knowledge Sources

### Why FPViber?

The FPV drone hobby is technically demanding. A single mistake — wrong motor KV for a given prop, incorrect ESC current rating, incorrect PID values — can destroy expensive hardware or create a safety hazard. Scattered blog posts and forum threads vary widely in quality, and beginners routinely burn out ESCs or crash builds because they relied on unverified information.

FPViber addresses this by providing **verified, targeted, and efficiently retrievable knowledge**. Instead of trawling through dozens of web pages, a builder can ask a precise question and receive a precise, source-cited answer in seconds.

### Knowledge Base (`data/`)

The documents in the `data/` folder come from a combination of:

- **Highly trusted public FPV resources**, including material drawn from sources such as:
  - [FPV Know It All](https://www.fpvknowitall.com/) — one of the most comprehensive FPV reference libraries available.
  - [Oscar Liang's Blog](https://oscarliang.com/) — widely regarded as the gold-standard technical reference for FPV builders.
- **Aviad Moshe's extensive personal expertise** in the FPV domain — practical, hands-on knowledge accumulated through real builds, tuning sessions, and flying experience.

By combining authoritative published sources with domain-expert curation, the knowledge base reflects the real-world complexity of modern FPV systems.

---

## 3. Features

| Feature | Details |
|---|---|
| **Multi-format ingestion** | `.txt`, `.pdf`, `.docx` |
| **Noise cleaning** | Strips page numbers, repeated headers/footers, soft hyphens |
| **Math-aware chunking** | Protects LaTeX blocks and pipe-delimited tables from being split |
| **IBM Granite embeddings** | 384-dim multilingual retrieval model via Hugging Face |
| **FAISS vector search** | Cosine similarity (L2-normalised inner product) |
| **Strict grounding** | Gemini only answers from retrieved context; out-of-context responses are labelled |
| **Markdown rendering** | `marked.js` parses Markdown to HTML with a custom typewriter effect |
| **LaTeX math rendering** | `KaTeX` renders formulas inline and in display blocks |
| **Source attribution** | Every grounded reply shows which document files it used |
| **Session memory** | Per-session SQLite history, isolated by `session_id` |
| **Dockerised** | Persistent `/app/data` and `/app/chat.db` volume mounts |

---

## 4. Architecture

```
┌──────────────┐    HTTP     ┌──────────────────────┐
│   Browser    │ ──────────▶ │  Flask  (app.py)     │
│  (FPViber UI)│  /api/...   │  routes + isolation  │
└──────────────┘             └──────────┬───────────┘
                                        │
               ┌────────────────────────┼──────────────────────────┐
               ▼                        ▼                          ▼
      ┌────────────────┐    ┌───────────────────────┐  ┌────────────────────┐
      │  database.py   │    │    rag_engine.py       │  │  document_loader   │
      │  SQLite        │    │  retrieve()            │  │  + chunker.py      │
      │  (chat.db)     │    │  ask_gemini()          │  │  clean + split     │
      │  session-safe  │    │  answer()              │  └────────────────────┘
      └────────────────┘    └──────────┬────────────┘
                                       │
              ┌────────────────────────┼──────────────────┐
              ▼                        ▼                   ▼
   ┌──────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
   │  Hugging Face    │  │  FAISS               │  │  Google Gemini   │
   │  IBM Granite     │  │  IndexFlatIP         │  │  gemini-3-flash  │
   │  (embeddings)    │  │  cosine similarity   │  │  (generation)    │
   └──────────────────┘  └──────────────────────┘  └──────────────────┘
```

The RAG engine initialises in a **background thread** at startup. The UI renders immediately and polls `/api/status` until the engine reports `ready`.

---

## 5. Technical Implementation

### 5.1 Supported File Types

Detected and loaded automatically from the `data/` folder by `document_loader.py`:

| Extension | Library | Notes |
|---|---|---|
| `.txt` | built-in | Direct UTF-8 read |
| `.pdf` | `pypdf` | Page-by-page text extraction |
| `.docx` | `python-docx` | Walks paragraphs and tables in document order; extracts OMML math |

### 5.2 Document Cleaning Pipeline

Each loaded file passes through a multi-step cleaning process before chunking:

1. **Format-specific extraction** — PDF pages extracted individually; DOCX paragraphs and tables walked in body order; OMML (Word equation objects) linearised to LaTeX-like `$...$` strings.
2. **Hyphenation repair** — Soft hyphens (`\u00ad`) and PDF line-break hyphens (`word-\nword`) are joined.
3. **Whitespace normalisation** — Multiple spaces, tabs, and excess blank lines are collapsed.
4. **Running header/footer detection** — Lines that appear verbatim at the top or bottom of ≥50% of pages are identified and stripped.
5. **Noise line removal** — Pure page numbers, single-character lines, and ornamental punctuation lines are dropped.
6. **Math formula preservation** — Lines matching equation patterns (LaTeX delimiters, assignment-style formulas, known FPV metric patterns) are wrapped in `\[...\]` so the downstream chunker treats them as atomic blocks.

### 5.3 Chunking Strategy

`chunker.py` implements a custom **protect-first, recursive character text splitter**.

#### Why not NLTK sentence splitting?

NLTK's `sent_tokenize` slices mid-formula (e.g., cutting `F =` from its right-hand side), breaks pipe-delimited spec tables across chunk boundaries, and separates numbered build steps from their context. FPViber replaces it entirely.

#### Pipeline

1. **Isolate math blocks** — Regex patterns match `$$...$$`, `\[...\]`, `\(...\)`, and `$...$`. All matched spans are marked as *protected* (atomic; do not split further).
2. **Isolate table blocks** — Consecutive lines containing `|` separators are glued together and also marked *protected*.
3. **Recursive prose splitting** — Free text between protected blocks is split on a decreasing hierarchy of separators:
   ```
   \n\n  →  \n  →  ". "  →  "? "  →  "! "  →  "; "  →  ", "  →  " "
   ```
4. **Greedy packing with overlap** — Pieces are greedily merged into chunks of **~900 characters** (`CHUNK_SIZE`). Adjacent chunks share **150 characters** of overlap (`CHUNK_OVERLAP`) so context bleeds across boundaries. Protected blocks larger than 900 chars are kept whole, capped at 1 600 chars (`HARD_MAX`).

#### Result

Build steps, motor/ESC spec tables, and thrust equations always land inside a single coherent chunk, making retrieved context directly usable by the LLM.

### 5.4 Embedding Model

**`ibm-granite/granite-embedding-97m-multilingual-r2`** (IBM, 97M parameters)

| Property | Value |
|---|---|
| Dimensions | 384 |
| Languages | Multilingual |
| Tuning | Dense retrieval (not just semantic similarity) |
| Serving | Hugging Face Inference API — no local model download |

This model was chosen because:
- **Retrieval-tuned** — technical FPV terminology (ESC, BLHeli, PID, DSHOT, betaflight) clusters semantically, not just lexically.
- **Multilingual** — documentation often mixes English and other languages.
- **Lightweight** — 384-dim vectors are fast to index and search in FAISS.
- **API-served** — the Docker image stays small; no multi-GB model weights shipped.

Embeddings are computed in batches of 8 (`BATCH_SIZE`) with up to 5 retries on transient API errors.

### 5.5 Vector Database & Retrieval

**FAISS `IndexFlatIP`** (Facebook AI Similarity Search, Flat Inner Product)

- All chunk embeddings and query embeddings are **L2-normalised** before indexing and querying, making inner-product search equivalent to **cosine similarity**.
- At query time the top `K = 4` chunks are retrieved (`TOP_K`).
- A **relevance threshold** of `0.30` (cosine similarity) filters out low-quality matches. If the best-scoring chunk falls below this threshold, the engine forces the *out-of-context* path — Gemini never receives irrelevant documents as context.

### 5.6 LLM & System Rules

**Model:** `gemini-3-flash-preview` (Google Gemini)

The engine constructs a structured prompt containing:

```
<SYSTEM PROMPT>   (rules below)
<CONVERSATION>    (last 20 turns of session history)
<CONTEXT>         (top-K retrieved chunks, or "no relevant context")
<USER_QUESTION>   (the current user message)
```

#### Complete System Rules

The following rules are injected into every Gemini call via `SYSTEM_PROMPT` in `rag_engine.py`:

| # | Rule |
|---|---|
| 1 | **Context-first grounding.** Answer ONLY from the `<CONTEXT>` provided. |
| 2 | **No invention.** Never invent, paraphrase, or extrapolate facts not present in the context as if they came from the documents. |
| 3 | **Out-of-context fallback.** If context is insufficient, the reply MUST begin with the exact sentence: *"I do not have enough information in the documents, but based on general knowledge"* — on its own line — followed by a clearly-labelled general-knowledge answer. |
| 4 | **Off-topic redirect.** If the input is a greeting or completely unrelated to FPV, reply with: *"Your question is unrelated to FPV or does not appear in the documents."* and guide the user toward FPV topics. |
| 5 | **Gibberish / unclear input.** If the message is empty, a single letter, or nonsensical, reply with: *"Your message is unclear."* and politely ask for clarification. The out-of-context prefix is NOT added in this case. |
| 6 | **History resolution only.** Use conversation history only to resolve pronouns like "it" or "the previous one". Never invent earlier turns. |
| 7 | **Concise and technical.** Keep answers concise and step-by-step where appropriate. |
| 8 | **Strict formula use.** For calculation questions (thrust, efficiency, torque, TWR, battery draw, PID gains, or any quantitative FPV metric): quote the formula from the documents first, then substitute values step by step. Never use a different formula unless the out-of-context disclaimer applies. |

In addition to the system prompt, a **similarity threshold** (`RELEVANCE_THRESHOLD = 0.30`) in `rag_engine.py` prevents low-quality chunks from reaching the LLM at all. If no chunk clears the threshold, `force_out_of_context=True` is passed to `ask_gemini`, and the context block is replaced with `"(no relevant context retrieved from the documents)"`.

**Gemini generation settings:** `temperature=0.2`, `max_output_tokens=700`, thinking budget disabled.

### 5.7 Frontend

| Technology | Role |
|---|---|
| **Flask / Jinja2** | Server-side routing and template rendering |
| **Vanilla JS** | Chat UI, session management, API calls |
| **[marked.js](https://marked.js.org/) v12** | Parses Markdown in LLM responses to HTML |
| **[KaTeX](https://katex.org/) v0.16** | Renders LaTeX math formulas (`$...$`, `\[...\]`) |
| Custom typewriter | DOM-walking character reveal — skips `.katex` subtrees to prevent corruption |
| Wave loader | Dark ↔ yellow CSS gradient animation while waiting for Gemini |
| **CSS custom properties** | Orbitron display font, black + yellow (`#facc15`) hi-tech palette |

---

## 6. Project Layout

```
.
├── app.py                 # Flask routes, background init, session isolation
├── rag_engine.py          # Embed / retrieve / ask Gemini + all system rules
├── document_loader.py     # TXT / PDF / DOCX extraction, cleaning, math wrapping
├── chunker.py             # Recursive character splitter (table & math safe)
├── database.py            # SQLite, session-isolated conversation memory
├── templates/
│   └── index.html         # FPViber UI + KaTeX integration
├── static/
│   ├── css/style.css      # Black + yellow hi-tech theme
│   └── js/app.js          # UI logic, Markdown, typewriter, wave loader
├── data/                  # YOUR knowledge base (.txt / .pdf / .docx)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 7. Quick Start

### Prerequisites

- Python 3.11+
- pip

```bash
# 1. Clone / cd into the project directory

# 2. Create and activate a virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows PowerShell:
venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your FPV documents to ./data
#    Supported: .txt, .pdf, .docx

# 5. Run the server
python app.py
# → http://localhost:5000
```

On first run the app embeds all documents via the Hugging Face Inference API — no local model download required. Subsequent runs reuse the same `chat.db` session history.

---

## 8. Environment Variables

All variables are optional. If not set, the application falls back to baked-in defaults.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | fallback token | Google Gemini API key |
| `HF_TOKEN` | fallback token | Hugging Face Inference API token |
| `FPVIBER_DATA_FOLDER` | `data` | Path to the knowledge base directory |
| `FPVIBER_DB_PATH` | `./chat.db` | SQLite database file path |
| `FPVIBER_HOST` | `0.0.0.0` | Flask bind host |
| `FPVIBER_PORT` | `5000` | Flask bind port |

---

## 9. Docker

### Build

```bash
docker build -t fpviber .
```

### Run with persistent data and conversation history

```bash
# Linux / macOS (bash)
docker run --rm -it \
  -p 5000:5000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/chat.db:/app/chat.db" \
  --name fpviber \
  fpviber
```

```powershell
# Windows PowerShell
docker run --rm -it `
  -p 5000:5000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/chat.db:/app/chat.db" `
  --name fpviber `
  fpviber
```

Then open <http://localhost:5000>.

**Volume mounts:**
- `/app/data` — bind-mount your local `./data` folder. Drop `.pdf` / `.docx` / `.txt` files there and restart the container to re-index.
- `/app/chat.db` — bind-mount the SQLite file so conversation history persists across container restarts.

> **Windows note:** Create `chat.db` before mounting (`New-Item chat.db` in PowerShell). Docker will otherwise create it as a directory.

---

## 10. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Engine status, chunk count, document count |
| `GET` | `/api/sessions` | List all sessions (most-recent first) |
| `POST` | `/api/sessions` | Create a new session |
| `GET` | `/api/sessions/<id>` | Get session metadata + full message history |
| `PATCH` | `/api/sessions/<id>` | Rename a session |
| `DELETE` | `/api/sessions/<id>` | Delete a session and all its messages |
| `POST` | `/api/sessions/<id>/messages` | Send a user message, receive assistant reply |

---

## 11. Example Questions

```
What is FPV?
How do I build a 5-inch freestyle drone from scratch?
What is the minimum budget for a complete FPV build?
What KV motor should I use with 5-inch props on a 4S battery?
How do I calculate the thrust-to-weight ratio of my drone?
What is the difference between BLHeli_S and BLHeli_32?
How do I tune PID values for a freestyle quad?
What are the Israeli FPV regulations?
What props work best for long-range FPV?
How does DSHOT differ from Multishot?
```

---

## 12. Reflection — What Works & What Could Be Improved

### What Works Well

- **Efficient multi-query handling** — The system accurately answers compound and multi-part questions within a single response, drawing from the correct context sections for each part.
- **Edge-case awareness** — Greetings, off-topic inputs, and gibberish are each handled by a dedicated rule, producing a clean, informative reply rather than a confusing or hallucinated one.
- **Math formula rendering** — LaTeX formulas from the LLM are rendered correctly as visual equations using KaTeX. The custom typewriter effect was engineered to avoid corrupting KaTeX's DOM nodes during animation.
- **Strict context grounding** — Answers are reliably anchored to the provided documents. The relevance threshold (`0.30`) ensures that a low-similarity retrieval triggers an explicit disclaimer rather than a confident but wrong answer.
- **Source attribution** — Every grounded reply displays yellow source chips showing which document files the answer came from, allowing users to verify information directly.

### What Could Be Improved

| Area | Description |
|---|---|
| **Response latency** | End-to-end latency (embedding query → FAISS search → Gemini generation) can be noticeable. Caching frequent query embeddings or switching to a faster Gemini tier would help. |
| **Granular question depth** | Very specific or highly detailed questions (e.g., exact torque specs for an obscure motor) may not be answered well if the database does not contain that level of detail. Enriching the knowledge base with more granular technical datasheets would improve coverage. |
| **Multimodal RAG (images & diagrams)** | The FPV world is heavily visual. Flight controller and ESC wiring diagrams, pin-out charts, and build photos are critical for assembly. The current pipeline is text-only. Adding support for parsing, embedding, and retrieving image content and wiring diagrams would be a significant capability upgrade and one of the highest-value future enhancements. |

---

## License

This project is provided for educational and portfolio purposes by Aviad Moshe. FPV documentation files in `./data` retain their original licenses from their respective sources.
