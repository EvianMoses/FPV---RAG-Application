"""
Flask web app exposing the FPViber RAG pipeline as a chat interface with
persistent SQLite-backed conversation memory.

All chat operations are scoped to a `session_id` URL parameter, and the
database layer enforces session isolation at the SQL level: every read /
write filters on `session_id`, so concurrent users cannot see or affect
each other's history.
"""

import os
import threading
import traceback

from flask import Flask, jsonify, render_template, request

import database
from rag_engine import RAGEngine


app = Flask(__name__)
engine = RAGEngine()


# ==========================================================
# BACKGROUND INITIALISATION
# ==========================================================
# Loading documents + computing embeddings can take a while. We do it in a
# background thread so the UI renders immediately and can poll /api/status.

_init_error: dict | None = None


def _initialise_engine_background():
    global _init_error
    try:
        engine.initialise()
    except Exception as exc:
        _init_error = {
            "message": str(exc),
            "type": exc.__class__.__name__,
        }
        traceback.print_exc()


def _start_background_init():
    database.init_db()
    thread = threading.Thread(
        target=_initialise_engine_background, daemon=True, name="rag-init"
    )
    thread.start()


# ==========================================================
# ROUTES
# ==========================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "ready": engine.ready,
        "status": engine.status,
        "chunks": len(engine.chunks),
        "documents": len(set(engine.sources)) if engine.sources else 0,
        "progress": engine.progress,
        "error": _init_error,
    })


# ---------- sessions ----------

@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    return jsonify({"sessions": database.list_sessions()})


@app.route("/api/sessions", methods=["POST"])
def api_create_session():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "New conversation").strip() or "New conversation"
    session = database.create_session(title=title)
    return jsonify(session), 201


@app.route("/api/sessions/<session_id>", methods=["GET"])
def api_get_session(session_id):
    session = database.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    session["messages"] = database.get_messages(session_id)
    return jsonify(session)


@app.route("/api/sessions/<session_id>", methods=["PATCH"])
def api_update_session(session_id):
    if not database.get_session(session_id):
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    database.update_session_title(session_id, title)
    return jsonify(database.get_session(session_id))


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    if not database.get_session(session_id):
        return jsonify({"error": "Session not found"}), 404
    database.delete_session(session_id)
    return jsonify({"ok": True})


# ---------- chat ----------

@app.route("/api/sessions/<session_id>/messages", methods=["POST"])
def api_send_message(session_id):
    if not engine.ready:
        return jsonify({
            "error": "FPViber engine is still initialising.",
            "status": engine.status,
        }), 503

    session = database.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    question = (payload.get("content") or "").strip()
    if not question:
        return jsonify({"error": "content is required"}), 400

    history = database.get_history_for_llm(session_id, limit=20)
    user_msg = database.add_message(session_id, "user", question)

    try:
        result = engine.answer(question=question, history=history)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "error": "Failed to generate answer.",
            "detail": str(exc),
            "user_message": user_msg,
        }), 500

    metadata = {
        "out_of_context": result.get("out_of_context", False),
        "grounded": result.get("grounded", True),
        "sources": result.get("sources", []),
    }

    assistant_msg = database.add_message(
        session_id,
        "assistant",
        result["answer"],
        context=result["context"],
        metadata=metadata,
    )

    if session["title"] == "New conversation":
        new_title = question[:60] + ("..." if len(question) > 60 else "")
        database.update_session_title(session_id, new_title)

    return jsonify({
        "user_message": user_msg,
        "assistant_message": assistant_msg,
    })


# ==========================================================
# ENTRY POINT
# ==========================================================

_start_background_init()


if __name__ == "__main__":
    host = os.environ.get("FPVIBER_HOST", "0.0.0.0")
    port = int(os.environ.get("FPVIBER_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
