"""
backend/main.py – FastAPI bridge for the AquaMind Agricultural RAG project.

Responsibilities:
  1. Dynamically load the existing root-level pipeline modules
     (retrieve_context.py, prompting.py) WITHOUT touching or renaming them —
     the academic submission in the project root stays untouched.
  2. Expose POST /api/chat which:
       a. runs hybrid retrieval (BM25 + Vector + RRF + FlashRank) against
          ChromaDB via retrieve_context.hybrid_search_with_reranking()
       b. sends a "citations" SSE event with the retrieved chunks/sources
       c. streams the Groq (llama-3.3-70b-versatile) answer token-by-token
          as "token" SSE events
       d. finishes with a "done" event
  3. Exposes GET /api/health and GET /api/telemetry (mock sensor widgets).

Run with:
    uvicorn main:app --reload --port 8000
(from inside the backend/ directory, with backend/.env populated)
"""

import os
import sys
import json
import queue
import random
import asyncio
import threading
import importlib.util
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────
# 0. PATHS & ENV
# ─────────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent  # project root: where 01_documents.py ... live

# Load the ISOLATED backend .env (never the academic root .env, if any)
load_dotenv(BACKEND_DIR / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")

if not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY is not set in backend/.env — /api/chat will fail until it is.")

# The root pipeline modules use bare `import`/`importlib.import_module`
# calls internally (e.g. retrieve_context.py imports "vector_representation").
# Adding ROOT_DIR to sys.path lets those bare imports resolve without any
# changes to the original files.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Also make sure Groq env vars are visible to prompting.py at import time,
# since it reads os.environ.get(...) at module load.
os.environ.setdefault("GROQ_API_KEY", GROQ_API_KEY)
os.environ.setdefault("GROQ_MODEL", GROQ_MODEL)
os.environ.setdefault("GROQ_VISION_MODEL", GROQ_VISION_MODEL)

# Run inside ROOT_DIR context so relative paths in the pipeline
# (chroma_db/, data/images/, etc.) resolve correctly regardless of where
# uvicorn was launched from.
_original_cwd = os.getcwd()
os.chdir(ROOT_DIR)


def _import_root_module(filename: str):
    """Load a .py file from the project root by path, mirroring the same
    pattern prompting.py already uses for its own numbered-file imports."""
    filepath = ROOT_DIR / filename
    if not filepath.is_file():
        raise ImportError(f"Cannot find root file: {filepath}")
    module_name = filepath.stem + "_root_mod"
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ── Import the existing academic pipeline files directly (unmodified) ──
retrieve_module = _import_root_module("retrieve_context.py")
prompting_module = _import_root_module("prompting.py")

os.chdir(_original_cwd)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ─────────────────────────────────────────────────────────────────────────
# 1. FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AquaMind RAG Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────
# 2. SCHEMAS
# ─────────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    role: Literal["farmer", "engineer"] = "farmer"
    history: list[ChatMessage] = []
    k: int = 5


# ─────────────────────────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _docs_to_citations(docs) -> list[dict]:
    """Convert LangChain Documents from ChromaDB into JSON-serializable
    citation cards for the frontend (used by both Farmer and Engineer views)."""
    citations = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        citations.append(
            {
                "index": i,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
                "type": meta.get("type", "text"),
                "snippet": doc.page_content[:280].strip(),
                "full_content": doc.page_content,
            }
        )
    return citations


def _stream_groq_tokens(messages: list[dict], model: str, out_queue: "queue.Queue"):
    """Runs in a background thread: consumes the (blocking) Groq streaming
    iterator and pushes each token onto a thread-safe queue so the async
    SSE generator can yield them without blocking the event loop."""
    try:
        stream = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                out_queue.put(("token", token))
        out_queue.put(("done", None))
    except Exception as e:  # noqa: BLE001
        out_queue.put(("error", str(e)))


# ─────────────────────────────────────────────────────────────────────────
# 4. ROUTES
# ─────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "chat_model": GROQ_MODEL,
        "vision_model": GROQ_VISION_MODEL,
    }


@app.get("/api/telemetry")
def telemetry():
    """Mock IoT sensor widgets for the dashboard header."""
    return {
        "soil_moisture_pct": 34,
        "ambient_temp_c": 28,
        "humidity_pct": 62,
        "recommended_irrigation_min": 18,
        "status": "Irrigation recommended within 6 hours",
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if groq_client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on the server.")

    async def event_generator():
        try:
            # ── Step 1: hybrid retrieval (BM25 + Vector + RRF + FlashRank) ──
            # This is blocking CPU/IO work, so run it off the event loop.
            yield _sse({"type": "status", "content": "Searching knowledge base..."})
            docs = await asyncio.to_thread(
                retrieve_module.hybrid_search_with_reranking, req.message, req.k
            )
            citations = _docs_to_citations(docs)
            context_text = retrieve_module.format_retrieved_docs(docs)

            # ── Step 2: emit citations immediately so the UI can render
            # source cards / chunk previews while the answer streams in ──
            yield _sse({"type": "citations", "content": citations})

            # ── Step 3: build the prompt using the existing system prompt ──
            system_prompt = prompting_module.build_system_prompt()
            history_messages = [{"role": m.role, "content": m.content} for m in req.history]
            user_turn = (
                f"Knowledge Base Context:\n{context_text}\n\n"
                f"User Question ({req.role} view): {req.message}"
            )
            messages = (
                [{"role": "system", "content": system_prompt}]
                + history_messages
                + [{"role": "user", "content": user_turn}]
            )

            # ── Step 4: stream Groq tokens via a background thread + queue ──
            token_queue: "queue.Queue" = queue.Queue()
            thread = threading.Thread(
                target=_stream_groq_tokens,
                args=(messages, GROQ_MODEL, token_queue),
                daemon=True,
            )
            thread.start()

            loop = asyncio.get_event_loop()
            while True:
                kind, payload = await loop.run_in_executor(None, token_queue.get)
                if kind == "token":
                    yield _sse({"type": "token", "content": payload})
                elif kind == "error":
                    yield _sse({"type": "error", "content": payload})
                    break
                else:  # "done"
                    yield _sse({"type": "done"})
                    break

        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "content": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
