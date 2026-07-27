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
import math
import base64
import queue
import random
import asyncio
import threading
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from groq import Groq
import google.generativeai as genai
import httpx

# ─────────────────────────────────────────────────────────────────────────
# 0. PATHS & ENV
# ─────────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent  # project root: where 01_documents.py ... live

# Load the ISOLATED backend .env (never the academic root .env, if any)
load_dotenv(BACKEND_DIR / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

FEEDBACK_LOG = BACKEND_DIR / "feedback_log.jsonl"

# Starter questions shown in the FAQ tab. Each maps to a real chat turn —
# clicking one in the UI sends it through the normal /api/chat + RAG pipeline
# rather than hard-coding an answer here, so responses stay grounded in the
# actual knowledge base instead of going stale.
FAQ_ITEMS = [
    {
        "category": "Irrigation Basics",
        "question": "How do I calculate the right drip irrigation duration for my field?",
        "answer": "Use the Calculators tab: enter your emitter count and flow rate per "
        "emitter to get total flow, volume, and per-emitter output. Combine that with "
        "the ET0 / crop water need estimate to size run time for your crop stage.",
    },
    {
        "category": "Irrigation Basics",
        "question": "What is a crop coefficient (Kc) and why does it matter?",
        "answer": "Kc adjusts reference evapotranspiration (ET0) for a specific crop and "
        "growth stage (initial, mid, late) to estimate actual crop water need (ETc = "
        "Kc x ET0). Look it up for your crop in the Calculators tab.",
    },
    {
        "category": "Using AquaMind",
        "question": "Can AquaMind analyze a photo of my field?",
        "answer": "Yes - open the Analyze Photo tab, upload a field or crop photo, "
        "optionally add a question, and the vision model will assess irrigation issues, "
        "crop health, and soil condition.",
    },
    {
        "category": "Using AquaMind",
        "question": "Why does every answer include a citation like [1] (source: ...)?",
        "answer": "AquaMind only answers technical questions using facts retrieved from "
        "the FAO manuals and irrigation guides in its knowledge base, and cites the exact "
        "source/page for every claim so you can verify it.",
    },
    {
        "category": "Troubleshooting",
        "question": "My soil moisture reading looks off - what should I check first?",
        "answer": "Ask AquaMind directly in the Chat tab (e.g. 'why would soil moisture "
        "readings drift?') - it will search the troubleshooting sections of the knowledge "
        "base and cite the relevant manual pages.",
    },
    {
        "category": "Troubleshooting",
        "question": "How accurate is the location-based water need estimate?",
        "answer": "The ET0 estimate uses a simplified seasonal climate model based on "
        "latitude, meant for quick planning - not a substitute for local weather-station "
        "ET0 data when precision matters.",
    },
]


def _call_tool(tool_obj, **kwargs):
    """Call a LangChain @tool-wrapped function directly, without going through
    the agent executor. Tries the underlying .func first (keeps this a plain,
    fast, synchronous call); falls back to .invoke() for older/newer LangChain
    tool implementations. The root prompting.py file is never modified."""
    func = getattr(tool_obj, "func", None)
    if callable(func):
        return func(**kwargs)
    return tool_obj.invoke(kwargs)

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
    """Mock IoT sensor widgets for the dashboard header with simulated dynamic jitter."""
    base_moisture = 34
    base_temp = 28.0
    base_hum = 62
    
    return {
        "soil_moisture_pct": max(0, min(100, base_moisture + random.randint(-2, 2))),
        "ambient_temp_c": round(base_temp + random.uniform(-1.5, 1.5), 1),
        "humidity_pct": max(0, min(100, base_hum + random.randint(-3, 3))),
        "recommended_irrigation_min": 18,
        "status": "Irrigation recommended within 6 hours",
    }


@app.post("/api/analyze-vision")
async def analyze_vision(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None)
):
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        print("🚨 GEMINI API ERROR: GEMINI_API_KEY is missing from backend/.env")
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    
    upload_file = file or image
    if not upload_file:
        raise HTTPException(status_code=400, detail="No image file provided in request.")
    
    try:
        contents = await upload_file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        mime_type = upload_file.content_type or "image/jpeg"
        
        user_prompt = prompt or "Analyze this crop/field image."
        full_prompt = (
            "Act as an expert agronomist. Analyze the provided image for plant health, "
            "nutrient deficiencies, pest issues, soil moisture status, or drip irrigation anomalies. "
            "Provide structured, action-oriented recommendations.\n\n"
            f"User prompt: {user_prompt}"
        )
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": full_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        }
        
        if gemini_key.startswith("AQ"):
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {gemini_key}"
            }
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            headers = {
                "Content-Type": "application/json"
            }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"🚨 GEMINI API ERROR: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Gemini REST API error: {error_msg}")
        
        data = response.json()
        analysis_text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"analysis": analysis_text, "result": analysis_text}
    except HTTPException:
        raise
    except Exception as e:
        print(f"🚨 GEMINI API ERROR: {type(e).__name__} -> {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {str(e)}")


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
