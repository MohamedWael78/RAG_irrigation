"""
07_prompting.py – LLM (Groq), tools, agent, and image description.

Uses Groq for ultra-fast inference:
  - Agent:    llama-3.3-70b-versatile (tool-calling, streaming)
  - Vision:   llama-3.2-90b-vision-preview (image analysis)
  - Speed:    ~300+ tokens/sec on Groq LPU hardware
"""

import math
import os
import base64
import json
import io
import importlib.util

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from PIL import Image

# ── Robust import for numbered filenames ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _import_module(filename):
    filepath = os.path.join(SCRIPT_DIR, filename)
    if not os.path.isfile(filepath):
        filepath = os.path.join(os.getcwd(), filename)
    if not os.path.isfile(filepath):
        raise ImportError(f"Cannot find: {filename}")
    module_name = os.path.splitext(filename)[0] + "_mod"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

retrieve_module = _import_module("retrieve_context.py")
docs_module     = _import_module("documents.py")

# ── API configuration ──
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL: str = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")

# ── Tables ──
KC_TABLE = {
    "tomato": {"initial": 0.60, "mid": 1.15, "late": 0.80},
    "wheat": {"initial": 0.30, "mid": 1.15, "late": 0.25},
    "maize": {"initial": 0.30, "mid": 1.20, "late": 0.50},
    "corn": {"initial": 0.30, "mid": 1.20, "late": 0.50},
    "rice": {"initial": 1.05, "mid": 1.20, "late": 0.90},
    "cotton": {"initial": 0.35, "mid": 1.20, "late": 0.65},
    "potato": {"initial": 0.40, "mid": 1.10, "late": 0.75},
    "onion": {"initial": 0.50, "mid": 1.05, "late": 0.75},
    "citrus": {"initial": 0.70, "mid": 0.85, "late": 0.65},
    "grape": {"initial": 0.30, "mid": 0.70, "late": 0.45},
    "grapevine": {"initial": 0.30, "mid": 0.70, "late": 0.45},
    "olive": {"initial": 0.55, "mid": 0.65, "late": 0.55},
    "alfalfa": {"initial": 0.40, "mid": 1.20, "late": 1.05},
    "lettuce": {"initial": 0.30, "mid": 1.00, "late": 0.90},
    "cabbage": {"initial": 0.40, "mid": 1.05, "late": 0.90},
    "sunflower": {"initial": 0.35, "mid": 1.15, "late": 0.35},
}

SOIL_TABLE = {
    "sandy": {"field_capacity": 0.09, "wilting_point": 0.03, "awc_mm_per_m": 60, "infiltration_mm_h": 120},
    "loamy sand": {"field_capacity": 0.13, "wilting_point": 0.05, "awc_mm_per_m": 80, "infiltration_mm_h": 50},
    "sandy loam": {"field_capacity": 0.21, "wilting_point": 0.08, "awc_mm_per_m": 130, "infiltration_mm_h": 40},
    "loam": {"field_capacity": 0.28, "wilting_point": 0.12, "awc_mm_per_m": 160, "infiltration_mm_h": 20},
    "silt loam": {"field_capacity": 0.33, "wilting_point": 0.14, "awc_mm_per_m": 190, "infiltration_mm_h": 12},
    "clay loam": {"field_capacity": 0.35, "wilting_point": 0.18, "awc_mm_per_m": 170, "infiltration_mm_h": 5},
    "clay": {"field_capacity": 0.42, "wilting_point": 0.24, "awc_mm_per_m": 180, "infiltration_mm_h": 2},
}

IMAGE_DESCRIPTION_PROMPT = (
    "You are an agricultural irrigation expert analyzing a figure, diagram, or photograph "
    "from an irrigation manual. Describe this image in detail:\n"
    "1. Type of visual (schematic, graph, photo, map, etc.)\n"
    "2. Key information about irrigation, soil, crops, or water management\n"
    "3. Any specific numbers, measurements, or data values visible\n"
    "4. How this relates to irrigation system design or operation\n\n"
    "Provide 2-4 paragraphs for a searchable knowledge base.\n"
)

IMAGE_DESCRIPTIONS_FILE = os.path.join("data", "images", "image_descriptions.json")

# ── Max image size for Groq vision (resize if larger) ──
MAX_IMAGE_SIZE = 1024  # pixels


def _resize_image(image_bytes: bytes, max_size: int = MAX_IMAGE_SIZE) -> bytes:
    """Resize image if too large for vision model limits."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        # Convert to RGB if RGBA (PNG transparency)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    except Exception:
        return image_bytes  # If resize fails, send original


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE DESCRIPTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def describe_image(image_path: str, source_pdf: str, page: int, image_index: int) -> dict:
    """Describe a single image using Groq vision model."""
    if not GROQ_API_KEY:
        image_id = f"{os.path.splitext(source_pdf)[0]}_page{page}_img{image_index}"
        return {
            "description": f"[Image from {source_pdf}, page {page} - API key not set]",
            "metadata": {
                "source": source_pdf.replace(".pdf", ".txt"),
                "page": page, "type": "image", "image_id": image_id,
            },
        }
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Resize for Groq vision limits
        image_bytes = _resize_image(image_bytes)
        encoded = base64.b64encode(image_bytes).decode("utf-8")

        vision_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_VISION_MODEL,
            temperature=0,
        )
        user_content = [
            {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ]
        response = vision_llm.invoke([HumanMessage(content=user_content)])
        description = response.content
    except Exception as e:
        description = f"[Image from {source_pdf}, page {page} - failed: {str(e)[:80]}]"

    image_id = f"{os.path.splitext(source_pdf)[0]}_page{page}_img{image_index}"
    return {
        "description": description,
        "metadata": {
            "source": source_pdf.replace(".pdf", ".txt"),
            "page": page, "type": "image", "image_id": image_id,
        },
    }


def describe_all_pdf_images() -> list[dict]:
    """Describe all extracted PDF images. Cached in JSON."""
    images_metadata = docs_module.load_image_metadata()
    if not images_metadata:
        print("📷 No PDF images to describe.")
        return []
    cached = _load_image_descriptions()
    if cached and len(cached) == len(images_metadata):
        print(f"📷 {len(cached)} image descriptions cached.")
        return cached
    descriptions = []
    for img in images_metadata:
        print(f"🔍 Describing: {img['filename']} (p.{img['page']} of {img['source_pdf']}) ...")
        desc = describe_image(img["path"], img["source_pdf"], img["page"], img["image_index"])
        descriptions.append(desc)
    _save_image_descriptions(descriptions)
    print(f"✅ Described {len(descriptions)} images.")
    return descriptions


def image_descriptions_to_documents(descriptions: list[dict]) -> list[Document]:
    return [Document(page_content=d["description"], metadata=d["metadata"]) for d in descriptions]


def _load_image_descriptions() -> list[dict] | None:
    if not os.path.isfile(IMAGE_DESCRIPTIONS_FILE):
        return None
    try:
        with open(IMAGE_DESCRIPTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_image_descriptions(descriptions: list[dict]):
    os.makedirs(os.path.dirname(IMAGE_DESCRIPTIONS_FILE), exist_ok=True)
    with open(IMAGE_DESCRIPTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)


def analyze_field_image(image_bytes: bytes, user_prompt: str = "") -> str:
    """Analyze a user-uploaded field photo using Groq vision model."""
    if not GROQ_API_KEY:
        return "⚠️ Cannot analyze: GROQ_API_KEY not configured."

    # Resize for Groq vision limits
    image_bytes = _resize_image(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    analysis_prompt = (
        "You are an expert agricultural irrigation consultant analyzing a field photograph.\n"
        "Examine and provide detailed analysis:\n\n"
        "1. **Visual Assessment**: crop type, growth stage, soil, infrastructure\n"
        "2. **Irrigation Issues**: over/under-irrigation, waterlogging, emitter problems\n"
        "3. **Crop Health**: leaf color, wilting, pest/disease symptoms\n"
        "4. **Soil Condition**: cracking, erosion, salinity, moisture\n"
        "5. **Recommendations**: specific actionable steps\n\n"
        f"User question: {user_prompt if user_prompt else 'General analysis'}\n"
    )

    vision_llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_VISION_MODEL,
        temperature=0,
    )
    user_content = [
        {"type": "text", "text": analysis_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
    ]
    response = vision_llm.invoke([HumanMessage(content=user_content)])
    return response.content


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def search_knowledge_base(query: str) -> str:
    """Searches FAO irrigation manuals, crop water requirements, soil types,
    troubleshooting guides, AND image/diagram descriptions. Uses hybrid
    BM25+Vector+RRF+FlashRank. Always cite [n] markers."""
    docs = retrieve_module.hybrid_search_with_reranking(query, k=5)
    if not docs:
        return "No relevant documents found in the knowledge base."
    return retrieve_module.format_retrieved_docs(docs)


@tool
def calculate_drip_irrigation(num_emitters: int, flow_rate_per_emitter: float, operation_hours: float) -> str:
    """Calculates total flow rate, water volume, and per-emitter volume for a
    drip irrigation system given emitter count, flow rate per emitter, and
    operation duration. Returns raw numeric results for the given inputs."""
    total_flow_rate = num_emitters * flow_rate_per_emitter
    total_volume_liters = total_flow_rate * operation_hours
    total_volume_m3 = total_volume_liters / 1000.0
    volume_per_emitter = flow_rate_per_emitter * operation_hours
    flow_l_s = total_flow_rate / 3600.0
    return (f"Drip Calculation (RAW NUMBERS):\n━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Emitters: {num_emitters} | Rate: {flow_rate_per_emitter} L/h\n"
            f"• Total Flow: {total_flow_rate} L/h ({flow_l_s:.4f} L/s)\n"
            f"• Hours: {operation_hours} h\n"
            f"• Volume: {total_volume_liters:.1f} L ({total_volume_m3:.4f} m³)\n"
            f"• Per emitter: {volume_per_emitter:.1f} L\n━━━━━━━━━━━━━━━━━━━━━\n")


@tool
def get_reference_evapotranspiration(latitude: float, longitude: float) -> str:
    """Estimates daily/weekly/monthly reference evapotranspiration (ET0) for
    a given latitude and longitude using a simple seasonal climate model.
    Use this before calling lookup_crop_coefficient to get ETc."""
    abs_lat = abs(latitude)
    if abs_lat < 15: s,w,a = 4.8,3.7,4.2
    elif abs_lat < 30: s,w,a = 6.8,3.0,4.8
    elif abs_lat < 45: s,w,a = 6.3,2.0,4.0
    elif abs_lat < 60: s,w,a = 4.2,1.2,2.5
    else: s,w,a = 2.5,0.5,1.5
    amp = (s-w)/2.0
    cur = a + amp * math.sin(2*math.pi*(180-80)/365)
    return (f"ET0 (RAW NUMBERS):\n━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Location: ({latitude:.4f}, {longitude:.4f})\n"
            f"• Daily: {cur:.2f} mm/d | Weekly: {cur*7:.1f} mm | Monthly: {cur*30:.1f} mm\n"
            f"• Summer: {s:.1f} | Winter: {w:.1f} | Annual avg: {a:.1f} mm/d\n━━━━━\n")


@tool
def lookup_crop_coefficient(crop_name: str, growth_stage: str = "mid") -> str:
    """Looks up the crop coefficient (Kc) for a given crop and growth stage
    (initial, mid, or late) and estimates crop water need (ETc) from it.
    Returns available crop names if the requested crop isn't found."""
    cl = crop_name.strip().lower()
    sl = growth_stage.strip().lower()
    if sl not in ("initial","mid","late"):
        return f"Invalid stage. Use: initial, mid, or late."
    kd = KC_TABLE.get(cl)
    if kd is None:
        return f"'{crop_name}' not found. Available: {', '.join(sorted(KC_TABLE.keys()))}"
    kc = kd[sl]
    etc = kc * 5.0
    return (f"Kc (RAW NUMBERS):\n━━━━━━━━━━━━━━━━━━━━━\n"
            f"• {crop_name} @ {growth_stage}: Kc={kc}\n"
            f"• ETc = {kc}×5.0 = {etc:.2f} mm/d | Weekly: {etc*7:.1f} | Monthly: {etc*30:.1f}\n"
            f"• All: ini={kd['initial']}, mid={kd['mid']}, late={kd['late']}\n━━━━━\n")


# ═══════════════════════════════════════════════════════════════════════════
# LLM + AGENT (GROQ)
# ═══════════════════════════════════════════════════════════════════════════

def get_llm(temperature=0, streaming=True):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Configure via Streamlit secrets.")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=temperature,
        streaming=streaming,
    )


TOOLS = [search_knowledge_base, calculate_drip_irrigation,
         get_reference_evapotranspiration, lookup_crop_coefficient]


def build_system_prompt(field_lat=30.0444, field_lon=31.2357):
    return (
        "You are 'AquaMind', expert irrigation AI. You have a knowledge base "
        "(hybrid BM25+Vector+FlashRank, text AND image descriptions), a calculator, "
        "ET0 estimator, and Kc lookup. Answer accurately.\n"
        f"Default location: lat {field_lat}, lon {field_lon}.\n\n"

        "CRITICAL GROUNDING & HALLUCINATION PREVENTION RULES:\n"
        "1. When answering technical questions, you MUST use the `search_knowledge_base` tool first.\n"
        "2. If `search_knowledge_base` returns 'No relevant documents found' or if the retrieved context is insufficient:\n"
        "   - State clearly: 'I do not have sufficient information in my knowledge base to answer this question accurately.'\n"
        "   - DO NOT fabricate, guess, or synthesize information from outside the retrieved knowledge base.\n\n"

        "STRICT CITATION RULES:\n"
        "1. You MUST include citations directly inside your final answer for any fact retrieved from the knowledge base.\n"
        "2. Follow this exact citation pattern in your output text: [Index] (source: filename.txt, page: X)\n"
        "   Example: Drip emitter flow rate should be 2.0 L/h [1] (source: irrigation_manual.txt, page: 14).\n"
        "3. For type=image sources, mention it is a diagram or figure.\n\n"

        "EXECUTION RULES:\n"
        "1. After a tool call, DO NOT call unnecessary extra tools. Answer immediately.\n"
        "2. Always display RAW NUMBERS from calculations.\n"
    )


def create_agent_executor(llm=None, field_lat=30.0444, field_lon=31.2357):
    if llm is None:
        llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", build_system_prompt(field_lat, field_lon)),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent, tools=TOOLS, verbose=False,
        max_iterations=10, handle_parsing_errors=True,
    )


def get_soil_properties(soil_type):
    d = SOIL_TABLE.get(soil_type.strip().lower())
    if not d:
        return None
    return (f"Soil {soil_type}: FC={d['field_capacity']}, WP={d['wilting_point']}, "
            f"AWC={d['awc_mm_per_m']}mm/m, Inf={d['infiltration_mm_h']}mm/h\n")