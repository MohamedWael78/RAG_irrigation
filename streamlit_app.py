"""
streamlit_app.py – AquaMind AI with full Image Reader system & Analytics Dashboard.
"""

import os
import re
import base64
import json
import importlib
import pandas as pd
import numpy as np

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# ── Import pipeline modules ──
m01 = importlib.import_module("documents")
m02 = importlib.import_module("preprocessing")
m03 = importlib.import_module("chunking")
m04 = importlib.import_module("vector_representation")
m05 = importlib.import_module("create_chroma_store")
m06 = importlib.import_module("retrieve_context")
m07 = importlib.import_module("prompting")

# ── API key from Streamlit secrets (GROQ) ──
try:
    if not m07.GROQ_API_KEY:
        m07.GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    m07.GROQ_MODEL = st.secrets.get("GROQ_MODEL", m07.GROQ_MODEL)
    m07.GROQ_VISION_MODEL = st.secrets.get("GROQ_VISION_MODEL", m07.GROQ_VISION_MODEL)
except Exception:
    pass

# ── Auto-build knowledge base ──
def ensure_knowledge_base():
    chroma_dir = "chroma_db"
    chunks_file = os.path.join("data", "chunks", "chunks.json")
    need_build = not os.path.isdir(chroma_dir) or not os.listdir(chroma_dir)

    if need_build:
        with st.spinner("🔧 Building knowledge base from PDFs ..."):
            m01.create_documents()
            m02.preprocess_all_documents()
            m03.chunk_all_documents()
            embedding_model = m04.get_embedding_model()
            m05.create_chroma_store(embedding_model=embedding_model)

    desc_file = os.path.join("data", "images", "image_descriptions.json")
    images_meta = m01.load_image_metadata()
    if images_meta and m07.GROQ_API_KEY and not os.path.isfile(desc_file):
        with st.spinner("📷 Describing PDF images ..."):
            descriptions = m07.describe_all_pdf_images()
            if descriptions:
                img_docs = m07.image_descriptions_to_documents(descriptions)
                embedding_model = m04.get_embedding_model()
                m05.add_documents_to_store(img_docs, embedding_model=embedding_model)
                st.success(f"✅ {len(descriptions)} images described & in KB!")
    else:
        try:
            vs = m06.get_vector_store()
            test_results = vs.similarity_search("diagram irrigation schematic", k=1, filter={"type": "image"})
            if not test_results:
                with st.spinner("📷 Adding cached image descriptions to knowledge base ..."):
                    with open(desc_file, "r", encoding="utf-8") as f:
                        descriptions = json.load(f)
                    img_docs = m07.image_descriptions_to_documents(descriptions)
                    embedding_model = m04.get_embedding_model()
                    m05.add_documents_to_store(img_docs, embedding_model=embedding_model)
        except Exception:
            pass

ensure_knowledge_base()

# ── Citation utilities ──
def parse_citations(text: str) -> list[dict]:
    citations = []
    pattern = r'\[(\d+)\]\s*\(\s*source:\s*([^,]+)\s*,\s*page:\s*([^)]+)\s*\)\s*:\s*(.{0,120})'
    for match in re.finditer(pattern, text):
        citations.append({
            "index": int(match.group(1)),
            "source": match.group(2).strip(),
            "page": match.group(3).strip(),
            "snippet": match.group(4).strip() + " ...",
        })
    return citations

def render_citation_cards_html(citations: list[dict]) -> str:
    if not citations:
        return ""
    cards = []
    for c in citations:
        pg = f"p.{c['page']}" if c['page'] != "?" else "unknown page"
        cards.append(
            f'<div style="background:rgba(14,165,183,0.06);border:1px solid rgba(14,165,183,0.2);'
            f'border-radius:8px;padding:8px 12px;margin:4px 0;font-family:JetBrains Mono,monospace;'
            f'font-size:0.72rem;color:#0e1b1a;">'
            f'<span style="background:#0ea5b7;color:white;padding:2px 6px;border-radius:4px;'
            f'font-weight:600;margin-right:6px;">[{c["index"]}]</span>'
            f'<b>{c["source"]}</b> · {pg}<br>'
            f'<span style="color:#5b6b67;">{c["snippet"]}</span>'
            f'</div>'
        )
    return (
        f'<div style="padding:4px 0;">'
        f'<div style="font-family:Space Grotesk,sans-serif;font-size:0.68rem;font-weight:600;'
        f'color:#0ea5b7;margin-bottom:4px;letter-spacing:0.08em;">📚 SOURCES</div>'
        + "".join(cards) + '</div>'
    )

def build_chat_history(messages: list[dict]) -> list:
    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))
    return history

# ── Page config + design ──
st.set_page_config(page_title="AquaMind AI", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    :root {
        --c-canvas-a:#f4f7f5;--c-canvas-b:#eef4f2;--c-canvas-c:#eaf6f5;
        --c-ink:#0e1b1a;--c-muted:#5b6b67;--c-border:#dbe6e2;
        --c-flow:#0ea5b7;--c-flow-dark:#0b8494;--c-growth:#67a63c;--c-growth-dark:#4f7f2c;--c-clay:#c97a3d;
        --font-display:'Space Grotesk',sans-serif;--font-body:'Plus Jakarta Sans',sans-serif;--font-mono:'JetBrains Mono',monospace;
    }
    *{font-family:var(--font-body);}
    h1,h2,h3,h4{font-family:var(--font-display)!important;}
    code,.stCodeBlock,[data-testid="stMetricValue"]{font-family:var(--font-mono)!important;}
    .stApp{
        background:linear-gradient(rgba(14,165,183,0.05) 1px,transparent 1px) 0 0/100% 34px,
        linear-gradient(90deg,rgba(14,165,183,0.05) 1px,transparent 1px) 0 0/34px 100%,
        radial-gradient(circle at top right,var(--c-canvas-c) 0%,var(--c-canvas-b) 40%,var(--c-canvas-a) 100%);
    }
    .brand-row{display:flex;align-items:center;justify-content:center;gap:14px;margin-bottom:2px;}
    .brand-mark{flex:0 0 auto;filter:drop-shadow(0 3px 8px rgba(14,165,183,0.35));}
    .main-title{font-family:var(--font-display)!important;color:var(--c-ink);font-weight:700;font-size:2.6rem!important;letter-spacing:-0.5px;margin:0;}
    .main-title .accent{color:var(--c-flow);}
    .subtitle{text-align:center;color:var(--c-muted);font-size:1.05rem;font-weight:500;margin-top:2px;}
    .flow-meter{display:flex;justify-content:center;margin:14px auto 6px auto;}
    .flow-meter svg{overflow:visible;}
    .status-strip{display:flex;justify-content:center;flex-wrap:wrap;gap:18px;margin:4px 0 22px 0;padding:8px 0;border-top:1px solid var(--c-border);border-bottom:1px solid var(--c-border);}
    .status-item{display:flex;align-items:center;gap:6px;font-family:var(--font-mono);font-size:0.68rem;letter-spacing:0.06em;color:var(--c-muted);text-transform:uppercase;}
    .status-dot{width:7px;height:7px;border-radius:50%;box-shadow:0 0 0 3px rgba(14,165,183,0.12);}
    [data-testid="stChatMessage"]{position:relative;background-color:rgba(255,255,255,0.88);backdrop-filter:blur(12px);border-radius:14px;box-shadow:0 4px 15px rgba(14,27,26,0.04);padding:22px 20px 18px 20px;margin-bottom:20px;margin-top:8px;border:1px solid var(--c-border);transition:transform 0.2s ease;}
    [data-testid="stChatMessage"]:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(14,27,26,0.07);}
    [data-testid="stChatMessage"][data-test-user="assistant"]{border-left:3px solid var(--c-growth);}
    [data-testid="stChatMessage"][data-test-user="user"]{border-left:3px solid var(--c-flow);background-color:rgba(248,250,252,0.92);}
    [data-testid="stChatMessage"][data-test-user="assistant"]::before{content:"AGENT";position:absolute;top:-9px;left:16px;font-family:var(--font-mono);font-size:0.6rem;font-weight:600;letter-spacing:0.14em;background:var(--c-growth);color:white;padding:2px 8px;border-radius:5px;}
    [data-testid="stSidebar"] [data-testid="stNumberInput"]>div>div{background-color:rgba(7,19,18,0.9)!important;border:1px solid rgba(14,165,183,0.4)!important;border-radius:8px!important;}
    [data-testid="stSidebar"] input[type="number"]{color:#7be0ea!important;-webkit-text-fill-color:#7be0ea!important;background-color:transparent!important;font-family:var(--font-mono)!important;font-weight:600!important;}
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button{color:#7be0ea!important;background-color:rgba(14,165,183,0.1)!important;}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background-color:rgba(7,19,18,0.9)!important;border:1.5px dashed rgba(14,165,183,0.5)!important;border-radius:12px!important;}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p{color:#b8d8d4!important;-webkit-text-fill-color:#b8d8d4!important;}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button{background-color:rgba(14,165,183,0.2)!important;color:#0ea5b7!important;border:1px solid #0ea5b7!important;}
    .stButton>button{background-color:#ffffff;color:var(--c-ink);border:1px solid var(--c-border);border-radius:10px;padding:10px 18px;box-shadow:0 2px 4px rgba(0,0,0,0.02);transition:all 0.2s ease;text-align:left;font-weight:500;font-size:0.92rem;}
    .stButton>button:hover{border-color:var(--c-flow);background-color:#ecfbfa;color:var(--c-flow-dark);box-shadow:0 4px 12px rgba(14,165,183,0.14);}
    .streamlit-expanderHeader{background-color:#f1f6f5!important;border-radius:8px!important;border:1px solid var(--c-border)!important;color:var(--c-ink)!important;}
    [data-testid="stChatInput"]{border-radius:12px!important;border:1px solid #cfe0dc!important;background-color:white!important;box-shadow:0 10px 25px rgba(14,27,26,0.07)!important;}
    [data-testid="stChatInput"] textarea{color:var(--c-ink)!important;}
</style>
""", unsafe_allow_html=True)

# ── Header ──
_TICKS = "".join(f'<line x1="{20+i*26}" y1="16" x2="{20+i*26}" y2="24" stroke="#c3d6d1" stroke-width="1.5"/>' for i in range(12))

st.markdown(f"""
<div class="brand-row">
  <svg class="brand-mark" width="40" height="40" viewBox="0 0 48 48">
    <defs><linearGradient id="dropGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#0ea5b7"/><stop offset="100%" stop-color="#67a63c"/></linearGradient></defs>
    <path d="M24 4 C24 4 8 25 8 34 A16 16 0 0 0 40 34 C40 25 24 4 24 4 Z" fill="url(#dropGrad)"/>
    <path d="M15 31 Q24 21 33 31" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" opacity="0.85"/>
  </svg>
  <h1 class="main-title">AquaMind<span class="accent"> AI</span></h1>
</div>
<p class="subtitle">Next-Gen Smart Irrigation Agent &amp; Hydraulic Calculator</p>
<div class="flow-meter">
  <svg width="320" height="34" viewBox="0 0 320 34" xmlns="http://www.w3.org/2000/svg">
    <line x1="12" y1="20" x2="308" y2="20" stroke="#c3d6d1" stroke-width="2"/>
    {_TICKS}
    <circle r="5" fill="url(#dropGrad2)"><animateMotion dur="3.4s" repeatCount="indefinite" path="M12,20 L308,20"/></circle>
    <defs><linearGradient id="dropGrad2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#0ea5b7"/><stop offset="100%" stop-color="#67a63c"/></linearGradient></defs>
  </svg>
</div>
<div class="status-strip">
  <span class="status-item"><span class="status-dot" style="background:#0ea5b7"></span>BM25 + Vector RRF</span>
  <span class="status-item"><span class="status-dot" style="background:#67a63c"></span>FlashRank Reranker</span>
  <span class="status-item"><span class="status-dot" style="background:#0ea5b7"></span>GPT-4o-mini Vision</span>
  <span class="status-item"><span class="status-dot" style="background:#c97a3d"></span>4 Tools + Image Reader</span>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🌿 AquaMind Console")
    st.caption("SYSTEM STATUS: ONLINE")

    st.markdown("#### ⚙️ Agent Dashboard")
    cap_col, tech_col = st.columns(2)
    with cap_col:
        st.markdown("""
        **Capabilities**
        🧠 Hydraulic Design
        🌱 Crop Coefficients
        🛠️ Hardware Troubleshoot
        🌤️ ET0 Estimation
        📷 Image Reader
        """)
        try:
            vs = m06.get_vector_store()
            doc_count = vs._collection.count()
            img_count = len(m01.load_image_metadata())
            st.markdown(f"📚 **Text Chunks:** {doc_count - img_count}")
            st.markdown(f"📷 **Image Descriptions:** {img_count}")
        except Exception:
            st.markdown("📚 **KB Loaded:** ✅")

    with tech_col:
        st.markdown("""
        **Tech Stack**
        <span style='font-size:0.7em;background-color:rgba(14,165,183,0.15);color:#0ea5b7;padding:2px 6px;border-radius:4px;margin:1px;display:inline-block;'>BM25</span>
        <span style='font-size:0.7em;background-color:rgba(14,165,183,0.15);color:#0ea5b7;padding:2px 6px;border-radius:4px;margin:1px;display:inline-block;'>RRF Fusion</span>
        <span style='font-size:0.7em;background-color:rgba(14,165,183,0.15);color:#0ea5b7;padding:2px 6px;border-radius:4px;margin:1px;display:inline-block;'>FlashRank</span>
        <span style='font-size:0.7em;background-color:rgba(14,165,183,0.15);color:#0ea5b7;padding:2px 6px;border-radius:4px;margin:1px;display:inline-block;'>Vision</span>
        <span style='font-size:0.7em;background-color:rgba(14,165,183,0.15);color:#0ea5b7;padding:2px 6px;border-radius:4px;margin:1px;display:inline-block;'>ChromaDB</span>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 📷 Document Images")
    extracted_images = m01.load_image_metadata()

    if extracted_images:
        with st.expander(f"📷 View {len(extracted_images)} Extracted PDF Images", expanded=False):
            desc_file = os.path.join("data", "images", "image_descriptions.json")
            descriptions = {}
            if os.path.isfile(desc_file):
                with open(desc_file, "r", encoding="utf-8") as f:
                    desc_list = json.load(f)
                    for d in desc_list:
                        iid = d["metadata"].get("image_id", "")
                        descriptions[iid] = d["description"]

            for img in extracted_images[:10]:
                img_id = f"{os.path.splitext(img['source_pdf'])[0]}_page{img['page']}_img{img['image_index']}"
                caption = f"p.{img['page']} of {img['source_pdf']}"

                if os.path.isfile(img["path"]):
                    st.image(img["path"], caption=caption, use_container_width=True)
                    desc = descriptions.get(img_id, "")
                    if desc:
                        st.markdown(
                            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                            f"color:#5b6b67;padding:4px 8px;background:rgba(14,165,183,0.06);"
                            f"border-radius:6px;margin:4px 0;'>🔍 {desc[:150]}...</div>",
                            unsafe_allow_html=True,
                        )

            if len(extracted_images) > 10:
                st.caption(f"... and {len(extracted_images) - 10} more images")
    else:
        st.caption("No images extracted from PDFs")

    if extracted_images and m07.OPENROUTER_API_KEY:
        desc_file = os.path.join("data", "images", "image_descriptions.json")
        if not os.path.isfile(desc_file):
            if st.button("🔍 Describe All PDF Images", use_container_width=True):
                with st.spinner("Describing images with vision model ..."):
                    descriptions = m07.describe_all_pdf_images()
                    if descriptions:
                        img_docs = m07.image_descriptions_to_documents(descriptions)
                        embedding_model = m04.get_embedding_model()
                        m05.add_documents_to_store(img_docs, embedding_model=embedding_model)
                        st.success(f"✅ {len(descriptions)} images described & added to KB!")
                        st.rerun()

    st.markdown("---")

    st.markdown("#### 🧮 Engineering Tools")

    with st.expander("💧 Drip Calculator", expanded=False):
        drip_emitters = st.number_input("Number of Emitters", min_value=1, value=500, key="drip_emit")
        drip_flow = st.number_input("Flow Rate (L/h per emitter)", min_value=0.1, value=2.0, format="%.1f", key="drip_flow")
        drip_hours = st.number_input("Operation Hours", min_value=0.1, value=1.5, format="%.1f", key="drip_hours")
        if st.button("💧 Calculate Drip Volume", use_container_width=True):
            st.session_state.active_question = (
                f"Calculate drip irrigation: {drip_emitters} emitters, {drip_flow} L/h each, "
                f"operating {drip_hours} hours. Show all raw numbers."
            )

    with st.expander("🧪 Soil Estimator", expanded=False):
        soil_types = ["Sandy","Loamy Sand","Sandy Loam","Loam","Silt Loam","Clay Loam","Clay"]
        selected_soil = st.selectbox("Select Soil Type", soil_types, key="soil_select")
        if st.button("🧪 Get Soil Properties", use_container_width=True):
            st.session_state.active_question = f"Estimate soil properties for {selected_soil} soil. Show raw numbers."

    st.markdown("---")

    st.markdown("#### 🌤️ Field Context")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        field_lat = st.number_input("Latitude", value=30.0444, format="%.4f", key="lat_input")
    with col_lon:
        field_lon = st.number_input("Longitude", value=31.2357, format="%.4f", key="lon_input")
    if st.button("🌤️ Get ET0 Forecast", use_container_width=True):
        st.session_state.active_question = f"ET0 forecast for lat {field_lat}, lon {field_lon}. Show raw numbers."

    st.markdown("---")

    st.markdown("#### 📷 Field Image Analyzer")
    st.caption("Upload a field photo for AI analysis")

    uploaded_image = st.file_uploader(
        "Upload field image",
        type=["jpg", "jpeg", "png"],
        key="field_image_upload",
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Your field image", use_container_width=True)

        analysis_presets = [
            "General Assessment",
            "🔍 Identify Irrigation Issues",
            "🌱 Assess Crop Health",
            "🧪 Identify Soil Type",
            "🛠️ Diagnose Equipment Problems",
        ]
        selected_analysis = st.selectbox("Analysis Type", analysis_presets, key="analysis_type")

        custom_prompt = st.text_input(
            "Additional question (optional)",
            placeholder="e.g., What's causing the yellow leaves?",
            key="custom_image_prompt",
        )

        if st.button("🔍 Analyze Image", use_container_width=True):
            image_bytes = uploaded_image.read()

            preset_prompts = {
                "General Assessment": "Provide a general assessment of this field image.",
                "🔍 Identify Irrigation Issues": "Focus on identifying any irrigation problems: over/under-irrigation, emitter issues, waterlogging, drought stress.",
                "🌱 Assess Crop Health": "Focus on crop health: leaf color, wilting, growth stage, pest/disease symptoms.",
                "🧪 Identify Soil Type": "Focus on soil identification: texture, color, cracking, moisture level, erosion signs.",
                "🛠️ Diagnose Equipment Problems": "Focus on irrigation equipment: visible damage, leaks, valve issues, pipe problems.",
            }

            full_prompt = preset_prompts.get(selected_analysis, "General assessment")
            if custom_prompt:
                full_prompt += f"\n\nUser's specific question: {custom_prompt}"

            with st.spinner("🔍 Vision model analyzing your field image ..."):
                try:
                    analysis = m07.analyze_field_image(image_bytes, full_prompt)
                    st.session_state.image_analysis = analysis
                    st.session_state.image_analysis_prompt = full_prompt
                    st.session_state.active_question = f"📷 Field Image Analysis ({selected_analysis}): {custom_prompt if custom_prompt else full_prompt}"
                except Exception as e:
                    st.error(f"Image analysis failed: {e}")

    st.markdown("---")

    with st.expander("💡 Knowledge Base Q&A"):
        kb_questions = [
            "Recommended emitter spacing for sandy soil?",
            "Soil moisture reads 100% but plants are wilting, why?",
            "What is the Kc value for tomatoes at mid-season?",
            "How to troubleshoot a Hunter PGV valve that won't open?",
            "Show me diagrams of drip irrigation layouts",
        ]
        for q in kb_questions:
            if st.button(f"› {q}", key=q):
                st.session_state.active_question = q

    st.markdown("---")
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── LLM + Agent ──
try:
    llm = m07.get_llm(temperature=0, streaming=True)
    agent_executor = m07.create_agent_executor(llm=llm, field_lat=field_lat, field_lon=field_lon)
    llm_available = True
except ValueError:
    llm_available = False
    st.warning("⚠️ OPENROUTER_API_KEY not configured.")

# ── Chat state ──
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Welcome to AquaMind! 👋 I'm your Smart Irrigation AI with **Image Reader** support.\n\n"
            "Upload a field image in the sidebar or ask me anything!"
        ),
        "citations": [],
    }]

if "active_question" not in st.session_state:
    st.session_state.active_question = None

if "image_analysis" not in st.session_state:
    st.session_state.image_analysis = None

# ═══════════════════════════════════════════════════════════════════════════
# 🚀 LAYOUT: ANALYTICS DASHBOARD (Left) + CHAT (Right)
# ═══════════════════════════════════════════════════════════════════════════

col_dashboard, col_chat = st.columns([1.4, 1.0], gap="large")

# ── LEFT PANEL: Analytics Dashboard ──
with col_dashboard:
    st.markdown("### 📊 لوحة المراقبة والتحليل الزراعي")
    st.caption("Field Analytics Dashboard")
    
    # 1. Quick Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(label="معدل ETo اليومي", value="4.8 mm/day", delta="-0.2")
    m2.metric(label="احتياج المزرعة المائي", value="12,500 L", delta="+500 L")
    m3.metric(label="كفاءة الري", value="88%", delta="1.5%")
    
    st.markdown("---")
    
    # 2. Charts
    st.markdown("#### 📈 توقعات البخر-نتح (ETo) للأيام القادمة")
    chart_data = pd.DataFrame(
        np.random.randn(7, 2) + [4.5, 3.8],
        columns=["ETo المتوقع (mm)", "ETo الموصى به"]
    )
    st.line_chart(chart_data)


# ── RIGHT PANEL: Agent Chat ──
with col_chat:
    st.markdown("### 💬 المساعد الهندسي")
    st.caption("Ask AquaMind about your data")
    
    chat_container = st.container(height=520)
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
                if message.get("citations"):
                    try:
                        components.html(render_citation_cards_html(message["citations"]), height=140, scrolling=True)
                    except Exception:
                        pass
    
    input_text = st.chat_input("Ask about irrigation, crops, soil... 📷")

    if st.session_state.active_question:
        input_text = st.session_state.active_question
        st.session_state.active_question = None

    if input_text:
        with chat_container:
            st.chat_message("user").markdown(input_text)
            
            if st.session_state.image_analysis:
                answer = st.session_state.image_analysis
                analysis_prompt = st.session_state.image_analysis_prompt
                
                st.chat_message("assistant").markdown(answer, unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "user", "content": f"📷 {analysis_prompt}", "citations": []})
                st.session_state.messages.append({"role": "assistant", "content": answer, "citations": []})
                
                st.session_state.image_analysis = None
                st.session_state.image_analysis_prompt = None
            else:
                st.session_state.messages.append({"role": "user", "content": input_text, "citations": []})
                chat_history = build_chat_history(st.session_state.messages[:-1])
                
                with st.chat_message("assistant"):
                    with st.spinner("AquaMind Engineering Manager is working..."):
                        try:
                            if not llm_available:
                                answer = "⚠️ OPENROUTER_API_KEY not configured."
                                all_citations = []
                            else:
                                response = agent_executor.invoke(
                                    {"input": input_text, "chat_history": chat_history},
                                    config={"return_intermediate_steps": True},
                                )
                                answer = response["output"]
                                all_citations = []
                                seen_sources = set()

                                for step in response.get("intermediate_steps", []):
                                    action = step[0]
                                    if action.tool == "search_knowledge_base":
                                        tool_output = step[1]
                                        output_text = getattr(tool_output, "content", None) or str(tool_output)
                                        for c in parse_citations(output_text):
                                            key = (c["source"], c["page"])
                                            if key not in seen_sources:
                                                seen_sources.add(key)
                                                all_citations.append(c)

                            st.markdown(answer, unsafe_allow_html=True)

                            if all_citations:
                                try:
                                    components.html(render_citation_cards_html(all_citations), height=140, scrolling=True)
                                except Exception:
                                    pass

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "citations": all_citations,
                            })

                        except Exception as e:
                            err_text = str(e)
                            if "tool_call" in err_text.lower():
                                st.error("Tool call formatting issue. Try rephrasing with explicit numbers.")
                            else:
                                st.error(f"Error: {e}")