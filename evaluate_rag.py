"""
evaluate_rag.py – Comprehensive RAG evaluation including FastAPI backend.

Evaluates:
  1. Retrieval quality (hit rate, MRR, precision/recall/F1 macro)
     - Comparison: pure vector vs hybrid vs hybrid+FlashRank
  2. Agent answer quality (LLM-judge: faithfulness/relevance/correctness)
  3. Tool accuracy (drip calc, ET0, Kc)
  4. FastAPI backend endpoints:
     - /api/health check
     - /api/telemetry (mock sensors)
     - /api/chat SSE streaming (response quality, latency, citations)
     - /api/analyze-vision (Gemini image analysis)
  5. Comparison: Streamlit agent vs FastAPI backend response quality
  6. Classification metrics: accuracy, precision, recall, F1 macro per category

Run:  python evaluate_rag.py
      (requires backend running: uvicorn main:app --port 8000)
"""

import os
import json
import time
import csv
import importlib.util
import requests
import threading
import queue

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

m06 = _import_module("retrieve_context.py")
m07 = _import_module("prompting.py")

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

BACKEND_URL = "http://localhost:8000"


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

class ClassificationMetrics:
    """Compute accuracy, precision, recall, F1 per category + macro averages."""

    def __init__(self):
        self.categories = {}

    def add(self, category, prediction, ground_truth):
        if category not in self.categories:
            self.categories[category] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        if prediction and ground_truth:
            self.categories[category]["tp"] += 1
        elif prediction and not ground_truth:
            self.categories[category]["fp"] += 1
        elif not prediction and ground_truth:
            self.categories[category]["fn"] += 1
        else:
            self.categories[category]["tn"] += 1

    def precision(self, c):
        d = self.categories.get(c, {})
        tp, fp = d.get("tp", 0), d.get("fp", 0)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, c):
        d = self.categories.get(c, {})
        tp, fn = d.get("tp", 0), d.get("fn", 0)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1(self, c):
        p, r = self.precision(c), self.recall(c)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def support(self, c):
        d = self.categories.get(c, {})
        return d.get("tp", 0) + d.get("fn", 0)

    def macro_precision(self):
        vals = [self.precision(c) for c in self.categories if self.support(c) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def macro_recall(self):
        vals = [self.recall(c) for c in self.categories if self.support(c) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def macro_f1(self):
        vals = [self.f1(c) for c in self.categories if self.support(c) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def accuracy(self):
        total_correct = sum(d.get("tp", 0) + d.get("tn", 0) for d in self.categories.values())
        total_all = sum(sum(d.values()) for d in self.categories.values())
        return total_correct / total_all if total_all > 0 else 0.0

    def weighted_f1(self):
        total_support = sum(self.support(c) for c in self.categories)
        if total_support == 0:
            return 0.0
        return sum(self.f1(c) * self.support(c) for c in self.categories) / total_support

    def report(self):
        lines = []
        lines.append(f"{'Category':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("-" * 57)
        for c in sorted(self.categories.keys()):
            s = self.support(c)
            if s > 0:
                lines.append(f"{c:<25} {self.precision(c):>10.3f} {self.recall(c):>10.3f} {self.f1(c):>10.3f} {s:>10}")
        lines.append("-" * 57)
        total = sum(self.support(c) for c in self.categories)
        lines.append(f"{'Macro Avg':<25} {self.macro_precision():>10.3f} {self.macro_recall():>10.3f} {self.macro_f1():>10.3f} {total:>10}")
        lines.append(f"{'Weighted Avg':<25} {self.macro_precision():>10.3f} {self.macro_recall():>10.3f} {self.weighted_f1():>10.3f} {total:>10}")
        lines.append(f"{'Accuracy':<25} {self.accuracy():>46.3f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SSE STREAM PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_sse_stream(response_text: str) -> dict:
    """Parse SSE event stream from FastAPI /api/chat endpoint.

    Returns dict with:
      - status: list of status messages
      - citations: list of citation dicts
      - tokens: list of token strings
      - full_answer: concatenated token string
      - errors: list of error messages
      - done: boolean
    """
    result = {
        "status": [],
        "citations": [],
        "tokens": [],
        "full_answer": "",
        "errors": [],
        "done": False,
    }

    for line in response_text.strip().split("\n"):
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[5:].strip())
            event_type = data.get("type", "")
            content = data.get("content", "")

            if event_type == "status":
                result["status"].append(content)
            elif event_type == "citations":
                result["citations"] = content if isinstance(content, list) else []
            elif event_type == "token":
                result["tokens"].append(content)
            elif event_type == "error":
                result["errors"].append(content)
            elif event_type == "done":
                result["done"] = True
        except json.JSONDecodeError:
            continue

    result["full_answer"] = "".join(result["tokens"])
    return result


def call_backend_chat(query: str, k: int = 5, history: list = None) -> dict:
    """Call FastAPI /api/chat endpoint and parse SSE response.

    Returns parsed SSE result dict with citations, full answer, etc.
    """
    payload = {
        "message": query,
        "role": "farmer",
        "k": k,
        "history": history or [],
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

        # Read full SSE stream
        full_text = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            full_text += chunk

        return parse_sse_stream(full_text)

    except requests.exceptions.ConnectionError:
        return {"errors": ["Backend not reachable — is uvicorn running on port 8000?"]}
    except Exception as e:
        return {"errors": [str(e)]}


def call_backend_vision(image_path: str, prompt: str = "") -> dict:
    """Call FastAPI /api/analyze-vision endpoint for image analysis."""
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"prompt": prompt or "Analyze this crop/field image."}
            response = requests.post(
                f"{BACKEND_URL}/api/analyze-vision",
                files=files,
                data=data,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Backend not reachable"}
    except Exception as e:
        return {"error": str(e)}


def check_backend_health() -> dict:
    """Check if FastAPI backend is running and configured."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "groq_configured": False}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def call_backend_telemetry() -> dict:
    """Get mock sensor data from /api/telemetry."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/telemetry", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# TEST SETS (same as before + backend-specific tests)
# ═══════════════════════════════════════════════════════════════════════════

RETRIEVAL_TEST_SET = [
    {"id": "R01", "query": "What is the recommended emitter spacing for sandy soil?",
     "expected_source": "fao_drip_irrigation_design.txt", "expected_page": 3,
     "expected_keywords": ["0.3", "0.5", "sandy", "spacing"], "category": "factual"},
    {"id": "R02", "query": "What is the Kc value for tomato at mid-season?",
     "expected_source": "crop_water_requirements.txt", "expected_page": 2,
     "expected_keywords": ["1.15", "tomato", "mid", "Kc"], "category": "factual"},
    {"id": "R03", "query": "How to troubleshoot a Hunter PGV valve that won't open?",
     "expected_source": "irrigation_troubleshooting.txt", "expected_page": 3,
     "expected_keywords": ["solenoid", "diaphragm", "24 VAC", "Hunter"], "category": "troubleshooting"},
    {"id": "R04", "query": "What is the field capacity of loam soil?",
     "expected_source": "soil_properties_guide.txt", "expected_page": 2,
     "expected_keywords": ["0.25", "0.32", "loam", "field capacity"], "category": "factual"},
    {"id": "R05", "query": "Penman-Monteith ET0 equation formula",
     "expected_source": "eto_calculation_methods.txt", "expected_page": 1,
     "expected_keywords": ["Penman", "Monteith", "ET0", "equation"], "category": "factual"},
    {"id": "R06", "query": "Drip irrigation filtration requirements mesh size",
     "expected_source": "fao_drip_irrigation_design.txt", "expected_page": 5,
     "expected_keywords": ["120", "200", "mesh", "filtration"], "category": "factual"},
    {"id": "R07", "query": "Why does soil moisture sensor read 100% but plants wilt?",
     "expected_source": "soil_properties_guide.txt", "expected_page": 5,
     "expected_keywords": ["sensor", "emitter", "root", "salinity"], "category": "troubleshooting"},
    {"id": "R08", "query": "Hargreaves ET0 calculation method",
     "expected_source": "eto_calculation_methods.txt", "expected_page": 2,
     "expected_keywords": ["Hargreaves", "temperature", "ET0"], "category": "factual"},
    {"id": "R09", "query": "Distribution uniformity DU formula catch-can test",
     "expected_source": "irrigation_troubleshooting.txt", "expected_page": 4,
     "expected_keywords": ["DU", "catch-can", "lowest 25%"], "category": "factual"},
    {"id": "R10", "query": "Total available water TAW formula soil water balance",
     "expected_source": "crop_water_requirements.txt", "expected_page": 5,
     "expected_keywords": ["TAW", "field capacity", "wilting point"], "category": "reasoning"},
    {"id": "R11", "query": "Emitter types point source inline micro-sprayer",
     "expected_source": "fao_drip_irrigation_design.txt", "expected_page": 2,
     "expected_keywords": ["point-source", "inline", "micro-sprayer"], "category": "factual"},
    {"id": "R12", "query": "Kc wheat initial stage value",
     "expected_source": "crop_water_requirements.txt", "expected_page": 2,
     "expected_keywords": ["wheat", "0.30", "initial"], "category": "factual"},
    {"id": "R13", "query": "Clay soil infiltration rate mm per hour",
     "expected_source": "soil_properties_guide.txt", "expected_page": 3,
     "expected_keywords": ["clay", "0.5", "infiltration"], "category": "factual"},
    {"id": "R14", "query": "ET0 for Cairo Egypt latitude 30 degrees",
     "expected_source": "eto_calculation_methods.txt", "expected_page": 4,
     "expected_keywords": ["Cairo", "30", "7.2"], "category": "factual"},
    {"id": "R15", "query": "Acid injection pH for drip system cleaning",
     "expected_source": "irrigation_troubleshooting.txt", "expected_page": 1,
     "expected_keywords": ["acid", "pH", "2-3"], "category": "factual"},
]

AGENT_TEST_SET = [
    {"id": "A01", "query": "What emitter spacing should I use for sandy soil?",
     "expected_tool": "search_knowledge_base",
     "expected_topics": ["sandy", "spacing", "0.3", "0.5"],
     "should_cite": True, "category": "factual"},
    {"id": "A02", "query": "Calculate total water volume for 500 emitters at 2 L/h operating 3 hours.",
     "expected_tool": "calculate_drip_irrigation",
     "expected_topics": ["1000", "3000", "500", "L/h", "3 hours"],
     "should_cite": False, "category": "calculation"},
    {"id": "A03", "query": "What is the Kc for maize at mid-season?",
     "expected_tool": "lookup_crop_coefficient",
     "expected_topics": ["maize", "1.20", "mid"],
     "should_cite": False, "category": "factual"},
    {"id": "A04", "query": "Soil moisture reads 100% but plants wilting. What could be wrong?",
     "expected_tool": "search_knowledge_base",
     "expected_topics": ["sensor", "root rot", "salinity", "placement"],
     "should_cite": True, "category": "troubleshooting"},
    {"id": "A05", "query": "ET0 for latitude 30 longitude 31?",
     "expected_tool": "get_reference_evapotranspiration",
     "expected_topics": ["ET0", "latitude", "mm"],
     "should_cite": False, "category": "calculation"},
    {"id": "A06", "query": "Troubleshoot Hunter PGV valve won't open.",
     "expected_tool": "search_knowledge_base",
     "expected_topics": ["solenoid", "diaphragm", "24 VAC"],
     "should_cite": True, "category": "troubleshooting"},
    {"id": "A07", "query": "What are soil water properties for loam?",
     "expected_tool": "search_knowledge_base",
     "expected_topics": ["loam", "0.25", "0.32", "field capacity"],
     "should_cite": True, "category": "factual"},
    {"id": "A08", "query": "Calculate drip: 200 emitters, 4 L/h each, 2 hours.",
     "expected_tool": "calculate_drip_irrigation",
     "expected_topics": ["800", "1600", "200", "4 L/h"],
     "should_cite": False, "category": "calculation"},
]

BACKEND_CHAT_TEST_SET = [
    {"id": "B01", "query": "What emitter spacing for sandy soil?",
     "expected_keywords": ["sandy", "spacing", "0.3", "0.5"],
     "should_have_citations": True, "category": "factual"},
    {"id": "B02", "query": "Calculate drip irrigation 500 emitters 2 L/h 3 hours",
     "expected_keywords": ["500", "1000", "3000", "L/h"],
     "should_have_citations": False, "category": "calculation"},
    {"id": "B03", "query": "Why soil moisture 100% but plants wilting?",
     "expected_keywords": ["sensor", "root", "salinity", "placement"],
     "should_have_citations": True, "category": "troubleshooting"},
    {"id": "B04", "query": "What is the Kc for tomato mid-season?",
     "expected_keywords": ["tomato", "1.15", "mid", "Kc"],
     "should_have_citations": True, "category": "factual"},
    {"id": "B05", "query": "ET0 for latitude 30 longitude 31",
     "expected_keywords": ["ET0", "latitude", "mm"],
     "should_have_citations": False, "category": "calculation"},
]

TOOL_TEST_SET = [
    {"id": "T01", "tool": "calculate_drip_irrigation",
     "input": {"num_emitters": 1000, "flow_rate_per_emitter": 2.0, "operation_hours": 4.0},
     "expected_output_contains": ["2000", "8000", "1000", "4 h"], "category": "calculation"},
    {"id": "T02", "tool": "lookup_crop_coefficient",
     "input": {"crop_name": "tomato", "growth_stage": "mid"},
     "expected_output_contains": ["1.15", "tomato", "mid"], "category": "factual"},
    {"id": "T03", "tool": "get_reference_evapotranspiration",
     "input": {"latitude": 30.0, "longitude": 31.0},
     "expected_output_contains": ["30", "ET0", "mm"], "category": "calculation"},
    {"id": "T04", "tool": "lookup_crop_coefficient",
     "input": {"crop_name": "cotton", "growth_stage": "initial"},
     "expected_keywords": ["0.35", "cotton", "initial"], "category": "factual"},
]


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    def __init__(self):
        self.results = []
        self.groq_api_key = m07.GROQ_API_KEY
        self.groq_model = m07.GROQ_MODEL

        self.retrieval_metrics = ClassificationMetrics()
        self.agent_metrics = ClassificationMetrics()
        self.tool_metrics = ClassificationMetrics()
        self.backend_metrics = ClassificationMetrics()

        self.retrieval_method_metrics = {}
        self.backend_available = False
        self.backend_health = {}

    def _is_doc_relevant(self, doc, test):
        src = doc.metadata.get("source", "")
        content = doc.page_content.lower()
        if test["expected_source"] in src:
            return True
        keyword_matches = sum(1 for kw in test["expected_keywords"] if kw.lower() in content)
        if keyword_matches >= 2:
            return True
        return False

    # ── RETRIEVAL ──

    def evaluate_retrieval(self):
        print("\n" + "=" * 70)
        print("📊 RETRIEVAL EVALUATION")
        print("=" * 70)

        methods = {
            "Pure Vector (similarity)": lambda q, k: m06.retrieve_context(q, k=k),
            "Pure Vector (MMR)": lambda q, k: m06.retrieve_context_mmr(q, k=k),
            "Hybrid + RRF + FlashRank": lambda q, k: m06.hybrid_search_with_reranking(q, k=k),
        }

        for method_name, search_fn in methods.items():
            print(f"\n🔍 Method: {method_name}")
            method_clf = ClassificationMetrics()
            self.retrieval_method_metrics[method_name] = method_clf

            mrr_scores = []
            latencies = []
            hit_count = 0
            page_match_count = 0

            for test in RETRIEVAL_TEST_SET:
                start = time.time()
                docs = search_fn(test["query"], k=5)
                latency = time.time() - start
                latencies.append(latency)

                relevant_in_results = 0
                found_source = False
                found_page = False
                first_relevant_rank = None

                for rank, doc in enumerate(docs):
                    is_rel = self._is_doc_relevant(doc, test)
                    if is_rel:
                        relevant_in_results += 1
                        if first_relevant_rank is None:
                            first_relevant_rank = rank + 1

                    src = doc.metadata.get("source", "")
                    pg = str(doc.metadata.get("page", ""))
                    if test["expected_source"] in src:
                        found_source = True
                        if pg == str(test["expected_page"]):
                            found_page = True

                prediction = relevant_in_results > 0
                method_clf.add(test["category"], prediction, True)
                method_clf.add(test["category"] + "_source", found_source, True)
                method_clf.add(test["category"] + "_page", found_page, True)

                if found_source: hit_count += 1
                if found_page: page_match_count += 1
                mrr_scores.append(1.0 / first_relevant_rank if first_relevant_rank else 0)

                self.results.append({
                    "test_id": test["id"], "category": test["category"],
                    "evaluation_type": "retrieval", "method": method_name,
                    "query": test["query"],
                    "found_source": found_source, "found_page": found_page,
                    "relevant_docs": relevant_in_results,
                    "latency_ms": round(latency * 1000, 1), "mrr": mrr_scores[-1],
                })

            n = len(RETRIEVAL_TEST_SET)
            print(f"  Hit Rate: {hit_count/n*100:.1f}% | Page Match: {page_match_count/n*100:.1f}%")
            print(f"  Avg MRR: {sum(mrr_scores)/n:.3f} | Latency: {sum(latencies)/n*1000:.1f}ms")
            print(f"\n  ── Classification Report ──")
            print(method_clf.report())

    # ── TOOLS ──

    def evaluate_tools(self):
        print("\n" + "=" * 70)
        print("🔧 TOOL EVALUATION")
        print("=" * 70)

        tools_map = {
            "calculate_drip_irrigation": m07.calculate_drip_irrigation,
            "lookup_crop_coefficient": m07.lookup_crop_coefficient,
            "get_reference_evapotranspiration": m07.get_reference_evapotranspiration,
        }

        for test in TOOL_TEST_SET:
            tool_fn = tools_map.get(test["tool"])
            if not tool_fn: continue

            result = tool_fn.invoke(test["input"])
            result_str = str(result)
            all_found = all(exp in result_str for exp in test["expected_output_contains"])
            self.tool_metrics.add(test["category"], all_found, True)
            status = "✅ PASS" if all_found else "❌ FAIL"
            print(f"  {test['id']}: {test['tool']} → {status}")
            if not all_found:
                missing = [e for e in test["expected_output_contains"] if e not in result_str]
                print(f"       Missing: {missing}")

            self.results.append({
                "test_id": test["id"], "category": test["category"],
                "evaluation_type": "tool", "method": test["tool"],
                "pass": all_found, "output": result_str[:200],
            })

        print(f"\n  ── Classification Report ──")
        print(self.tool_metrics.report())

    # ── AGENT (LLM-judge) ──

    def evaluate_agent(self):
        print("\n" + "=" * 70)
        print("🤖 AGENT EVALUATION (LLM-as-Judge)")
        print("=" * 70)

        if not self.groq_api_key:
            print("⚠️  GROQ_API_KEY not set.")
            return

        try:
            llm = m07.get_llm(temperature=0, streaming=False)
            agent_executor = m07.create_agent_executor(llm=llm)
        except Exception as e:
            print(f"⚠️  Agent failed: {e}")
            return

        judge_llm = ChatGroq(api_key=self.groq_api_key, model=self.groq_model, temperature=0)

        judge_prompt = """Score this answer 1-5:
QUESTION: {question}
EXPECTED: {expected}
ANSWER: {answer}
CONTEXT: {context}
Faithfulness, Relevance, Correctness, Citation Quality (1-5 each).
JSON ONLY: {"faithfulness":X,"relevance":X,"correctness":X,"citation_quality":X,"explanation":"brief"}"""

        scores = {"faithfulness": [], "relevance": [], "correctness": [], "citation_quality": []}