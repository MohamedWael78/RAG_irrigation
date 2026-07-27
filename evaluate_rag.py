"""
evaluate_rag.py – Comprehensive RAG evaluation with classification metrics.

Metrics computed:
  - Accuracy, Precision, Recall, F1 (per category + macro average)
  - Retrieval: hit rate, MRR, source/page match, doc-level TP/FP/FN
  - Generation: LLM-judge faithfulness/relevance/correctness/citation
  - Tools: accuracy with expected output verification
  - Latency: retrieval + agent response times

Classification approach:
  - Retrieval: TP = found correct source, FP = retrieved wrong source, FN = missed correct source
  - Agent: TP = answer correct (judge score ≥ 4), FP = wrong answer claimed correct, FN = correct answer missed
  - Tools: TP = correct output, FP = wrong output
  - Macro F1 = average F1 across categories (factual, troubleshooting, reasoning, calculation)

Run:  python evaluate_rag.py
"""

import os
import json
import time
import csv
import importlib.util

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


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

class ClassificationMetrics:
    """Compute accuracy, precision, recall, F1 per category + macro averages.

    Categories are tracked independently. For each category:
      TP = true positives (correct prediction for positive case)
      FP = false positives (wrong prediction claimed as positive)
      FN = false negatives (missed positive case)
      TN = true negatives (correctly identified negative case)

    Per-category:
      Precision_c = TP_c / (TP_c + FP_c)
      Recall_c    = TP_c / (TP_c + FN_c)
      F1_c        = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)

    Macro averages:
      Precision_macro = mean(Precision_c) across categories
      Recall_macro    = mean(Recall_c) across categories
      F1_macro        = mean(F1_c) across categories

    Accuracy = (Σ all TP + Σ all TN) / (Σ all predictions)
    """

    def __init__(self):
        self.categories = {}  # {category: {"tp":0, "fp":0, "fn":0, "tn":0}}

    def add(self, category: str, prediction: bool, ground_truth: bool):
        """Record a single prediction against ground truth for a category."""
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

    def precision(self, category: str) -> float:
        d = self.categories.get(category, {})
        tp, fp = d.get("tp", 0), d.get("fp", 0)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, category: str) -> float:
        d = self.categories.get(category, {})
        tp, fn = d.get("tp", 0), d.get("fn", 0)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1(self, category: str) -> float:
        p, r = self.precision(category), self.recall(category)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def support(self, category: str) -> int:
        """Number of positive ground truth cases in this category."""
        d = self.categories.get(category, {})
        return d.get("tp", 0) + d.get("fn", 0)

    def macro_precision(self) -> float:
        vals = [self.precision(c) for c in self.categories if self.support(c) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def macro_recall(self) -> float:
        vals = [self.recall(c) for c in self.categories if self.support(c) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def macro_f1(self) -> float:
        vals = [self.f1(c) for c in self.categories if self.support(c) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def accuracy(self) -> float:
        total_correct = sum(d.get("tp", 0) + d.get("tn", 0) for d in self.categories.values())
        total_all = sum(sum(d.values()) for d in self.categories.values())
        return total_correct / total_all if total_all > 0 else 0.0

    def weighted_f1(self) -> float:
        """F1 weighted by support (number of positive cases per category)."""
        total_support = sum(self.support(c) for c in self.categories)
        if total_support == 0:
            return 0.0
        return sum(self.f1(c) * self.support(c) for c in self.categories) / total_support

    def report(self) -> str:
        """Generate a formatted classification report string."""
        lines = []
        lines.append(f"{'Category':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("-" * 52)

        for c in sorted(self.categories.keys()):
            s = self.support(c)
            if s > 0:
                lines.append(
                    f"{c:<20} {self.precision(c):>10.3f} {self.recall(c):>10.3f} "
                    f"{self.f1(c):>10.3f} {s:>10}"
                )

        lines.append("-" * 52)
        total_support = sum(self.support(c) for c in self.categories)
        lines.append(
            f"{'Macro Avg':<20} {self.macro_precision():>10.3f} {self.macro_recall():>10.3f} "
            f"{self.macro_f1():>10.3f} {total_support:>10}"
        )
        lines.append(
            f"{'Weighted Avg':<20} {self.macro_precision():>10.3f} {self.macro_recall():>10.3f} "
            f"{self.weighted_f1():>10.3f} {total_support:>10}"
        )
        lines.append(f"{'Accuracy':<20} {self.accuracy():>41.3f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# TEST SETS
# ═══════════════════════════════════════════════════════════════════════════

RETRIEVAL_TEST_SET = [
    {
        "id": "R01", "query": "What is the recommended emitter spacing for sandy soil?",
        "expected_source": "fao_drip_irrigation_design.txt", "expected_page": 3,
        "expected_keywords": ["0.3", "0.5", "sandy", "spacing"],
        "category": "factual",
    },
    {
        "id": "R02", "query": "What is the Kc value for tomato at mid-season?",
        "expected_source": "crop_water_requirements.txt", "expected_page": 2,
        "expected_keywords": ["1.15", "tomato", "mid", "Kc"],
        "category": "factual",
    },
    {
        "id": "R03", "query": "How to troubleshoot a Hunter PGV valve that won't open?",
        "expected_source": "irrigation_troubleshooting.txt", "expected_page": 3,
        "expected_keywords": ["solenoid", "diaphragm", "24 VAC", "Hunter"],
        "category": "troubleshooting",
    },
    {
        "id": "R04", "query": "What is the field capacity of loam soil?",
        "expected_source": "soil_properties_guide.txt", "expected_page": 2,
        "expected_keywords": ["0.25", "0.32", "loam", "field capacity"],
        "category": "factual",
    },
    {
        "id": "R05", "query": "Penman-Monteith ET0 equation formula",
        "expected_source": "eto_calculation_methods.txt", "expected_page": 1,
        "expected_keywords": ["Penman", "Monteith", "ET0", "equation"],
        "category": "factual",
    },
    {
        "id": "R06", "query": "Drip irrigation filtration requirements mesh size",
        "expected_source": "fao_drip_irrigation_design.txt", "expected_page": 5,
        "expected_keywords": ["120", "200", "mesh", "filtration"],
        "category": "factual",
    },
    {
        "id": "R07", "query": "Why does soil moisture sensor read 100% but plants still wilt?",
        "expected_source": "soil_properties_guide.txt", "expected_page": 5,
        "expected_keywords": ["sensor", "emitter", "root", "salinity"],
        "category": "troubleshooting",
    },
    {
        "id": "R08", "query": "Hargreaves ET0 calculation method",
        "expected_source": "eto_calculation_methods.txt", "expected_page": 2,
        "expected_keywords": ["Hargreaves", "temperature", "ET0"],
        "category": "factual",
    },
    {
        "id": "R09", "query": "Distribution uniformity DU formula catch-can test",
        "expected_source": "irrigation_troubleshooting.txt", "expected_page": 4,
        "expected_keywords": ["DU", "catch-can", "lowest 25%"],
        "category": "factual",
    },
    {
        "id": "R10", "query": "Total available water TAW formula soil water balance",
        "expected_source": "crop_water_requirements.txt", "expected_page": 5,
        "expected_keywords": ["TAW", "field capacity", "wilting point"],
        "category": "reasoning",
    },
    {
        "id": "R11", "query": "Emitter types point source inline micro-sprayer",
        "expected_source": "fao_drip_irrigation_design.txt", "expected_page": 2,
        "expected_keywords": ["point-source", "inline", "micro-sprayer"],
        "category": "factual",
    },
    {
        "id": "R12", "query": "Kc wheat initial stage value",
        "expected_source": "crop_water_requirements.txt", "expected_page": 2,
        "expected_keywords": ["wheat", "0.30", "initial"],
        "category": "factual",
    },
    {
        "id": "R13", "query": "Clay soil infiltration rate mm per hour",
        "expected_source": "soil_properties_guide.txt", "expected_page": 3,
        "expected_keywords": ["clay", "0.5", "infiltration"],
        "category": "factual",
    },
    {
        "id": "R14", "query": "ET0 for Cairo Egypt latitude 30 degrees",
        "expected_source": "eto_calculation_methods.txt", "expected_page": 4,
        "expected_keywords": ["Cairo", "30", "7.2"],
        "category": "factual",
    },
    {
        "id": "R15", "query": "Acid injection pH for drip system cleaning",
        "expected_source": "irrigation_troubleshooting.txt", "expected_page": 1,
        "expected_keywords": ["acid", "pH", "2-3"],
        "category": "factual",
    },
]

AGENT_TEST_SET = [
    {
        "id": "A01",
        "query": "What emitter spacing should I use for sandy soil?",
        "expected_tool": "search_knowledge_base",
        "expected_topics": ["sandy", "spacing", "0.3", "0.5"],
        "should_cite": True,
        "category": "factual",
    },
    {
        "id": "A02",
        "query": "Calculate total water volume for 500 emitters at 2 L/h operating 3 hours.",
        "expected_tool": "calculate_drip_irrigation",
        "expected_topics": ["1000", "3000", "500", "L/h", "3 hours"],
        "should_cite": False,
        "category": "calculation",
    },
    {
        "id": "A03",
        "query": "What is the Kc for maize at mid-season?",
        "expected_tool": "lookup_crop_coefficient",
        "expected_topics": ["maize", "1.20", "mid"],
        "should_cite": False,
        "category": "factual",
    },
    {
        "id": "A04",
        "query": "Soil moisture reads 100% but plants wilting. What could be wrong?",
        "expected_tool": "search_knowledge_base",
        "expected_topics": ["sensor", "root rot", "salinity", "placement"],
        "should_cite": True,
        "category": "troubleshooting",
    },
    {
        "id": "A05",
        "query": "ET0 for latitude 30 longitude 31?",
        "expected_tool": "get_reference_evapotranspiration",
        "expected_topics": ["ET0", "latitude", "mm"],
        "should_cite": False,
        "category": "calculation",
    },
    {
        "id": "A06",
        "query": "Troubleshoot Hunter PGV valve won't open.",
        "expected_tool": "search_knowledge_base",
        "expected_topics": ["solenoid", "diaphragm", "24 VAC"],
        "should_cite": True,
        "category": "troubleshooting",
    },
    {
        "id": "A07",
        "query": "What are soil water properties for loam?",
        "expected_tool": "search_knowledge_base",
        "expected_topics": ["loam", "0.25", "0.32", "field capacity"],
        "should_cite": True,
        "category": "factual",
    },
    {
        "id": "A08",
        "query": "Calculate drip: 200 emitters, 4 L/h each, 2 hours.",
        "expected_tool": "calculate_drip_irrigation",
        "expected_topics": ["800", "1600", "200", "4 L/h"],
        "should_cite": False,
        "category": "calculation",
    },
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
     "expected_output_contains": ["0.35", "cotton", "initial"], "category": "factual"},
]


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    def __init__(self):
        self.results = []
        self.groq_api_key = m07.GROQ_API_KEY
        self.groq_model = m07.GROQ_MODEL

        # Classification metrics trackers (one per evaluation type)
        self.retrieval_metrics = ClassificationMetrics()
        self.agent_metrics = ClassificationMetrics()
        self.tool_metrics = ClassificationMetrics()

        # Per-method retrieval classification
        self.retrieval_method_metrics = {}

    # ── HELPER: check if a doc is relevant ────────────────────────────

    def _is_doc_relevant(self, doc, test) -> bool:
        """Check if a retrieved document is relevant to the test query.

        Relevance criteria (any of these):
          1. Source filename matches expected source
          2. Page number matches expected page
          3. Content contains expected keywords
        """
        src = doc.metadata.get("source", "")
        pg = str(doc.metadata.get("page", ""))
        content = doc.page_content.lower()

        # Source match (partial, since names may differ slightly)
        if test["expected_source"] in src or src.replace(".txt", "") in test["expected_source"]:
            return True
        # Page match (if source also matches partially)
        if pg == str(test["expected_page"]) and test["expected_source"].split("_")[0] in src:
            return True
        # Keyword match (at least 2 expected keywords present)
        keyword_matches = sum(1 for kw in test["expected_keywords"] if kw.lower() in content)
        if keyword_matches >= 2:
            return True
        return False

    # ── RETRIEVAL EVALUATION ──────────────────────────────────────────

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

            # Per-method classification metrics
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

                # ── Document-level TP/FP/FN ──
                relevant_in_results = 0
                irrelevant_in_results = 0
                found_expected_source = False
                found_expected_page = False

                for rank, doc in enumerate(docs):
                    is_rel = self._is_doc_relevant(doc, test)
                    if is_rel:
                        relevant_in_results += 1
                    else:
                        irrelevant_in_results += 1

                    # Check for exact source+page match
                    src = doc.metadata.get("source", "")
                    pg = str(doc.metadata.get("page", ""))
                    if test["expected_source"] in src:
                        found_expected_source = True
                        if pg == str(test["expected_page"]):
                            found_expected_page = True

                # Binary classification for this query:
                # Ground truth: True (there IS a relevant doc for this query)
                # Prediction: True if at least one relevant doc was retrieved
                prediction = relevant_in_results > 0
                ground_truth = True  # every query has relevant docs in the KB

                method_clf.add(test["category"], prediction, ground_truth)

                # Also track source-level accuracy
                method_clf.add(test["category"] + "_source", found_expected_source, True)
                method_clf.add(test["category"] + "_page", found_expected_page, True)

                if found_expected_source:
                    hit_count += 1
                if found_expected_page:
                    page_match_count += 1

                # MRR
                first_relevant_rank = None
                for rank, doc in enumerate(docs):
                    if self._is_doc_relevant(doc, test):
                        first_relevant_rank = rank + 1
                        break
                mrr_scores.append(1.0 / first_relevant_rank if first_relevant_rank else 0)

                # Log result
                self.results.append({
                    "test_id": test["id"], "category": test["category"],
                    "evaluation_type": "retrieval", "method": method_name,
                    "query": test["query"],
                    "expected_source": test["expected_source"],
                    "expected_page": test["expected_page"],
                    "found_source": found_expected_source,
                    "found_page": found_expected_page,
                    "relevant_docs_in_top5": relevant_in_results,
                    "irrelevant_docs_in_top5": irrelevant_in_results,
                    "num_docs_retrieved": len(docs),
                    "latency_ms": round(latency * 1000, 1),
                    "mrr": mrr_scores[-1],
                })

            n = len(RETRIEVAL_TEST_SET)
            print(f"  Hit Rate (source): {hit_count/n*100:.1f}% ({hit_count}/{n})")
            print(f"  Source+Page Match:  {page_match_count/n*100:.1f}% ({page_match_count}/{n})")
            print(f"  Avg MRR:           {sum(mrr_scores)/n:.3f}")
            print(f"  Avg Latency:       {sum(latencies)/n*1000:.1f} ms")
            print()
            print(f"  ── Classification Report (doc-level relevance) ──")
            print(method_clf.report())

    # ── TOOL EVALUATION ──────────────────────────────────────────────

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
            if not tool_fn:
                # search_knowledge_base needs a live KB, skip in standalone tool test
                print(f"  {test['id']}: {test['tool']} → ⏭️ Skipped (requires live KB)")
                continue

            result = tool_fn.invoke(test["input"])
            result_str = str(result)

            # Check expected values
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
                "query": str(test["input"]),
                "pass": all_found,
                "output_snippet": result_str[:200],
            })

        print(f"\n  ── Classification Report ──")
        print(self.tool_metrics.report())

    # ── AGENT EVALUATION ──────────────────────────────────────────────

    def evaluate_agent(self):
        print("\n" + "=" * 70)
        print("🤖 AGENT EVALUATION (LLM-as-Judge)")
        print("=" * 70)

        if not self.groq_api_key:
            print("⚠️  GROQ_API_KEY not set — skipping agent evaluation.")
            return

        try:
            llm = m07.get_llm(temperature=0, streaming=False)
            agent_executor = m07.create_agent_executor(llm=llm)
        except Exception as e:
            print(f"⚠️  Agent creation failed: {e}")
            return

        judge_llm = ChatGroq(api_key=self.groq_api_key, model=self.groq_model, temperature=0)

        judge_prompt = """You are an expert RAG evaluation judge. Score this answer:

QUESTION: {question}
EXPECTED_TOPICS: {expected_topics}
ANSWER: {answer}
CONTEXT_USED: {context}

Score 1-5 for each:
1. Faithfulness: answer sticks to context without hallucinating? (5=fully faithful, 1=hallucinated)
2. Relevance: answer directly addresses the question? (5=fully relevant, 1=off-topic)
3. Correctness: answer contains expected factual info? (5=all expected info present, 1=nothing expected)
4. Citation Quality: proper [n] source markers? (5=all cited, 1=no citations)

JSON format ONLY:
{{"faithfulness": X, "relevance": X, "correctness": X, "citation_quality": X, "explanation": "brief"}}
"""

        scores = {"faithfulness": [], "relevance": [], "correctness": [], "citation_quality": []}
        latencies = []

        for test in AGENT_TEST_SET:
            print(f"\n  {test['id']}: {test['query'][:60]}...")

            start = time.time()
            try:
                response = agent_executor.invoke(
                    {"input": test["query"], "chat_history": []},
                    config={"return_intermediate_steps": True},
                )
                latency = time.time() - start
                latencies.append(latency)

                answer = response["output"]
                context = ""
                for step in response.get("intermediate_steps", []):
                    if step[0].tool == "search_knowledge_base":
                        context = str(step[1])[:500]

                has_citations = "[" in answer and "source:" in answer

                # ── LLM-as-Judge ──
                judge_response = judge_llm.invoke([HumanMessage(content=judge_prompt.format(
                    question=test["query"],
                    expected_topics=str(test["expected_topics"]),
                    answer=answer[:1500],
                    context=context[:500],
                ))])

                judge_text = judge_response.content
                try:
                    json_start = judge_text.find("{")
                    json_end = judge_text.find("}") + 1
                    if json_start >= 0:
                        judge_json = json.loads(judge_text[json_start:json_end])
                        for metric in ["faithfulness", "relevance", "correctness", "citation_quality"]:
                            if metric in judge_json:
                                scores[metric].append(judge_json[metric])
                        explanation = judge_json.get("explanation", "")
                    else:
                        for metric in scores: scores[metric].append(3)
                        explanation = "Parse failed"
                except Exception:
                    for metric in scores: scores[metric].append(3)
                    explanation = "Parse failed"

                # ── Classification: answer correct if judge correctness ≥ 4 ──
                is_correct = scores["correctness"][-1] >= 4
                self.agent_metrics.add(test["category"], is_correct, True)

                # ── Classification: citations present ──
                self.agent_metrics.add(test["category"] + "_citations", has_citations, test["should_cite"])

                # ── Classification: used expected tool ──
                used_tools = [step[0].tool for step in response.get("intermediate_steps", [])]
                used_expected_tool = test["expected_tool"] in used_tools
                self.agent_metrics.add(test["category"] + "_tool", used_expected_tool, True)

                print(f"    Latency: {latency:.1f}s | Cited: {has_citations} | Tool: {used_tools}")
                print(f"    Judge: F={scores['faithfulness'][-1]} R={scores['relevance'][-1]} "
                      f"C={scores['correctness'][-1]} Cit={scores['citation_quality'][-1]}")
                print(f"    Correct (C≥4): {is_correct} | {explanation[:80]}")

                self.results.append({
                    "test_id": test["id"], "category": test["category"],
                    "evaluation_type": "agent",
                    "query": test["query"],
                    "expected_tool": test["expected_tool"],
                    "used_tools": str(used_tools),
                    "used_expected_tool": used_expected_tool,
                    "has_citations": has_citations,
                    "is_correct": is_correct,
                    "latency_s": round(latency, 2),
                    "faithfulness": scores["faithfulness"][-1],
                    "relevance": scores["relevance"][-1],
                    "correctness": scores["correctness"][-1],
                    "citation_quality": scores["citation_quality"][-1],
                    "judge_explanation": explanation[:100],
                    "answer_snippet": answer[:200],
                })

            except Exception as e:
                print(f"    ❌ Error: {e}")
                latencies.append(0)
                for metric in scores: scores[metric].append(0)
                self.agent_metrics.add(test["category"], False, True)

        # Summary
        n = len(AGENT_TEST_SET)
        if n > 0 and any(scores["faithfulness"]):
            print(f"\n  ── Agent Judge Scores ──")
            for metric in ["faithfulness", "relevance", "correctness", "citation_quality"]:
                avg = sum(scores[metric]) / n
                print(f"  Avg {metric}: {avg:.2f}/5.0")
            print(f"  Avg Latency: {sum(latencies)/n:.2f}s")

            print(f"\n  ── Classification Report ──")
            print(self.agent_metrics.report())

    # ── EXPORT ────────────────────────────────────────────────────────

    def export_csv(self):
        filepath = "evaluation_results.csv"
        if not self.results:
            return
        fieldnames = list(self.results[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\n💾 Exported: {filepath} ({len(self.results)} rows)")

    # ── FINAL REPORT ──────────────────────────────────────────────────

    def generate_report(self):
        print("\n" + "=" * 70)
        print("📋 COMPREHENSIVE EVALUATION REPORT")
        print("=" * 70)

        # ── Retrieval comparison ──
        print("\n📊 RETRIEVAL METHOD COMPARISON")
        print("-" * 70)
        for method_name, clf in self.retrieval_method_metrics.items():
            print(f"\n  {method_name}:")
            print(clf.report())

        # ── Best retrieval method ──
        best_method = max(
            self.retrieval_method_metrics.items(),
            key=lambda x: x[1].macro_f1()
        )
        print(f"\n  🏆 Best Retrieval Method: {best_method[0]} (F1 macro = {best_method[1].macro_f1():.3f})")

        # ── Tool accuracy ──
        print(f"\n🔧 TOOL ACCURACY")
        print("-" * 70)
        print(self.tool_metrics.report())

        # ── Agent quality ──
        print(f"\n🤖 AGENT QUALITY")
        print("-" * 70)
        print(self.agent_metrics.report())

        # ── Overall summary ──
        print(f"\n{'=' * 70}")
        print("🎯 OVERALL SUMMARY")
        print("=" * 70)

        best_retrieval_f1 = best_method[1].macro_f1()
        tool_f1 = self.tool_metrics.macro_f1()
        agent_f1 = self.agent_metrics.macro_f1()

        print(f"  Retrieval F1 Macro:  {best_retrieval_f1:.3f}")
        print(f"  Tool F1 Macro:       {tool_f1:.3f}")
        print(f"  Agent F1 Macro:      {agent_f1:.3f}")
        print(f"  System F1 Macro:     {(best_retrieval_f1 + tool_f1 + agent_f1) / 3:.3f}")
        print(f"  Retrieval Accuracy:  {best_method[1].accuracy():.3f}")
        print(f"  Tool Accuracy:       {self.tool_metrics.accuracy():.3f}")
        print(f"  Agent Accuracy:      {self.agent_metrics.accuracy():.3f}")
        print(f"  System Accuracy:     {(best_method[1].accuracy() + self.tool_metrics.accuracy() + self.agent_metrics.accuracy()) / 3:.3f}")

        print(f"\n{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 AQUAMIND RAG SYSTEM EVALUATION")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Key: {'✅ Set' if m07.GROQ_API_KEY else '❌ Not set'}")
    print(f"Model: {m07.GROQ_MODEL}")

    evaluator = RAGEvaluator()
    evaluator.evaluate_retrieval()
    evaluator.evaluate_tools()
    evaluator.evaluate_agent()
    evaluator.export_csv()
    evaluator.generate_report()

    print("\n✅ Evaluation complete!")