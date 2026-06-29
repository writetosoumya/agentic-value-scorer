"""
evaluator.py
Updated 20-question evaluation suite for AVS Week 3.

Changes from Week 2:
  - Retrieval k reduced from 5 → 4 (matches agent config)
  - ARM™ domain names updated to published framework:
    execution_integrity, decision_integrity, capability_debt,
    knowledge_sustainability, enterprise_adaptation
  - AVRE™ lens names updated: ROI, ROE (Return on Effort), ROF (Return on Future)
  - 5 new Layer 2 questions testing Meridian-specific corpus:
    company_ai_policy, insurance_sector_regulation, cost_parameters
  - NEV score interpretation updated to published AVRE™ scale
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from rich.console import Console
from rich.table import Table
from rich import box

from corpus import build_dual_corpus, DualCorpus

console = Console()

# ── Dual corpus retriever wrapper ─────────────────────────────────────────────
# Wraps DualCorpus.retrieve() to match the retrieve_context(query, retriever)
# interface used by the rest of the eval suite.

def _get_dual_retriever():
    """Load or build the dual corpus and return a retriever-like object."""
    dc = build_dual_corpus()

    class DualRetriever:
        def __init__(self, dual: DualCorpus):
            self.dual = dual
        def invoke(self, query: str):
            result = self.dual.retrieve(query, layers="both", k=4)
            # Return list of doc-like objects with page_content and metadata
            class Doc:
                def __init__(self, content, source):
                    self.page_content = content
                    self.metadata = {"source_file": source}
            docs = []
            for chunk in result["context"].split("\n\n---\n\n"):
                docs.append(Doc(chunk, "dual_corpus"))
            return docs

    return DualRetriever(dc)


def retrieve_context(query: str, retriever) -> dict:
    """Retrieve context using dual corpus — both L1 and L2."""
    dc: DualCorpus = retriever.dual
    result = dc.retrieve(query, layers="both", k=4)
    return {
        "context":    result["context"],
        "sources":    result["sources"],
        "num_chunks": result["num_chunks"],
    }

# ── Evaluation questions ──────────────────────────────────────────────────────

EVAL_QUESTIONS = [

    # ══ LAYER 1 — ARM™ Framework (published domains) ══════════════════════════

    {
        "id": "Q01",
        "category": "ARM™ — Decision Integrity",
        "query": "What is Decision Integrity risk in ARM™ and what is Concurrence Validation?",
        "expected_concepts": ["decision integrity", "concurrence validation", "authority shaping", "rubber", "ceremonial"],
        "expected_sources": ["arm_framework"],
        "difficulty": "easy",
        "layer": "L1",
    },
    {
        "id": "Q02",
        "category": "ARM™ — Capability Debt",
        "query": "Explain Capability Debt and Cognitive Atrophy risk in the ARM™ framework",
        "expected_concepts": ["capability debt", "cognitive atrophy", "succession failure", "skill", "expertise"],
        "expected_sources": ["arm_framework"],
        "difficulty": "easy",
        "layer": "L1",
    },
    {
        "id": "Q03",
        "category": "ARM™ — Enterprise Adaptation",
        "query": "What is Edge Case Fragility and Escalation Integrity Failure in ARM™?",
        "expected_concepts": ["edge case", "escalation integrity", "novel", "fragility", "adapt"],
        "expected_sources": ["arm_framework"],
        "difficulty": "easy",
        "layer": "L1",
    },
    {
        "id": "Q04",
        "category": "ARM™ — Execution Integrity",
        "query": "How does ARM™ score Execution Integrity risk for an agent that processes claims autonomously?",
        "expected_concepts": ["execution integrity", "autonomously", "error", "override", "quality"],
        "expected_sources": ["arm_framework"],
        "difficulty": "medium",
        "layer": "L1",
    },
    {
        "id": "Q05",
        "category": "ARM™ — Knowledge Sustainability",
        "query": "What does Knowledge Sustainability risk measure in ARM™ and why does tacit knowledge matter?",
        "expected_concepts": ["knowledge sustainability", "tacit", "institutional", "preserved", "locked"],
        "expected_sources": ["arm_framework"],
        "difficulty": "easy",
        "layer": "L1",
    },

    # ══ LAYER 1 — AVRE™ Framework (published lenses) ═════════════════════════

    {
        "id": "Q06",
        "category": "AVRE™ — NEV Formula",
        "query": "What is the AVRE™ Net Enterprise Value formula and what does ARM™ Risk Penalty mean?",
        "expected_concepts": ["net enterprise value", "roi", "roe", "rof", "risk penalty", "agentification"],
        "expected_sources": ["avre_framework"],
        "difficulty": "easy",
        "layer": "L1",
    },
    {
        "id": "Q07",
        "category": "AVRE™ — ROE Lens",
        "query": "What is Return on Effort in AVRE™ and how is it different from ROI?",
        "expected_concepts": ["return on effort", "quality", "rework", "volume", "operational"],
        "expected_sources": ["avre_framework"],
        "difficulty": "easy",
        "layer": "L1",
    },
    {
        "id": "Q08",
        "category": "AVRE™ — ROF Lens",
        "query": "What is Return on Future in AVRE™ and what makes a deployment score high on ROF?",
        "expected_concepts": ["return on future", "reusable", "platform", "dead end", "capability", "strategic"],
        "expected_sources": ["avre_framework"],
        "difficulty": "easy",
        "layer": "L1",
    },
    {
        "id": "Q09",
        "category": "AVRE™ — NEV Scoring",
        "query": "What does a NEV score of 28 mean in AVRE™ and when should you retire an AI deployment?",
        "expected_concepts": ["28", "poor", "retire", "0-39", "risk penalty", "liability"],
        "expected_sources": ["avre_framework"],
        "difficulty": "medium",
        "layer": "L1",
    },
    {
        "id": "Q10",
        "category": "Cross-framework",
        "query": "An HR resume screening agent with high Decision Integrity risk — what does that mean for its AVRE™ NEV score?",
        "expected_concepts": ["decision integrity", "net enterprise value", "risk penalty", "multiplier", "capability debt"],
        "expected_sources": ["arm_framework", "avre_framework"],
        "difficulty": "hard",
        "layer": "L1",
    },

    # ══ LAYER 2 — Meridian Company AI Policy ══════════════════════════════════

    {
        "id": "Q11",
        "category": "L2 — Meridian AI Policy",
        "query": "What are Meridian's autonomy thresholds for AI agents handling claims and payments?",
        "expected_concepts": ["2,500", "1,000", "human review", "authorization", "autonomy"],
        "expected_sources": ["company_ai_policy"],
        "difficulty": "easy",
        "layer": "L2",
    },
    {
        "id": "Q12",
        "category": "L2 — Meridian AI Policy",
        "query": "What are the HITL requirements for AI agents at Meridian Insurance?",
        "expected_concepts": ["human-in-the-loop", "write action", "financial transaction", "irreversible", "customer record"],
        "expected_sources": ["company_ai_policy"],
        "difficulty": "easy",
        "layer": "L2",
    },
    {
        "id": "Q13",
        "category": "L2 — Meridian AI Policy",
        "query": "How does Meridian classify AI risk tiers and what does Tier 1 Critical mean?",
        "expected_concepts": ["tier 1", "critical", "board", "audit", "explainability", "72-hour"],
        "expected_sources": ["company_ai_policy"],
        "difficulty": "medium",
        "layer": "L2",
    },
    {
        "id": "Q14",
        "category": "L2 — Insurance Regulation",
        "query": "What does Colorado SB 169 require for AI used in insurance underwriting?",
        "expected_concepts": ["colorado", "discrimination", "governance program", "testing", "unfair"],
        "expected_sources": ["insurance_sector_regulation"],
        "difficulty": "easy",
        "layer": "L2",
    },
    {
        "id": "Q15",
        "category": "L2 — Insurance Regulation",
        "query": "What NAIC model bulletin requirements apply to AI claims handling at Meridian?",
        "expected_concepts": ["naic", "claims", "human review", "unfair", "dispute", "audit trail"],
        "expected_sources": ["insurance_sector_regulation"],
        "difficulty": "medium",
        "layer": "L2",
    },
    {
        "id": "Q16",
        "category": "L2 — Cost Parameters",
        "query": "What is the implementation cost range for a Tier 2 moderate AI deployment at Meridian?",
        "expected_concepts": ["350,000", "950,000", "tier 2", "moderate", "rag", "16"],
        "expected_sources": ["cost_parameters"],
        "difficulty": "easy",
        "layer": "L2",
    },
    {
        "id": "Q17",
        "category": "L2 — Cost Parameters",
        "query": "What minimum ROI thresholds does Meridian require for AI investment approval by Year 3?",
        "expected_concepts": ["35%", "year 3", "cumulative", "payback", "hurdle rate"],
        "expected_sources": ["cost_parameters"],
        "difficulty": "medium",
        "layer": "L2",
    },

    # ══ CROSS-LAYER — L1 + L2 synthesis ══════════════════════════════════════

    {
        "id": "Q18",
        "category": "Cross-layer synthesis",
        "query": "A claims fraud detection agent with autonomous account holds — what ARM™ risks and Meridian policy constraints apply?",
        "expected_concepts": ["execution integrity", "decision integrity", "2,500", "human review", "tier 1"],
        "expected_sources": ["arm_framework", "company_ai_policy"],
        "difficulty": "hard",
        "layer": "Both",
    },
    {
        "id": "Q19",
        "category": "Cross-layer synthesis",
        "query": "How should Meridian's AVRE™ ROI calibration thresholds affect scoring a customer service AI agent?",
        "expected_concepts": ["roi", "payback", "year 3", "35%", "benefit realization"],
        "expected_sources": ["avre_framework", "cost_parameters"],
        "difficulty": "hard",
        "layer": "Both",
    },

    # ══ EDGE / OUT-OF-CORPUS ══════════════════════════════════════════════════

    {
        "id": "Q20",
        "category": "Out of corpus",
        "query": "What is the exact fine amount for violating EU AI Act Article 5 prohibited practices?",
        "expected_concepts": [],
        "expected_sources": ["governance_frameworks"],
        "difficulty": "out-of-corpus",
        "layer": "L1",
        "note": "Specific fine amounts not in corpus — tests graceful handling of knowledge gaps.",
    },
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    question_id:             str
    category:                str
    query:                   str
    difficulty:              str
    layer:                   str
    retrieved_sources:       List[str]
    retrieved_context_preview: str
    concept_hits:            int
    concept_total:           int
    source_hit:              bool
    faithfulness_score:      float
    retrieval_score:         float
    notes:                   str = ""


# ── Evaluator ─────────────────────────────────────────────────────────────────

def evaluate_retrieval(query_dict: dict, retriever) -> EvalResult:
    result = retrieve_context(query_dict["query"], retriever)

    context_lower     = result["context"].lower()
    expected_concepts = query_dict.get("expected_concepts", [])
    expected_sources  = query_dict.get("expected_sources", [])

    hits        = sum(1 for c in expected_concepts if c.lower() in context_lower)
    faithfulness = round(hits / len(expected_concepts), 2) if expected_concepts else 1.0
    source_hit  = any(src in result["sources"] for src in expected_sources) if expected_sources else True

    return EvalResult(
        question_id              = query_dict["id"],
        category                 = query_dict["category"],
        query                    = query_dict["query"],
        difficulty               = query_dict["difficulty"],
        layer                    = query_dict.get("layer", "L1"),
        retrieved_sources        = result["sources"],
        retrieved_context_preview = result["context"][:300],
        concept_hits             = hits,
        concept_total            = len(expected_concepts),
        source_hit               = source_hit,
        faithfulness_score       = faithfulness,
        retrieval_score          = 1.0 if source_hit else 0.0,
        notes                    = query_dict.get("note", ""),
    )


def run_evaluation_suite(retriever=None):
    """
    Run all 20 evaluation questions using the dual corpus retriever.
    The retriever param is kept for backward compatibility but ignored —
    dual corpus is always used for accurate L1 + L2 coverage.
    """
    console.print("\n[bold cyan]═══ AVS RAG EVALUATION SUITE — 20 Questions ═══[/bold cyan]\n")
    console.print("ARM™ published domains · AVRE™ published lenses · Layer 2 Meridian corpus · k=4 retrieval\n")

    # Always use dual corpus retriever
    dual_retriever = _get_dual_retriever()
    results: List[EvalResult] = []

    for q in EVAL_QUESTIONS:
        console.print(f"[dim]Running {q['id']} [{q.get('layer','L1')}]: {q['query'][:65]}...[/dim]")
        r = evaluate_retrieval(q, dual_retriever)
        results.append(r)
        time.sleep(0.1)

    # ── Results table ──────────────────────────────────────────────────────────
    table = Table(
        title="\n📊 AVS Evaluation Results — ARM™ × AVRE™ · Dual Corpus · k=4",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("ID",          style="bold", width=4)
    table.add_column("Layer",       width=6)
    table.add_column("Category",    width=22)
    table.add_column("Difficulty",  width=12)
    table.add_column("Faithfulness",justify="center", width=14)
    table.add_column("Source Hit",  justify="center", width=10)
    table.add_column("Retrieved",   width=28)

    def fc(s):
        if s >= 0.75: return "green"
        elif s >= 0.5: return "yellow"
        return "red"

    for r in results:
        fp   = f"{r.faithfulness_score*100:.0f}% ({r.concept_hits}/{r.concept_total})"
        src  = "✅" if r.source_hit else "❌"
        table.add_row(
            r.question_id,
            r.layer,
            r.category,
            r.difficulty,
            f"[{fc(r.faithfulness_score)}]{fp}[/{fc(r.faithfulness_score)}]",
            src,
            ", ".join(r.retrieved_sources)[:28],
        )

    console.print(table)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    scoreable  = [r for r in results if r.difficulty != "out-of-corpus"]
    l1_results = [r for r in scoreable if r.layer == "L1"]
    l2_results = [r for r in scoreable if r.layer == "L2"]
    both       = [r for r in scoreable if r.layer == "Both"]

    avg_faith  = sum(r.faithfulness_score for r in scoreable) / len(scoreable)
    avg_source = sum(r.retrieval_score    for r in results)   / len(results)
    total_pass = sum(1 for r in scoreable if r.faithfulness_score >= 0.5)

    l1_faith   = sum(r.faithfulness_score for r in l1_results) / len(l1_results) if l1_results else 0
    l2_faith   = sum(r.faithfulness_score for r in l2_results) / len(l2_results) if l2_results else 0

    console.print(f"\n[bold]═══ Aggregate Metrics ═══[/bold]")
    console.print(f"  Total questions:            {len(results)} (20)")
    console.print(f"  Scoreable (excl. OOC):      {len(scoreable)}")
    console.print(f"  Overall avg faithfulness:   [{fc(avg_faith)}]{avg_faith*100:.1f}%[/{fc(avg_faith)}]")
    console.print(f"  Layer 1 faithfulness:       [{fc(l1_faith)}]{l1_faith*100:.1f}%[/{fc(l1_faith)}]")
    console.print(f"  Layer 2 faithfulness:       [{fc(l2_faith)}]{l2_faith*100:.1f}%[/{fc(l2_faith)}]")
    console.print(f"  Source retrieval accuracy:  [{fc(avg_source)}]{avg_source*100:.1f}%[/{fc(avg_source)}]")
    console.print(f"  Questions passing (≥50%):   {total_pass}/{len(scoreable)}")
    console.print(f"  Retrieval k:                4 (reduced from 5)")

    # ── Failure analysis ───────────────────────────────────────────────────────
    failures = [r for r in scoreable if r.faithfulness_score < 0.5]
    if failures:
        console.print(f"\n[bold red]Failures ({len(failures)} below 50%):[/bold red]")
        for f in failures:
            console.print(f"  • {f.question_id} [{f.layer}] ({f.category}): {f.faithfulness_score*100:.0f}%")
            console.print(f"    Retrieved: {f.retrieved_sources}")

    # ── Save JSON report ───────────────────────────────────────────────────────
    output = {
        "evaluation_summary": {
            "total_questions":              len(results),
            "scoreable_questions":          len(scoreable),
            "avg_faithfulness_pct":         round(avg_faith * 100, 1),
            "l1_faithfulness_pct":          round(l1_faith * 100, 1),
            "l2_faithfulness_pct":          round(l2_faith * 100, 1),
            "source_retrieval_accuracy_pct": round(avg_source * 100, 1),
            "questions_passing_50pct":      f"{total_pass}/{len(scoreable)}",
            "retrieval_k":                  4,
            "embed_model":                  "TF-IDF custom (no external model)",
            "retrieval_strategy":           "MMR k=4 per layer, dual corpus",
            "corpus_layers":                "L1: ARM™, AVRE™, EU AI Act, NIST AI RMF | L2: Meridian policy, regulation, cost",
            "arm_domains":                  "execution_integrity, decision_integrity, capability_debt, knowledge_sustainability, enterprise_adaptation",
            "avre_lenses":                  "ROI, ROE (Return on Effort), ROF (Return on Future)",
        },
        "results": [
            {
                "id":                r.question_id,
                "category":          r.category,
                "layer":             r.layer,
                "query":             r.query,
                "difficulty":        r.difficulty,
                "faithfulness_score": r.faithfulness_score,
                "concept_hits":      r.concept_hits,
                "concept_total":     r.concept_total,
                "source_hit":        r.source_hit,
                "retrieved_sources": r.retrieved_sources,
                "context_preview":   r.retrieved_context_preview,
                "notes":             r.notes,
            }
            for r in results
        ],
    }

    out_path = Path(__file__).parent.parent / "outputs" / "eval_report.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[dim]Evaluation report saved → {out_path}[/dim]")
    return output
