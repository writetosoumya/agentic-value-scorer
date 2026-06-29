# Agentification Value Scorer
### ARM™ × AVRE™ LangGraph Multi-Agent Governance Intelligence

**Week 3 Project | The Gen Academy Mastering Agentic AI Bootcamp**
**Author:** Soumya V. Jom | Enterprise AI Strategy & Governance, Cognizant PS&E

---

## What This Is

The **Agentification Value Scorer (AVS)** is a RAG-powered, LangGraph multi-agent governance intelligence platform that scores enterprise AI agent use cases for agentification readiness using two proprietary frameworks:

**ARM™ — Agentification Risk Model** (published May 2026)
Five risk domains scored 0–10, weighted into a composite ARM™ score → risk tier:
- **Execution Integrity** (25%) — Can the work still be executed correctly without the agent?
- **Decision Integrity** (25%) — Are humans deciding, or merely approving? (Authority Shaping, Concurrence Validation)
- **Capability Debt** (20%) — What human capabilities are being consumed? (Cognitive Atrophy, Succession Failure)
- **Knowledge Sustainability** (15%) — Is enterprise knowledge preserved or locked in agent logic?
- **Enterprise Adaptation** (15%) — Can the org still respond when reality changes? (Edge Case Fragility, Escalation Integrity Failure)

Risk tiers: 🟢 GREEN (0–3.0) / 🟡 AMBER (3.1–5.5) / 🔴 RED (5.6–7.5) / ⛔ CRITICAL (7.6–10.0)

**AVRE™ — Agentic Value Realization Engine** (published June 2026)
NEV Formula: `Net Enterprise Value = (ROI + ROE + ROF) − Total Cost of Agentification − ARM™ Risk Penalty`
- **ROI** — Return on Investment: cost saved, revenue lifted, hours recovered (40%)
- **ROE** — Return on Effort: did output quality improve, not just volume? (30%)
- **ROF** — Return on Future: did this build a reusable capability or a dead end? (30%)

NEV scale: 90–100 Transformational / 75–89 High Value / 60–74 Valuable / 40–59 Marginal / 0–39 Poor

---

## The Agent One-Liner

> My agent helps **enterprise AI governance teams and strategy leaders** score and advise on AI agentification use cases in a **Streamlit chat interface**, replacing 4–6 hours of manual ARM™ × AVRE™ framework analysis per use case. It **scores, clarifies, checks compliance, and searches** using 4 specialist agents across a dual governance corpus, **hands off to a human before any governance brief is saved**, and I'll know it works when a governance analyst can get a defensible scored brief in under 10 minutes 8 times out of 10.

---

## Week 3 Architecture — LangGraph Multi-Agent System

```
User message
      │
      ▼
Fast Keyword Intent Router (zero LLM calls — saves 2–4s/turn)
      │
      ├─► SCORE      → ARM™ × AVRE™ dual-corpus scoring
      ├─► CLARIFY    → Targeted follow-up (vagueness gate)
      ├─► COMPLIANCE → Regulatory gap analysis (Layer 2)
      ├─► SEARCH     → RAG knowledge retrieval (both layers)
      └─► BRIEF      → [INTERRUPT] → Human Approve/Reject → Save .md

SQLite checkpointer persists full conversation state between turns.
Error node catches failures → re-routes to CLARIFY (never crashes).
```

### Dual Corpus Architecture

| Layer | Contents |
|-------|----------|
| **Layer 1 — Global** | ARM™ framework · AVRE™ framework · EU AI Act 2024/1689 · NIST AI RMF 1.0 · Use case reference library |
| **Layer 2 — Client** | Meridian Insurance: Company AI Governance Policy · Insurance Sector Regulation (NAIC, Colorado SB 169, NYDFS, FCRA, GLBA) · AI Program Economics Reference |

### Resilience Architecture

- **Retry with exponential backoff** — all LLM calls retry 3× (1.5s → 3s → 6s)
- **Error node** — score failures route to structured recovery, never a crash
- **Vagueness gate** — inputs under 8 words auto-escalate to CLARIFY before scoring
- **HITL gate** — LangGraph `interrupt_before=['brief_node']` prevents autonomous brief saves
- **Scoring confidence signal** — every response shows which corpus layers contributed

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph (StateGraph, interrupt, SqliteSaver) |
| RAG framework | LangChain |
| Vectorstore | Custom TF-IDF + MMR (scikit-learn) — no external vector DB |
| LLM inference | Llama 3.3 70B via Nebius Token Factory |
| Session memory | SQLite via LangGraph checkpointer |
| Frontend | Streamlit (mode toggle, chat UI, HITL gate, eval report) |
| Scoring frameworks | ARM™ and AVRE™ (proprietary IP, Soumya V. Jom) |
| Language | Python 3.13 |

---

## Project Structure

```
agentic-value-scorer/
├── app.py                        # Streamlit UI — Score Mode + Agent Mode + Eval + About
├── main.py                       # Week 2 CLI entry point (preserved)
├── requirements.txt
├── README.md
│
├── src/
│   ├── corpus.py                 # Dual TF-IDF vectorstore (Layer 1 + Layer 2)
│   ├── graph.py                  # LangGraph state machine + keyword intent router
│   ├── agents.py                 # Five specialist agents + retry + vagueness gate
│   ├── hitl.py                   # HITL gate + brief persistence
│   ├── memory.py                 # SQLite session memory
│   ├── scoring_engine.py         # ARM™ + AVRE™ dataclasses (published domain names)
│   ├── llm_scorer.py             # Score Mode LLM scorer (Week 2, updated field names)
│   ├── rag_engine.py             # Week 2 RAG pipeline (preserved)
│   └── evaluator.py              # 20-question dual-corpus evaluation suite
│
├── corpus/
│   ├── layer1/                   # Global governance frameworks
│   │   ├── arm_framework.txt
│   │   ├── avre_framework.txt
│   │   ├── governance_frameworks.txt
│   │   └── use_case_reference.txt
│   └── layer2/                   # Client-specific (Meridian Insurance)
│       ├── company_ai_policy.txt
│       ├── insurance_sector_regulation.txt
│       └── cost_parameters.txt
│
├── outputs/                      # Eval reports, score JSONs (gitignored: sessions.db)
└── briefs/                       # HITL-approved governance briefs (gitignored)
```

---

## Setup

```bash
git clone <repo>
cd agentic-value-scorer
pip3 install -r requirements.txt
```

Set your Nebius API key (required every terminal session):
```bash
export NEBIUS_API_KEY=your_key_here
```

To persist across sessions:
```bash
echo 'export NEBIUS_API_KEY=your_key_here' >> ~/.zshrc
source ~/.zshrc
```

Build the dual vectorstore:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from corpus import build_dual_corpus
dc = build_dual_corpus(force_rebuild=True)
print('L1:', len(dc.l1.chunks), '| L2:', len(dc.l2.chunks))
"
```

Launch:
```bash
streamlit run app.py
```

---

## Usage — Agent Mode

| Intent | Trigger | Example |
|--------|---------|---------|
| **SCORE** | Describe a use case | *"Score an AI agent that auto-approves claims under $2,500"* |
| **CLARIFY** | Vague input | *"I want to build an AI agent for ops"* |
| **COMPLIANCE** | Regulation question | *"What does Colorado SB 169 require for our underwriting AI?"* |
| **SEARCH** | Framework question | *"What is ARM™ Decision Integrity risk?"* |
| **BRIEF** | Generate report | *"Generate a governance brief"* → human approves → saves as .md |

Quick action buttons: 📋 Generate Brief · 🔍 Compliance Check · 🔄 Rescore · 🗑 Clear Chat

---

## Evaluation Results

20-question evaluation suite covering ARM™ published domains, AVRE™ published lenses, and Layer 2 Meridian corpus.

| Metric | Result |
|--------|--------|
| Overall faithfulness | **91.6%** |
| Layer 1 faithfulness | **91.0%** (ARM™, AVRE™, EU AI Act, NIST AI RMF) |
| Layer 2 faithfulness | **90.0%** (Meridian policy, regulation, cost parameters) |
| Source retrieval accuracy | **100%** |
| Questions passing ≥50% | **19/19** |
| Retrieval | MMR k=4, dual corpus |
| Vectorstore | Custom TF-IDF (no external model or API) |

---

## Frameworks (Proprietary IP)

ARM™ and AVRE™ are original frameworks developed independently by Soumya V. Jom. Published in *The Enterprise AI Lens* newsletter. Not Cognizant IP. Use under attribution.

- ARM™: [The Enterprise AI Lens, Issue 2](https://www.linkedin.com/newsletters/the-enterprise-ai-lens-7439093999786795008/)
- AVRE™: [The Enterprise AI Lens, Issue 4](https://www.linkedin.com/newsletters/the-enterprise-ai-lens-7439093999786795008/)

---

## Week 2 → Week 3 Evolution

| Week 2 | Week 3 |
|--------|--------|
| Single-shot RAG scorer | LangGraph multi-agent stateful system |
| Single TF-IDF corpus | Dual corpus (global + client-specific) |
| CLI interface | Streamlit chat + mode toggle |
| No memory | SQLite session memory |
| No routing | Fast keyword intent router (5 modes) |
| No error recovery | Retry + error node + vagueness gate |
| No HITL | Human-in-the-loop gate before brief save |
| 15-question eval | 20-question dual-corpus eval |
| 81.2% faithfulness | 91.6% faithfulness |
