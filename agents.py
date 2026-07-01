"""
agents.py
Five specialist agent node functions for the Week 3 AVS LangGraph graph.

Each function receives the AgentState dict and returns a partial state update.
All LLM calls go through the same Nebius/Llama 3.3 70B endpoint as Week 2.

Nodes:
  score_agent      — ARM™ × AVRE™ dual-corpus scoring (extends Week 2 score_use_case)
  clarify_agent    — Asks a single targeted follow-up question
  compliance_agent — Regulatory gap check against Layer 2 (Meridian policy + regulation)
  search_agent     — Free-text RAG search across both layers, returns citations
  brief_agent      — Synthesises conversation into a governance brief for HITL review
"""

import json
import re
import os
import time
import functools
from typing import Any, Dict

from openai import OpenAI

from corpus import DualCorpus
from scoring_engine import ARMScores, AVREScores, ScoringResult

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODEL    = "meta-llama/Llama-3.3-70B-Instruct"

# ── Vagueness threshold ───────────────────────────────────────────────────────
# Inputs shorter than this word count are auto-escalated to CLARIFY
# before scoring. Exposed here so graph.py can import it.
VAGUENESS_THRESHOLD = 8


# ── Retry decorator ───────────────────────────────────────────────────────────

def _retry(max_attempts: int = 3, base_delay: float = 1.5):
    """
    Exponential-backoff retry for Nebius API calls.
    Retries on any Exception (timeout, rate-limit, network blip).
    After max_attempts, re-raises the last exception so the caller
    can set state["error"] and the graph can fallback gracefully.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = base_delay * (2 ** (attempt - 1))   # 1.5s, 3s, 6s
                        print(f"[Retry] Attempt {attempt} failed: {exc}. Retrying in {wait:.1f}s...")
                        time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


# ── Shared Nebius client ──────────────────────────────────────────────────────

@_retry(max_attempts=3, base_delay=1.5)
def _llm(system: str, user: str, max_tokens: int = 1500) -> str:
    """
    Single LLM call with automatic retry (3 attempts, exponential backoff).
    Raises on final failure — callers should catch and set state['error'].
    """
    client = OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("NEBIUS_API_KEY"),
    )
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON in response: {text[:300]}")
    return json.loads(text[start:end])


# ── Vagueness check (used by router in graph.py) ──────────────────────────────

def is_vague(user_input: str) -> bool:
    """
    Returns True if the input is too short or generic to score reliably.
    Triggers auto-escalation to CLARIFY before scoring.

    Heuristics:
      - Fewer than VAGUENESS_THRESHOLD words
      - No domain nouns (agent, system, process, data, customer, claim, etc.)
    """
    words = user_input.strip().split()
    if len(words) < VAGUENESS_THRESHOLD:
        return True
    domain_nouns = {
        "agent", "system", "bot", "process", "workflow", "automation",
        "data", "customer", "claim", "policy", "underwriting", "fraud",
        "hr", "finance", "compliance", "contract", "invoice", "ticket",
        "model", "decision", "review", "approval", "detection", "scoring",
    }
    has_domain = any(w.lower() in domain_nouns for w in words)
    return not has_domain


# ── Helper: build conversation history string ─────────────────────────────────

def _history_str(messages: list, max_turns: int = 6) -> str:
    """Render last N turns as plain text for LLM context."""
    recent = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SCORE AGENT
# Runs dual-corpus ARM™ × AVRE™ scoring.  Layer 2 context is appended to
# enrich scores with Meridian-specific cost/compliance calibration.
# ─────────────────────────────────────────────────────────────────────────────

SCORE_SYSTEM = """You are an expert Enterprise AI Governance analyst. You score enterprise AI agent use cases using two proprietary frameworks authored by Soumya V. Jom.

━━━ ARM™ — AGENTIFICATION RISK MODEL ━━━
ARM™ asks: How do we govern what agentification does to the enterprise?
Core principle: Net Enterprise Value = Benefit Realization − Capability Debt

Score FIVE domains 0–10. Higher = more risk.

1. execution_integrity — Can the work still be executed correctly without the agent?
   Assess: accuracy degradation, error detection loss, override mechanism failure, quality assurance gaps.

2. decision_integrity — Are humans deciding, or merely approving?
   Assess: Authority Shaping (AI recommendations become de facto decisions), Concurrence Validation
   (ceremonial approval), meaningful oversight erosion.

3. capability_debt — What human capabilities are being consumed in pursuit of efficiency?
   Assess: Cognitive Atrophy (skill degradation through substitution), Succession Failure
   (institutional knowledge locked in agent logic), training pipeline collapse.

4. knowledge_sustainability — Is enterprise knowledge being preserved or transferred to systems?
   Assess: tacit knowledge loss, institutional memory transfer failure, explainability gap.

5. enterprise_adaptation — Can the organization still respond when reality changes?
   Assess: Edge Case Fragility, Escalation Integrity Failure, regulatory adaptation speed.

ARM™ Risk Tiers:
0.0–3.0  → GREEN    Proceed with standard governance
3.1–5.5  → AMBER    Enhanced controls required
5.6–7.5  → RED      Phased deployment only. Independent review required.
7.6–10.0 → CRITICAL Do not deploy without executive sign-off and external audit

━━━ AVRE™ — AGENTIC VALUE REALIZATION ENGINE ━━━
AVRE™ formula: NEV = (ROI + ROE + ROF) − Total Cost of Agentification − ARM™ Risk Penalty

Score THREE lenses 0–10. Higher = more value.

ROI — Return on Investment (Financial Lens)
Cost saved. Revenue lifted. Hours recovered. The lens everyone uses. Never enough on its own.
Calibrate against: implementation cost, operational cost, change management cost.
Important: high volume savings with poor quality outcomes do NOT score high on ROI.

ROE — Return on Effort (Operational Lens)
Did output quality improve? Did rework drop? Did people get to do better work or just more of it?
The lens most platforms ignore. A use case that saves hours but increases rework scores LOW.

ROF — Return on Future (Strategic Lens)
Did this build a reusable capability — or a dead end?
Reusable agent frameworks and shared governance infrastructure score HIGH.
Single-use automations with no architectural carry-forward score LOW.

opportunity_cost — urgency of acting (0–10, higher = more urgent to agentify now)

ARM™ Risk Penalty application: A deployment with critical ARM™ flags (composite > 7.5) should
produce a NEV in the 0–39 range regardless of ROI. Risk penalty is multiplicative, not additive.

NEV Interpretation (0–100 scale mapped from scores):
90–100 → Transformational. Scale it.
75–89  → High Value. Grow with confidence.
60–74  → Valuable. Optimize first.
40–59  → Marginal. Redesign before you expand.
0–39   → Poor. Retire it.

━━━ LAYER INSTRUCTIONS ━━━
Layer 1 = global ARM™ and AVRE™ framework principles — use for scoring rubric.
Layer 2 = client-specific policy, regulation, and cost parameters — use to calibrate scores to
the client's actual risk thresholds, ROI benchmarks, and regulatory obligations.

Return ONLY valid JSON — no markdown, no preamble:
{
  "arm": {
    "execution_integrity": float,
    "decision_integrity": float,
    "capability_debt": float,
    "knowledge_sustainability": float,
    "enterprise_adaptation": float
  },
  "avre": {
    "roi_score": float,
    "roe_score": float,
    "rof_score": float,
    "opportunity_cost": float
  },
  "rationale": {
    "execution_integrity": str,
    "decision_integrity": str,
    "capability_debt": str,
    "knowledge_sustainability": str,
    "enterprise_adaptation": str,
    "roi": str,
    "roe": str,
    "rof": str,
    "opportunity_cost": str
  },
  "recommendations": [str, str, str]
}"""


def score_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """ARM™ × AVRE™ dual-corpus scoring agent."""
    corpus: DualCorpus = state["corpus"]
    query = f"ARM risk scoring AVRE value assessment: {state['use_case_name']} {state['user_input']}"

    rag = corpus.retrieve(query, layers="both", k=5)

    user_msg = f"""USE CASE:
Name: {state['use_case_name']}
Description: {state['user_input']}

LAYER 1 — GLOBAL GOVERNANCE CONTEXT:
{rag['context']}

LAYER 2 — CLIENT POLICY & COST PARAMETERS:
(Sources: {', '.join(rag['layer_breakdown']['l2_chunks'])})

Score this use case now. Return only valid JSON."""

    try:
        raw    = _llm(SCORE_SYSTEM, user_msg)
        parsed = _extract_json(raw)
    except Exception as e:
        return {
            **state,
            "error":          f"score_failed: {str(e)[:120]}",
            "agent_response": (
                "⚠️ I wasn't able to produce a reliable score for this input after retrying. "
                "This usually means the use case needs more detail. "
                "Could you describe the agent's autonomy level, data it accesses, and the decisions it makes?"
            ),
        }

    arm = ARMScores(
        execution_integrity      = float(parsed["arm"]["execution_integrity"]),
        decision_integrity       = float(parsed["arm"]["decision_integrity"]),
        capability_debt          = float(parsed["arm"]["capability_debt"]),
        knowledge_sustainability = float(parsed["arm"]["knowledge_sustainability"]),
        enterprise_adaptation    = float(parsed["arm"]["enterprise_adaptation"]),
    )
    avre = AVREScores(
        roi_score        = float(parsed["avre"]["roi_score"]),
        roe_score        = float(parsed["avre"]["roe_score"]),
        rof_score        = float(parsed["avre"]["rof_score"]),
        opportunity_cost = float(parsed["avre"]["opportunity_cost"]),
    )
    result = ScoringResult(
        use_case_name        = state.get("use_case_name", "Unnamed Use Case"),
        use_case_description = state["user_input"],
        arm                  = arm,
        avre                 = avre,
        rationale            = parsed.get("rationale", {}),
        rag_sources          = rag["sources"],
        recommendations      = parsed.get("recommendations", []),
    )

    nev = result.net_enterprise_value()
    br  = result.avre.benefit_realization()

    # ── Conversational narrative response ─────────────────────────────────────
    arm_score  = result.arm.composite()
    top_risk   = _top_risk(result.arm)
    tier       = result.arm.risk_tier()
    nev_rating = result.nev_rating()
    recs       = result.recommendations

    # Tier-based opening line
    if arm_score >= 7.6:
        opening = f"Honestly, this one raises serious red flags. With an ARM™ score of **{arm_score:.1f}/10**, this use case lands in **CRITICAL** territory — that means do not deploy without executive sign-off and an independent audit."
    elif arm_score >= 5.6:
        opening = f"This is a high-risk deployment. The ARM™ score comes in at **{arm_score:.1f}/10** — that's a **RED** rating, meaning phased deployment only with an independent review before you go live."
    elif arm_score >= 3.1:
        opening = f"This one is manageable but needs careful handling. ARM™ score is **{arm_score:.1f}/10** — **AMBER** — so you can proceed, but enhanced controls are non-negotiable."
    else:
        opening = f"Good news here. ARM™ score is **{arm_score:.1f}/10** — that's a **GREEN** rating. This use case is relatively safe to agentify with standard governance in place."

    # Top risk explanation
    risk_narrative = f"The biggest concern is **{top_risk}**. "
    if "Decision Integrity" in top_risk:
        risk_narrative += "When an agent screens candidates and sends rejection emails autonomously, humans stop making real decisions — they just rubber-stamp what the AI recommends. That's Concurrence Validation, and it compounds quietly."
    elif "Capability Debt" in top_risk:
        risk_narrative += "The deeper risk here is Capability Debt — over time, the people who used to do this work lose the skill to do it. If the agent fails or needs to be replaced, you may not have the human expertise to fall back on."
    elif "Execution Integrity" in top_risk:
        risk_narrative += "The concern is whether the work can still be executed correctly without the agent. If the system fails or produces errors, is there a human who can step in and catch it?"
    elif "Knowledge Sustainability" in top_risk:
        risk_narrative += "Key institutional knowledge is at risk of being locked inside the agent's logic rather than staying with your people. That creates a fragile dependency."
    elif "Enterprise Adaptation" in top_risk:
        risk_narrative += "Edge cases are the real danger here. This agent will handle routine scenarios well — but when something novel comes up, can the organisation still respond without the agent's help?"

    # Value narrative
    if br >= 7:
        value_narrative = f"On the value side, the AVRE™ Benefit Realization score is **{br:.1f}/10** — strong ROI and operational improvement potential."
    elif br >= 5:
        value_narrative = f"The AVRE™ Benefit Realization score is **{br:.1f}/10** — there's real value here, but it's not transformational on its own."
    else:
        value_narrative = f"The AVRE™ Benefit Realization score is only **{br:.1f}/10** — the value case is weak relative to the risk exposure. That's a problem."

    # NEV conclusion
    if nev >= 7:
        nev_narrative = f"Putting it together, the Net Enterprise Value is **{nev:.1f}/10** — worth pursuing with the right controls."
    elif nev >= 4:
        nev_narrative = f"Net Enterprise Value lands at **{nev:.1f}/10** — marginal. The risk is eating into the value. Fix the governance architecture first, then revisit."
    else:
        nev_narrative = f"Net Enterprise Value is **{nev:.1f}/10** — the risk penalty is overwhelming the value case. I'd recommend a redesign before committing to this deployment."

    # Recommendations
    rec_narrative = ""
    if recs:
        rec_narrative = f"\n\nIf you do move forward, the top priority is: {recs[0]}"
        if len(recs) > 1:
            rec_narrative += f" Also worth addressing: {recs[1]}"

    # Corpus signal
    l2_chunks = rag["layer_breakdown"]["l2_chunks"]
    confidence = "🟢 High" if len(l2_chunks) >= 2 else ("🟡 Medium — limited client context" if len(l2_chunks) == 1 else "🔴 Low — no client corpus matched")

    response = (
        f"{opening}\n\n"
        f"{risk_narrative}\n\n"
        f"{value_narrative} {nev_narrative}"
        f"{rec_narrative}\n\n"
        f"---\n"
        f"*ARM™ {arm_score:.1f} · AVRE™ BR {br:.1f} · NEV {nev:.1f}/10 · Confidence: {confidence}*\n\n"
        f"*Want a deeper compliance breakdown, a full governance brief, or to rescore with more context? Just ask.*"
    )

    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": response})

    return {
        **state,
        "messages":       messages,
        "agent_response": response,
        "last_result":    result.to_dict(),
        "rag_sources":    rag["sources"],
        "error":          None,
    }


def _top_risk(arm: ARMScores) -> str:
    dims = {
        "Execution Integrity":      arm.execution_integrity,
        "Decision Integrity":       arm.decision_integrity,
        "Capability Debt":          arm.capability_debt,
        "Knowledge Sustainability": arm.knowledge_sustainability,
        "Enterprise Adaptation":    arm.enterprise_adaptation,
    }
    top = max(dims, key=dims.get)
    return f"{top} ({dims[top]:.1f}/10)"


# ─────────────────────────────────────────────────────────────────────────────
# CLARIFY AGENT
# Asks ONE targeted follow-up question to resolve ambiguity before scoring.
# ─────────────────────────────────────────────────────────────────────────────

CLARIFY_SYSTEM = """You are a friendly but sharp Enterprise AI Governance advisor.

The user has described something that needs more detail before you can score it accurately.
Ask ONE conversational follow-up question — the single most important thing you need to know.
Sound like a trusted advisor talking to a colleague, not a form asking for inputs.
No preamble, no bullet points. Just ask the question naturally."""


def clarify_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Asks a targeted clarifying question."""
    history = _history_str(state.get("messages", []))
    user_msg = f"""Conversation so far:
{history}

Latest input: {state['user_input']}

Ask the single most important clarifying question."""

    question = _llm(CLARIFY_SYSTEM, user_msg, max_tokens=200)

    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": question})

    return {
        **state,
        "messages":       messages,
        "agent_response": question,
        "error":          None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE AGENT
# Checks the use case against Layer 2 regulatory corpus
# (Meridian policy, NAIC, Colorado SB 169, NYDFS, EU AI Act, FCRA, GLBA).
# ─────────────────────────────────────────────────────────────────────────────

COMPLIANCE_SYSTEM = """You are a knowledgeable Enterprise AI Compliance advisor for an insurance company — think trusted in-house counsel, not a legal textbook.

Given a use case and regulatory context, walk the user through:
- Which regulations actually apply and why it matters for their specific situation
- What they concretely need to do before deploying
- Any hard blockers they can't work around without additional controls

Be direct and plain-spoken. Use regulation names precisely, but explain what they mean in practice.
No walls of bullet points — write like you're briefing a smart executive who doesn't have time for legalese."""


def compliance_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Regulatory gap analysis against Layer 2 compliance corpus."""
    corpus: DualCorpus = state["corpus"]

    # Compliance queries primarily target Layer 2
    rag = corpus.retrieve(
        f"regulatory compliance requirements: {state['user_input']}",
        layers="l2",
        k=6,
    )

    user_msg = f"""USE CASE: {state['user_input']}

REGULATORY CONTEXT (Meridian policy, NAIC, Colorado SB 169, NYDFS, EU AI Act, FCRA, GLBA):
{rag['context']}

Provide a compliance gap analysis for this use case."""

    response = _llm(COMPLIANCE_SYSTEM, user_msg, max_tokens=1200)

    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": response})

    return {
        **state,
        "messages":       messages,
        "agent_response": response,
        "rag_sources":    rag["sources"],
        "error":          None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH AGENT
# Free-text RAG search across both layers — returns grounded, cited answer.
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_SYSTEM = """You are a knowledgeable Enterprise AI Governance advisor with deep expertise in ARM™, AVRE™, EU AI Act, and NIST AI RMF.

Answer the user's question conversationally — like a colleague who knows this stuff well and is explaining it clearly.
Ground your answer in the retrieved context. If something isn't in the context, say so honestly rather than guessing.
Cite sources naturally in the flow of your answer, not as a footnote list.
Keep it crisp — no unnecessary hedging, no walls of text."""


def search_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """RAG search across both corpus layers."""
    corpus: DualCorpus = state["corpus"]

    rag = corpus.retrieve(state["user_input"], layers="both", k=5)

    user_msg = f"""QUESTION: {state['user_input']}

RETRIEVED CONTEXT:
{rag['context']}

Answer the question based solely on the context above. Cite your sources."""

    response = _llm(SEARCH_SYSTEM, user_msg, max_tokens=800)

    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": response})

    return {
        **state,
        "messages":       messages,
        "agent_response": response,
        "rag_sources":    rag["sources"],
        "error":          None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BRIEF AGENT
# Synthesises full conversation into a structured governance brief.
# Output is stored in state["pending_brief"] and held at the HITL gate.
# ─────────────────────────────────────────────────────────────────────────────

BRIEF_SYSTEM = """You are an Enterprise AI Governance advisor writing a formal governance brief.

Synthesise the conversation history into a structured governance brief with these sections:

## Governance Brief — [Use Case Name]

**Executive Summary** (2-3 sentences)

**Use Case Description**

**ARM™ Risk Assessment**
- Composite Score and tier
- Top 2-3 risk dimensions and rationale

**AVRE™ Value Assessment**
- Net Enterprise Value
- Key value drivers

**Regulatory Exposure**
- Applicable regulations
- Key obligations

**Deployment Recommendation**
- Go / Conditional Go / No-Go
- Required controls before deployment

**Next Steps** (3 bullet points)

---
*Generated by AVS Agent Mode | ARM™ × AVRE™ | Pending human review*

Be precise and professional. Use the conversation history to ground every claim."""


def brief_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesises conversation into a governance brief for HITL review."""
    history = _history_str(state.get("messages", []), max_turns=20)

    user_msg = f"""Conversation history:
{history}

Use case name: {state.get('use_case_name', 'Unnamed')}
Last scoring result: {json.dumps(state.get('last_result', {}), indent=2)[:1500]}

Write the governance brief now."""

    brief_text = _llm(BRIEF_SYSTEM, user_msg, max_tokens=2000)

    # Store in pending_brief — graph will INTERRUPT here for HITL approval
    messages = state.get("messages", [])
    messages.append({
        "role": "assistant",
        "content": "I've drafted a governance brief based on everything we've discussed. Take a look — hit **Approve** to save it permanently or **Reject** if you'd like to change anything."
    })

    return {
        **state,
        "messages":        messages,
        "agent_response":  "Brief ready for your review — approve to save it permanently.",
        "pending_brief":   brief_text,
        "hitl_required":   True,
        "error":           None,
    }
