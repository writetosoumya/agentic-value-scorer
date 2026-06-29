"""
graph.py
LangGraph multi-agent state machine for AVS Week 3.

FIX: DualCorpus is NOT stored in LangGraph state (causes msgpack serialization error).
     Instead it lives in a module-level _corpus_store dict keyed by session_id.
     Nodes retrieve it via get_corpus(session_id).
"""

import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from corpus import DualCorpus
from agents import (
    score_agent, clarify_agent, compliance_agent,
    search_agent, brief_agent, is_vague, VAGUENESS_THRESHOLD
)

# ── Module-level corpus store (not checkpointed) ──────────────────────────────
_corpus_store: Dict[str, DualCorpus] = {}

def set_corpus(session_id: str, corpus: DualCorpus):
    _corpus_store[session_id] = corpus

def get_corpus(session_id: str) -> Optional[DualCorpus]:
    return _corpus_store.get(session_id)


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    user_input:     str
    use_case_name:  str
    intent:         str
    session_id:     str
    messages:       List[Dict[str, str]]
    agent_response: str
    last_result:    Dict[str, Any]
    rag_sources:    List[str]
    pending_brief:  Optional[str]
    hitl_required:  bool
    error:          Optional[str]


# ── Agent node wrappers ───────────────────────────────────────────────────────

def _inject_corpus(state: AgentState) -> AgentState:
    sid    = state.get("session_id", "")
    corpus = get_corpus(sid)
    return {**state, "corpus": corpus}

def score_node(state: AgentState)      -> AgentState: return score_agent(_inject_corpus(state))
def clarify_node(state: AgentState)    -> AgentState: return clarify_agent(_inject_corpus(state))
def compliance_node(state: AgentState) -> AgentState: return compliance_agent(_inject_corpus(state))
def search_node(state: AgentState)     -> AgentState: return search_agent(_inject_corpus(state))
def brief_node(state: AgentState)      -> AgentState: return brief_agent(_inject_corpus(state))


# ── Fast keyword intent classifier (replaces LLM router — saves 2-4s/turn) ───

# Keyword sets for each intent — ordered by priority
_BRIEF_KEYWORDS = {
    "generate brief", "create brief", "save brief", "export brief",
    "governance brief", "create report", "save report", "export report",
    "summarise findings", "summarize findings", "write up", "final report",
    "brief me", "draft brief",
}
_COMPLIANCE_KEYWORDS = {
    "regulation", "regulatory", "comply", "compliance", "legal", "law",
    "naic", "gdpr", "ccpa", "hipaa", "eu ai act", "nist", "iso 42001",
    "colorado sb", "nydfs", "fcra", "glba", "ada", "obligation",
    "requirement", "policy", "prohibited", "permitted", "audit",
    "disclosure", "filing", "registration",
}
_SEARCH_KEYWORDS = {
    "what is", "what are", "explain", "how does", "how do", "define",
    "tell me about", "describe", "what does", "meaning of", "definition",
    "how works", "can you explain", "help me understand",
}
_SCORE_KEYWORDS = {
    "score", "evaluate", "assess", "rate", "agentif", "deploy",
    "implement", "automate", "automation", "agent", "bot", "system",
    "tool", "workflow", "process", "risk", "value", "worth it",
    "should we", "can we", "is it safe", "is it risky", "too risky",
    "ready", "readiness",
}
_RESCORE_KEYWORDS = {"rescore", "re-score", "score again", "re evaluate", "redo the score"}


def _keyword_classify(text: str) -> str:
    """
    Zero-latency keyword intent classifier.
    Replaces the LLM router — saves one full Nebius round-trip per turn.
    Priority: BRIEF > COMPLIANCE > SEARCH > RESCORE > SCORE > CLARIFY
    """
    t = text.lower().strip()

    # BRIEF — highest priority (explicit action request)
    if any(kw in t for kw in _BRIEF_KEYWORDS):
        return "BRIEF"

    # COMPLIANCE — regulatory language is unambiguous
    if any(kw in t for kw in _COMPLIANCE_KEYWORDS):
        return "COMPLIANCE"

    # SEARCH — question patterns
    if any(t.startswith(kw) or kw in t for kw in _SEARCH_KEYWORDS):
        # But if it also has score keywords, lean SCORE
        if not any(kw in t for kw in _SCORE_KEYWORDS):
            return "SEARCH"

    # RESCORE — explicit retry
    if any(kw in t for kw in _RESCORE_KEYWORDS):
        return "SCORE"

    # SCORE — use case evaluation language
    if any(kw in t for kw in _SCORE_KEYWORDS):
        return "SCORE"

    # Default — if nothing matches, try to score
    return "SCORE"


def route_intent(state: AgentState) -> AgentState:
    """
    Fast keyword intent router — zero LLM calls, zero latency.
    Vagueness gate runs first; keyword classifier handles the rest.
    Saves 2-4 seconds per turn vs. LLM-based routing.
    """
    user_input = state.get("user_input", "")
    history    = state.get("messages", [])

    messages = list(history)
    if not messages or messages[-1].get("content") != user_input:
        messages.append({"role": "user", "content": user_input})

    # Vagueness gate — escalate to CLARIFY before wasting a scoring call
    if is_vague(user_input):
        return {
            **state,
            "intent":        "CLARIFY",
            "use_case_name": state.get("use_case_name", ""),
            "messages":      messages,
            "hitl_required": False,
            "error":         None,
        }

    # Keyword classification — instant, no API call
    intent = _keyword_classify(user_input)

    use_case_name = state.get("use_case_name", "")
    if not use_case_name and intent in ("SCORE", "CLARIFY"):
        use_case_name = _extract_use_case_name(user_input)

    return {
        **state,
        "intent":         intent,
        "use_case_name":  use_case_name,
        "messages":       messages,
        "hitl_required":  False,
        "error":          None,
    }


# ── Error node ────────────────────────────────────────────────────────────────

def error_node(state: AgentState) -> AgentState:
    """Fallback node — surfaces clear failure message and re-routes to clarify."""
    error_msg = state.get("error", "unknown error")

    if "score_failed" in error_msg:
        explanation = (
            "**Scoring could not complete** after 3 attempts.\n\n"
            "To get an accurate score, please add:\n"
            "1. **Autonomy level** — does the agent act autonomously or recommend only?\n"
            "2. **Data accessed** — what data does it read or write?\n"
            "3. **Decision stakes** — what's the consequence of a wrong decision?\n\n"
            "*Let me ask a targeted question to help.*"
        )
    else:
        explanation = (
            f"**An error occurred:** `{error_msg[:100]}`\n\n"
            "Let me ask a clarifying question to recover."
        )

    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": explanation})

    clarify_state = {
        **state,
        "messages":       messages,
        "agent_response": explanation,
        "error":          None,
        "intent":         "CLARIFY",
    }
    return clarify_agent(_inject_corpus(clarify_state))


def _extract_use_case_name(text: str) -> str:
    patterns = [
        r"(?:for (?:a|an) )([\w\s]+?) (?:agent|system|bot|tool)",
        r"([\w\s]+?) (?:agent|system)",
        r"(?:score|evaluate|assess) ([\w\s]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()[:60]
    words = text.split()
    return " ".join(words[:8]).strip(".,?!") if words else "Use Case"


# ── Edges ─────────────────────────────────────────────────────────────────────

def dispatch(state: AgentState) -> str:
    intent_map = {
        "SCORE":      "score_node",
        "CLARIFY":    "clarify_node",
        "COMPLIANCE": "compliance_node",
        "SEARCH":     "search_node",
        "BRIEF":      "brief_node",
    }
    return intent_map.get(state.get("intent", "SCORE"), "score_node")


def score_dispatch(state: AgentState) -> str:
    return "error_node" if state.get("error") else END


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph(checkpointer: SqliteSaver) -> Any:
    builder = StateGraph(AgentState)

    builder.add_node("router",          route_intent)
    builder.add_node("score_node",      score_node)
    builder.add_node("clarify_node",    clarify_node)
    builder.add_node("compliance_node", compliance_node)
    builder.add_node("search_node",     search_node)
    builder.add_node("brief_node",      brief_node)
    builder.add_node("error_node",      error_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", dispatch, {
        "score_node":      "score_node",
        "clarify_node":    "clarify_node",
        "compliance_node": "compliance_node",
        "search_node":     "search_node",
        "brief_node":      "brief_node",
    })

    builder.add_conditional_edges("score_node", score_dispatch, {
        "error_node": "error_node",
        END:          END,
    })
    builder.add_edge("error_node",      END)
    builder.add_edge("clarify_node",    END)
    builder.add_edge("compliance_node", END)
    builder.add_edge("search_node",     END)
    builder.add_edge("brief_node",      END)

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["brief_node"],
    )
    return graph


# ── Public run functions ──────────────────────────────────────────────────────

def run_graph(
    graph,
    user_input: str,
    session_id: str,
    corpus: DualCorpus,
    existing_state: Optional[Dict] = None,
) -> AgentState:
    """
    Invoke the graph for one user turn.
    Corpus is stored in module-level dict — NOT in checkpointed state.
    """
    config = {"configurable": {"thread_id": session_id}}

    # Register corpus for this session
    set_corpus(session_id, corpus)

    input_state: AgentState = {
        "user_input":    user_input,
        "session_id":    session_id,
        "messages":      existing_state.get("messages", []) if existing_state else [],
        "use_case_name": existing_state.get("use_case_name", "") if existing_state else "",
        "last_result":   existing_state.get("last_result", {}) if existing_state else {},
        "pending_brief": existing_state.get("pending_brief") if existing_state else None,
        "hitl_required": False,
        "error":         None,
    }

    final_state = graph.invoke(input_state, config=config)
    return final_state


def resume_after_hitl(graph, session_id: str, corpus: DualCorpus) -> AgentState:
    """Resume after HITL interrupt. Re-registers corpus."""
    config = {"configurable": {"thread_id": session_id}}
    set_corpus(session_id, corpus)
    graph.update_state(config, {"session_id": session_id})
    final = graph.invoke(None, config=config)
    return final
