"""
app.py
Agentification Value Scorer — Week 3
ARM™ × AVRE™ Governance Intelligence Engine

Layout:
  Main window  — mode toggle (Score Mode / Agent Mode) + active mode UI
  Sidebar      — Evaluation Report · About · Session controls
"""

import sys
import json
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine   import build_vectorstore, get_retriever, retrieve_context
from llm_scorer   import score_use_case
from evaluator    import run_evaluation_suite
from corpus       import build_dual_corpus
from memory       import get_checkpointer, get_or_create_session_id, reset_session
from graph        import build_graph, run_graph
from hitl         import render_hitl_gate, render_saved_briefs_sidebar

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentification Value Scorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0f1117; }

/* Mode toggle */
.mode-toggle {
    display: flex; gap: 0.5rem; margin-bottom: 2rem;
}
.mode-btn {
    flex: 1; padding: 0.75rem 1rem; border-radius: 10px;
    border: 1px solid #2d2f45; background: #1a1d2e;
    color: #888; font-size: 0.9rem; font-weight: 600;
    text-align: center; cursor: pointer;
    transition: all 0.2s;
}
.mode-btn.active {
    background: #2d1b69; color: #a78bfa;
    border-color: #4c1d95;
}

/* Header */
.hero-title {
    font-size: 1.9rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.hero-sub { font-size: 0.9rem; color: #666; margin-bottom: 1.5rem; }

/* Score mode cards */
.metric-card  { background:#1a1d2e;border:1px solid #2d2f45;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem; }
.metric-label { font-size:0.75rem;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.3rem; }
.metric-value { font-size:2.5rem;font-weight:700;line-height:1; }
.metric-sub   { font-size:0.8rem;color:#888;margin-top:0.3rem; }
.risk-card    { background:#1a1d2e;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem;border-left:4px solid; }
.risk-label   { font-size:0.8rem;color:#aaa;margin-bottom:0.2rem; }
.risk-score   { font-size:1.6rem;font-weight:700; }
.risk-rationale { font-size:0.75rem;color:#888;margin-top:0.4rem;line-height:1.4; }
.tier-banner  { border-radius:12px;padding:1.5rem 2rem;text-align:center;margin:1.5rem 0; }
.nev-card     { border-radius:16px;padding:2rem;text-align:center;margin:1.5rem 0; }
.nev-label    { font-size:0.9rem;font-weight:600;opacity:0.8;margin-bottom:0.5rem; }
.nev-value    { font-size:4rem;font-weight:800;line-height:1; }
.nev-rating   { font-size:1rem;font-weight:500;margin-top:0.5rem;opacity:0.9; }
.rec-card     { background:#1a1d2e;border:1px solid #2d2f45;border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;display:flex;gap:1rem;align-items:flex-start; }
.rec-num      { background:#2d1b69;color:#a78bfa;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0; }
.rec-text     { font-size:0.85rem;color:#ccc;line-height:1.5; }
.source-tag   { display:inline-block;background:#1e3a5f;color:#60a5fa;border:1px solid #1d4ed8;border-radius:6px;padding:2px 10px;font-size:0.75rem;margin-right:6px;margin-top:4px; }
.section-header { font-size:0.75rem;font-weight:600;color:#666;text-transform:uppercase;letter-spacing:0.1em;margin:1.5rem 0 0.8rem;padding-bottom:0.5rem;border-bottom:1px solid #2d2f45; }

/* Chat styles */
.chat-bubble-user {
    background:#1e3a5f; border:1px solid #1d4ed8; border-radius:12px 12px 2px 12px;
    padding:0.8rem 1.1rem; margin:0.5rem 0; color:#e2e8f0; font-size:0.9rem;
    max-width:85%; margin-left:auto;
}
.chat-bubble-assistant {
    background:#1a1d2e; border:1px solid #2d2f45; border-radius:12px 12px 12px 2px;
    padding:0.8rem 1.1rem; margin:0.5rem 0; color:#e2e8f0; font-size:0.9rem;
    max-width:90%;
}
.intent-badge {
    display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.7rem;
    font-weight:700; letter-spacing:0.06em; margin-bottom:0.5rem;
}
.intent-SCORE      { background:#2d1b69; color:#a78bfa; border:1px solid #4c1d95; }
.intent-CLARIFY    { background:#1e3a5f; color:#60a5fa; border:1px solid #1d4ed8; }
.intent-COMPLIANCE { background:#431407; color:#fb923c; border:1px solid #9a3412; }
.intent-SEARCH     { background:#14532d; color:#4ade80; border:1px solid #166534; }
.intent-BRIEF      { background:#1a1a2e; color:#e879f9; border:1px solid #7e22ce; }

/* Eval */
.eval-pass { color:#4ade80;font-weight:600; }
.eval-fail { color:#f87171;font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Colour helpers ─────────────────────────────────────────────────────────────
def score_color(s):
    if s >= 7.6: return "#ef4444"
    elif s >= 5.6: return "#f97316"
    elif s >= 3.1: return "#eab308"
    return "#22c55e"

def value_color(s):
    if s >= 7: return "#22c55e"
    elif s >= 5: return "#a78bfa"
    elif s >= 3: return "#eab308"
    return "#ef4444"

def tier_style(tier):
    if "GREEN"    in tier: return {"bg":"#14532d","color":"#4ade80","label":"🟢 GREEN","text":"Proceed with standard governance"}
    if "AMBER"    in tier: return {"bg":"#451a03","color":"#fbbf24","label":"🟡 AMBER","text":"Enhanced controls required"}
    if "RED"      in tier: return {"bg":"#450a0a","color":"#f87171","label":"🔴 RED","text":"Phased deployment only"}
    return                         {"bg":"#1a1a2e","color":"#f43f5e","label":"⛔ CRITICAL","text":"Do not deploy without exec approval"}

def nev_style(nev):
    if nev >= 7:  return {"bg":"#14532d","color":"#4ade80","rating":"⭐ TRANSFORMATIONAL"}
    if nev >= 6:  return {"bg":"#1e3a5f","color":"#60a5fa","rating":"✅ HIGH VALUE"}
    if nev >= 4:  return {"bg":"#451a03","color":"#fbbf24","rating":"⚠️ MARGINAL"}
    return               {"bg":"#450a0a","color":"#f87171","rating":"❌ POOR"}

# ── Resource loading ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_w2_retriever():
    vs = build_vectorstore()
    return get_retriever(vs)

@st.cache_resource(show_spinner=False)
def load_w3_resources():
    corpus       = build_dual_corpus()
    checkpointer = get_checkpointer()
    graph        = build_graph(checkpointer)
    return corpus, graph

with st.spinner("Loading knowledge base..."):
    retriever = load_w2_retriever()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:0.5rem 0'>
  <div style='font-size:1.1rem;font-weight:700;color:#a78bfa'>⚡ AVS</div>
  <div style='font-size:0.7rem;color:#666'>ARM™ × AVRE™ Intelligence Engine</div>
</div>
""", unsafe_allow_html=True)

sidebar_page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Evaluation Report", "📖 About"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE — mode toggle + active mode
# ══════════════════════════════════════════════════════════════════════════════

if sidebar_page == "🏠 Home":

    # Header
    st.markdown('<div class="hero-title">Agentification Value Scorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ARM™ × AVRE™ · Dual corpus · LangGraph multi-agent</div>', unsafe_allow_html=True)

    # Mode toggle
    if "mode" not in st.session_state:
        st.session_state.mode = "score"

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(
            "⚡  Score Mode",
            use_container_width=True,
            type="primary" if st.session_state.mode == "score" else "secondary",
        ):
            st.session_state.mode = "score"
            st.rerun()
    with col_b:
        if st.button(
            "🤖  Agent Mode",
            use_container_width=True,
            type="primary" if st.session_state.mode == "agent" else "secondary",
        ):
            st.session_state.mode = "agent"
            st.rerun()

    st.markdown("---")

    # ── SCORE MODE ─────────────────────────────────────────────────────────────
    if st.session_state.mode == "score":

        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown('<div class="section-header">Use Case Details</div>', unsafe_allow_html=True)
            uc_name = st.text_input("Use Case Name", placeholder="e.g. Claims Triage Agent")
            uc_desc = st.text_area(
                "Use Case Description",
                height=200,
                placeholder="Describe the AI agent — its scope, autonomy level, data it accesses, decisions it makes, and any known constraints...",
            )
            run = st.button(
                "▶  Score This Use Case",
                type="primary",
                use_container_width=True,
                disabled=not uc_desc.strip(),
            )

        with col_result:
            if run and uc_desc.strip():
                with st.spinner("Retrieving governance context and scoring..."):
                    rag    = retrieve_context(f"ARM risk scoring AVRE value: {uc_name} {uc_desc}", retriever)
                    result = score_use_case(uc_name or uc_desc[:60], uc_desc, rag["context"], rag["sources"])

                st.markdown('<div class="section-header">ARM™ Risk Assessment</div>', unsafe_allow_html=True)
                arm_dims = [
                    ("Execution Integrity",      result.arm.execution_integrity,      "execution_integrity",      "25%"),
                    ("Decision Integrity",        result.arm.decision_integrity,       "decision_integrity",       "25%"),
                    ("Capability Debt",           result.arm.capability_debt,          "capability_debt",          "20%"),
                    ("Knowledge Sustainability",  result.arm.knowledge_sustainability, "knowledge_sustainability", "15%"),
                    ("Enterprise Adaptation",     result.arm.enterprise_adaptation,    "enterprise_adaptation",    "15%"),
                ]
                for label, score, key, wt in arm_dims:
                    c   = score_color(score)
                    rat = result.rationale.get(key, "")
                    st.markdown(f"""
                    <div class="risk-card" style="border-left-color:{c};display:flex;align-items:center;gap:1.5rem">
                      <div style="min-width:180px">
                        <div class="risk-label">{label} <span style="color:#555">{wt}</span></div>
                        <div class="risk-score" style="color:{c}">{score:.1f}</div>
                      </div>
                      <div style="flex:1">
                        <div style="background:#111;border-radius:4px;height:6px;margin-bottom:0.5rem">
                          <div style="background:{c};height:6px;border-radius:4px;width:{score*10}%"></div>
                        </div>
                        <div class="risk-rationale">{rat[:140]}{"..." if len(rat)>140 else ""}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                ts = tier_style(result.arm.risk_tier())
                st.markdown(f"""
                <div class="tier-banner" style="background:{ts['bg']};border:1px solid {ts['color']}">
                  <div style="font-size:0.85rem;font-weight:600;color:{ts['color']}">ARM™ COMPOSITE: {result.arm.composite():.2f}/10</div>
                  <div style="font-size:1.2rem;font-weight:700;color:{ts['color']}">{ts['label']}</div>
                  <div style="font-size:0.8rem;color:#aaa;margin-top:0.3rem">{ts['text']}</div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="section-header">AVRE™ Value Assessment</div>', unsafe_allow_html=True)
                avre_map = [
                    ("ROI — Return on Investment", result.avre.roi_score,        "roi",            "40%"),
                    ("ROE — Return on Effort",      result.avre.roe_score,        "roe",            "30%"),
                    ("ROF — Return on Future",      result.avre.rof_score,        "rof",            "30%"),
                    ("Opportunity Cost",            result.avre.opportunity_cost, "opportunity_cost","ref"),
                ]
                for label, score, key, wt in avre_map:
                    c   = value_color(score)
                    rat = result.rationale.get(key, "")
                    st.markdown(f"""
                    <div class="risk-card" style="border-left-color:{c};display:flex;align-items:center;gap:1.5rem">
                      <div style="min-width:180px">
                        <div class="risk-label">{label} <span style="color:#555">{wt}</span></div>
                        <div class="risk-score" style="color:{c}">{score:.1f}</div>
                      </div>
                      <div style="flex:1">
                        <div style="background:#111;border-radius:4px;height:6px;margin-bottom:0.5rem">
                          <div style="background:{c};height:6px;border-radius:4px;width:{score*10}%"></div>
                        </div>
                        <div class="risk-rationale">{rat[:140]}{"..." if len(rat)>140 else ""}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                nev = result.net_enterprise_value()
                ns  = nev_style(nev)
                st.markdown(f"""
                <div class="nev-card" style="background:{ns['bg']};border:2px solid {ns['color']}">
                  <div class="nev-label" style="color:{ns['color']}">NET ENTERPRISE VALUE</div>
                  <div class="nev-value" style="color:{ns['color']}">{nev}<span style="font-size:1.5rem">/10</span></div>
                  <div class="nev-rating" style="color:{ns['color']}">{ns['rating']}</div>
                  <div style="font-size:0.8rem;color:#666;margin-top:0.8rem">
                    ARM™ Risk Multiplier: <strong style="color:#aaa">{result.arm.risk_multiplier():.3f}</strong> &nbsp;|&nbsp;
                    Benefit Realization: <strong style="color:#aaa">{result.avre.benefit_realization():.2f}</strong>
                  </div>
                </div>""", unsafe_allow_html=True)

                if result.recommendations:
                    st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
                    for i, rec in enumerate(result.recommendations, 1):
                        st.markdown(f"""
                        <div class="rec-card">
                          <div class="rec-num">{i}</div>
                          <div class="rec-text">{rec}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown('<div class="section-header">RAG Sources</div>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="source-tag">{s}</span>' for s in result.rag_sources), unsafe_allow_html=True)

                out = Path(__file__).parent / "outputs" / f"{(uc_name or 'result').replace(' ','_')[:40]}_score.json"
                out.parent.mkdir(exist_ok=True)
                with open(out, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                st.caption(f"Saved → {out.name}")

    # ── AGENT MODE ─────────────────────────────────────────────────────────────
    else:
        corpus, graph = load_w3_resources()
        session_id    = get_or_create_session_id(st.session_state)

        # Sidebar session controls
        st.sidebar.markdown(f"**Session:** `{session_id[:8]}...`")
        if st.sidebar.button("🔄 New Session", use_container_width=True):
            reset_session(st.session_state)
            st.rerun()
        render_saved_briefs_sidebar(st)

        # Initialise state
        if "messages"     not in st.session_state: st.session_state.messages     = []
        if "agent_state"  not in st.session_state: st.session_state.agent_state  = {}
        if "hitl_pending" not in st.session_state: st.session_state.hitl_pending = False

        # Chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                intent     = msg.get("intent", "")
                intent_html = f'<span class="intent-badge intent-{intent}">{intent}</span><br>' if intent else ""
                st.markdown(f'<div class="chat-bubble-assistant">{intent_html}{msg["content"]}</div>', unsafe_allow_html=True)

        # HITL gate
        if st.session_state.hitl_pending:
            result_msg, approved = render_hitl_gate(st, st.session_state.agent_state, session_id)
            if result_msg is not None:
                st.session_state.hitl_pending = False
                st.session_state.agent_state["pending_brief"] = None
                st.session_state.messages.append({"role": "assistant", "content": result_msg, "intent": "BRIEF"})
                st.rerun()

        # Chat input
        if not st.session_state.hitl_pending:
            user_input = st.chat_input("Describe a use case, ask a compliance question, or say 'generate brief'...")

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.spinner("Agent thinking..."):
                    try:
                        final_state = run_graph(
                            graph          = graph,
                            user_input     = user_input,
                            session_id     = session_id,
                            corpus         = corpus,
                            existing_state = st.session_state.agent_state,
                        )
                        st.session_state.agent_state = {k: v for k, v in final_state.items() if k not in ("corpus",)}
                        intent   = final_state.get("intent", "SCORE")
                        response = final_state.get("agent_response", "")

                        if final_state.get("hitl_required") and final_state.get("pending_brief"):
                            st.session_state.hitl_pending = True
                            st.session_state.agent_state["pending_brief"] = final_state["pending_brief"]

                        st.session_state.messages.append({"role": "assistant", "content": response, "intent": intent})
                        st.session_state.agent_state["messages"] = [
                            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                        ]
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Agent error: {str(e)[:200]}. Please try again.", "intent": ""})
                st.rerun()

        # Quick actions
        if st.session_state.messages:
            st.markdown("---")
            qa1, qa2, qa3, qa4 = st.columns(4)
            if qa1.button("📋 Generate Brief",   use_container_width=True):
                st.session_state._quick_action = "Generate a governance brief for this conversation"
            if qa2.button("🔍 Compliance Check", use_container_width=True):
                st.session_state._quick_action = "What are the compliance obligations for the use case we discussed?"
            if qa3.button("🔄 Rescore",          use_container_width=True):
                st.session_state._quick_action = "rescore — please re-evaluate the use case with the full context of our conversation"
            if qa4.button("🗑 Clear Chat",        use_container_width=True):
                reset_session(st.session_state)
                st.rerun()

            if hasattr(st.session_state, "_quick_action") and st.session_state._quick_action:
                action = st.session_state._quick_action
                del st.session_state._quick_action
                st.session_state.messages.append({"role": "user", "content": action})
                with st.spinner("Agent thinking..."):
                    try:
                        final_state = run_graph(graph, action, session_id, corpus, st.session_state.agent_state)
                        st.session_state.agent_state = {k: v for k, v in final_state.items() if k != "corpus"}
                        if final_state.get("hitl_required") and final_state.get("pending_brief"):
                            st.session_state.hitl_pending = True
                            st.session_state.agent_state["pending_brief"] = final_state["pending_brief"]
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_state.get("agent_response", ""),
                            "intent": final_state.get("intent", ""),
                        })
                        st.session_state.agent_state["messages"] = [
                            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                        ]
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}", "intent": ""})
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

elif sidebar_page == "📊 Evaluation Report":
    st.markdown('<div class="hero-title">RAG Evaluation Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">20-question evaluation · ARM™ published domains · AVRE™ published lenses · Dual corpus k=4</div>', unsafe_allow_html=True)

    report_path = Path(__file__).parent / "outputs" / "eval_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        summary = report["evaluation_summary"]

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, sub in [
            (c1, "Overall Faithfulness", f"{summary['avg_faithfulness_pct']}%",          "across scoreable Qs"),
            (c2, "Source Accuracy",      f"{summary['source_retrieval_accuracy_pct']}%",  "correct doc retrieved"),
            (c3, "Questions Passing",    summary['questions_passing_50pct'],               "at ≥50% faithfulness"),
            (c4, "Retrieval",            f"MMR k={summary.get('retrieval_k', 4)}",         "dual corpus layers"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:#22c55e;font-size:1.8rem">{val}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        l1f = summary.get("l1_faithfulness_pct", "—")
        l2f = summary.get("l2_faithfulness_pct", "—")
        ca, cb = st.columns(2)
        ca.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Layer 1 Faithfulness</div>
          <div class="metric-value" style="color:#a78bfa;font-size:1.8rem">{l1f}%</div>
          <div class="metric-sub">ARM™ · AVRE™ · EU AI Act · NIST AI RMF</div>
        </div>""", unsafe_allow_html=True)
        cb.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Layer 2 Faithfulness</div>
          <div class="metric-value" style="color:#60a5fa;font-size:1.8rem">{l2f}%</div>
          <div class="metric-sub">Meridian policy · Regulation · Cost parameters</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#1a1d2e;border:1px solid #2d2f45;border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1rem;font-size:0.8rem;color:#666">
          <strong style="color:#aaa">ARM™ Domains:</strong> {summary.get('arm_domains','—')}<br>
          <strong style="color:#aaa">AVRE™ Lenses:</strong> {summary.get('avre_lenses','—')}<br>
          <strong style="color:#aaa">Corpus:</strong> {summary.get('corpus_layers','—')}
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Question-by-Question Results</div>', unsafe_allow_html=True)
        for r in report["results"]:
            faith = r["faithfulness_score"]
            color = "#22c55e" if faith >= 0.75 else "#eab308" if faith >= 0.5 else "#ef4444"
            icon  = "✅" if faith >= 0.5 else "❌"
            layer = r.get("layer", "L1")
            layer_color = "#a78bfa" if layer == "L1" else "#60a5fa" if layer == "L2" else "#4ade80"
            with st.expander(f"{icon} {r['id']} [{layer}] — {r['category']} · {r['query'][:60]}..."):
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.markdown(f"**Faithfulness:** <span style='color:{color}'>{faith*100:.0f}%</span>", unsafe_allow_html=True)
                cc2.markdown(f"**Source hit:** {'✅' if r['source_hit'] else '❌'}")
                cc3.markdown(f"**Difficulty:** {r['difficulty']}")
                cc4.markdown(f"**Layer:** <span style='color:{layer_color}'>{layer}</span>", unsafe_allow_html=True)
                st.markdown(f"**Query:** {r['query']}")
                st.markdown(f"**Sources retrieved:** {', '.join(r['retrieved_sources'])}")
                if r.get("notes"): st.info(r["notes"])
    else:
        st.warning("No evaluation report found. Run the suite to generate one.")
        if st.button("▶ Run 20-Question Evaluation Suite", type="primary"):
            with st.spinner("Running evaluation across dual corpus..."):
                run_evaluation_suite(retriever)
            st.success("Done! Refresh to see results.")

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif sidebar_page == "📖 About":
    st.markdown('<div class="hero-title">About the Agentification Value Scorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ARM™ × AVRE™ · Week 3 · The Gen Academy Mastering Agentic AI Bootcamp</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:760px;color:#aaa;line-height:1.8">

    <p>The <strong style="color:#ccc">Agentification Value Scorer (AVS)</strong> is a RAG-powered governance
    intelligence platform that scores enterprise AI agent use cases for readiness using two proprietary frameworks
    authored by Soumya V. Jom.</p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div style="background:#1a1d2e;border:1px solid #2d2f45;border-radius:14px;padding:1.5rem">
          <div style="font-size:0.7rem;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem">⚡ Score Mode</div>
          <div style="color:#e2e8f0;font-size:0.95rem;font-weight:600;margin-bottom:0.8rem">Single-shot RAG governance scoring</div>
          <div style="color:#888;font-size:0.85rem;line-height:1.7">
            Describe any enterprise AI agent use case and receive an immediate, framework-grounded score.<br><br>
            <strong style="color:#ccc">What it produces:</strong><br>
            • ARM™ score across 5 risk domains<br>
            • AVRE™ score across 3 value lenses<br>
            • Net Enterprise Value (NEV) 0–10<br>
            • Risk tier classification<br>
            • 3 actionable recommendations<br><br>
            <strong style="color:#ccc">Best for:</strong> Quick assessments, ideation reviews, executive briefings
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1a1d2e;border:1px solid #2d2f45;border-radius:14px;padding:1.5rem">
          <div style="font-size:0.7rem;font-weight:700;color:#60a5fa;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem">🤖 Agent Mode</div>
          <div style="color:#e2e8f0;font-size:0.95rem;font-weight:600;margin-bottom:0.8rem">LangGraph multi-agent conversation</div>
          <div style="color:#888;font-size:0.85rem;line-height:1.7">
            A stateful chat interface backed by 5 specialist agents and a dual governance corpus.<br><br>
            <strong style="color:#ccc">Five intent modes:</strong><br>
            • <strong style="color:#a78bfa">SCORE</strong> — ARM™ × AVRE™ dual-corpus scoring<br>
            • <strong style="color:#60a5fa">CLARIFY</strong> — targeted follow-up before scoring<br>
            • <strong style="color:#fb923c">COMPLIANCE</strong> — regulatory gap analysis<br>
            • <strong style="color:#4ade80">SEARCH</strong> — framework knowledge retrieval<br>
            • <strong style="color:#e879f9">BRIEF</strong> — governance brief with HITL gate<br><br>
            <strong style="color:#ccc">Best for:</strong> Deep assessments, compliance reviews, brief generation
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1a1d2e;border:1px solid #2d2f45;border-radius:14px;padding:1.5rem;max-width:760px">
      <div style="font-size:0.7rem;font-weight:700;color:#666;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:1rem">The Framework Architecture</div>
      <div style="display:flex;gap:1rem;flex-wrap:wrap">
        <div style="flex:1;min-width:180px">
          <div style="color:#a78bfa;font-weight:700;font-size:0.85rem">ARM™</div>
          <div style="color:#888;font-size:0.8rem;margin-top:0.3rem">Agentification Risk Model<br>18 risks · 5 domains<br>Are we building it responsibly?</div>
        </div>
        <div style="flex:1;min-width:180px">
          <div style="color:#60a5fa;font-weight:700;font-size:0.85rem">AVRE™</div>
          <div style="color:#888;font-size:0.8rem;margin-top:0.3rem">Agentic Value Realization Engine<br>ROI · ROE · ROF · NEV<br>Are we creating real enterprise value?</div>
        </div>
        <div style="flex:1;min-width:180px">
          <div style="color:#4ade80;font-weight:700;font-size:0.85rem">TraceSense™</div>
          <div style="color:#888;font-size:0.8rem;margin-top:0.3rem">Bidirectional traceability<br>Requirements to outcomes<br>Are we building the right AI?</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#555;font-size:0.8rem;max-width:760px">
      <strong style="color:#666">Tech Stack:</strong> LangGraph · LangChain · Custom TF-IDF + MMR (dual corpus) · Llama 3.3 70B via Nebius Token Factory · SQLite · Streamlit<br>
      <strong style="color:#666">Author:</strong> Soumya V. Jom | Enterprise AI Strategy & Governance, Cognizant PS&E<br>
      <strong style="color:#666">Bootcamp:</strong> The Gen Academy Mastering Agentic AI · Week 3
    </div>
    """, unsafe_allow_html=True)
