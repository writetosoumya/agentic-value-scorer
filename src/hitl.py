"""
hitl.py
Human-in-the-Loop gate for the AVS Week 3 governance brief workflow.

The brief_agent drafts a governance brief and sets state["hitl_required"] = True.
The LangGraph interrupt_before=["brief_node"] pauses execution BEFORE the node runs.

This module handles:
  - Rendering the pending brief in the Streamlit UI
  - Approve  → saves brief to disk, clears pending state, resumes conversation
  - Reject   → clears pending brief, returns feedback prompt to user

Brief files are saved to: briefs/{session_id[:8]}_{use_case_slug}.md
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

BRIEFS_DIR = Path(__file__).parent.parent / "briefs"


# ── File persistence ──────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    """Convert use case name to a safe filename slug."""
    return re.sub(r"[^\w]+", "_", text.lower())[:40].strip("_")


def save_brief(
    brief_text: str,
    use_case_name: str,
    session_id: str,
) -> Path:
    """
    Persist an approved governance brief to disk as Markdown.
    Returns the saved file path.
    """
    BRIEFS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    slug      = _slug(use_case_name or "governance_brief")
    filename  = f"{session_id[:8]}_{timestamp}_{slug}.md"
    path      = BRIEFS_DIR / filename

    header = (
        f"# Governance Brief — {use_case_name}\n"
        f"**Generated:** {datetime.now().strftime('%B %d, %Y %H:%M')}\n"
        f"**Session:** {session_id[:8]}\n"
        f"**Status:** ✅ Approved by human reviewer\n\n"
        f"---\n\n"
    )
    path.write_text(header + brief_text, encoding="utf-8")
    return path


def list_saved_briefs() -> list:
    """Return all saved brief files, newest first."""
    BRIEFS_DIR.mkdir(exist_ok=True)
    return sorted(BRIEFS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)


# ── Streamlit HITL UI component ───────────────────────────────────────────────

def render_hitl_gate(st, state: dict, session_id: str) -> Tuple[Optional[str], bool]:
    """
    Render the HITL approval gate in Streamlit.

    Displays:
      - The drafted governance brief
      - Approve button → saves brief, returns (path_str, True)
      - Reject button  → clears brief, returns (feedback_msg, False)

    Returns: (result_message, approved: bool)
    Called from app.py when state["hitl_required"] is True.
    """
    brief_text     = state.get("pending_brief", "")
    use_case_name  = state.get("use_case_name", "Governance Brief")

    st.markdown("---")
    st.markdown(
        """
        <div style='background:#1a2744;border:1px solid #3b82f6;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1rem'>
          <div style='color:#60a5fa;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em'>
            ⏸ Human-in-the-Loop Gate
          </div>
          <div style='color:#e2e8f0;margin-top:0.4rem;font-size:0.95rem'>
            The governance brief is ready for your review. Approve to save it permanently, or reject to revise.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show brief in an expander so it doesn't dominate the UI
    with st.expander("📄 Review Governance Brief", expanded=True):
        st.markdown(brief_text)

    col1, col2, col3 = st.columns([2, 2, 4])

    with col1:
        approved = st.button(
            "✅ Approve & Save",
            type="primary",
            key=f"hitl_approve_{session_id[:8]}",
            use_container_width=True,
        )

    with col2:
        rejected = st.button(
            "✗ Reject",
            type="secondary",
            key=f"hitl_reject_{session_id[:8]}",
            use_container_width=True,
        )

    if approved:
        path = save_brief(brief_text, use_case_name, session_id)
        msg  = f"✅ **Brief approved and saved** → `{path.name}`\n\nYou can download it from the **Saved Briefs** section in the sidebar."
        return msg, True

    if rejected:
        msg = "Brief rejected. Tell me what you'd like to change and I'll revise it."
        return msg, False

    return None, False


def render_saved_briefs_sidebar(st):
    """Render a sidebar section listing all saved briefs with download links."""
    briefs = list_saved_briefs()
    if not briefs:
        st.sidebar.caption("No saved briefs yet.")
        return

    st.sidebar.markdown("**📁 Saved Briefs**")
    for brief_path in briefs[:5]:  # show last 5
        text = brief_path.read_text(encoding="utf-8")
        st.sidebar.download_button(
            label=f"⬇ {brief_path.stem[:30]}",
            data=text,
            file_name=brief_path.name,
            mime="text/markdown",
            key=f"dl_{brief_path.stem}",
        )
