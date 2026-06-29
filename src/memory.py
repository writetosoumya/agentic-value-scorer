"""
memory.py
SQLite-backed session memory for the Week 3 AVS agent graph.

Uses LangGraph's SqliteSaver checkpointer to persist:
  - Full conversation history per session
  - Last intent routed
  - Accumulated scoring context across turns
  - Draft governance brief (pre-HITL)

Session IDs are stored in Streamlit session_state so each browser
tab gets an isolated conversation thread.
"""

import uuid
import sqlite3
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = Path(__file__).parent.parent / "outputs" / "sessions.db"


def get_checkpointer() -> SqliteSaver:
    """
    Return a SqliteSaver checkpointer connected to the local sessions DB.
    LangGraph uses this to persist the full graph state between invocations.
    """
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    return SqliteSaver(conn)


def new_session_id() -> str:
    """Generate a fresh session UUID."""
    return str(uuid.uuid4())


def get_or_create_session_id(st_session) -> str:
    """
    Retrieve the current session ID from Streamlit session_state,
    creating a new one if this is a fresh browser session.
    """
    if "avs_session_id" not in st_session:
        st_session["avs_session_id"] = new_session_id()
    return st_session["avs_session_id"]


def reset_session(st_session) -> str:
    """Force a new session — clears history for a fresh conversation."""
    st_session["avs_session_id"] = new_session_id()
    # Clear related state too
    for key in ["messages", "pending_brief", "hitl_pending"]:
        st_session.pop(key, None)
    return st_session["avs_session_id"]


def list_sessions(limit: int = 20) -> list:
    """
    Return recent session IDs and their first user message (for debug/admin).
    Reads directly from SQLite — not needed for normal operation.
    """
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC LIMIT ?",
            (limit,)
        )
        return [row[0] for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()
