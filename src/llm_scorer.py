"""
llm_scorer.py
LLM-powered scoring for Score Mode (Week 2 RAG pipeline).
Updated to match published ARM™ and AVRE™ framework definitions.

ARM™ domains (published May 26, 2026):
  execution_integrity, decision_integrity, capability_debt,
  knowledge_sustainability, enterprise_adaptation

AVRE™ lenses (published June 25, 2026):
  ROI — Return on Investment
  ROE — Return on Effort
  ROF — Return on Future
"""

import json
import re
import os
from openai import OpenAI
from scoring_engine import ARMScores, AVREScores, ScoringResult

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODEL    = "meta-llama/Llama-3.3-70B-Instruct"

SYSTEM_PROMPT = """You are an expert Enterprise AI Governance analyst specialising in ARM™ and AVRE™ frameworks authored by Soumya V. Jom.

━━━ ARM™ — AGENTIFICATION RISK MODEL ━━━
Score FIVE domains 0–10. Higher = more risk.

1. execution_integrity   — Can the work still be executed correctly without the agent?
2. decision_integrity    — Are humans deciding, or merely approving? (Authority Shaping, Concurrence Validation)
3. capability_debt       — What human capabilities are being consumed? (Cognitive Atrophy, Succession Failure)
4. knowledge_sustainability — Is enterprise knowledge preserved or locked in agent logic?
5. enterprise_adaptation — Can the org still respond when reality changes? (Edge Case Fragility, Escalation Integrity Failure)

━━━ AVRE™ — AGENTIC VALUE REALIZATION ENGINE ━━━
Score THREE lenses 0–10. Higher = more value.

roi_score        — Return on Investment: cost saved, revenue lifted, hours recovered
roe_score        — Return on Effort: did output QUALITY improve, not just volume?
rof_score        — Return on Future: did this build a reusable capability or a dead end?
opportunity_cost — Urgency of agentifying now (0–10, higher = more urgent)

Return ONLY valid JSON — no markdown, no preamble:
{
  "arm": {
    "execution_integrity": <float 0-10>,
    "decision_integrity": <float 0-10>,
    "capability_debt": <float 0-10>,
    "knowledge_sustainability": <float 0-10>,
    "enterprise_adaptation": <float 0-10>
  },
  "avre": {
    "roi_score": <float 0-10>,
    "roe_score": <float 0-10>,
    "rof_score": <float 0-10>,
    "opportunity_cost": <float 0-10>
  },
  "rationale": {
    "execution_integrity": "<1-2 sentence rationale>",
    "decision_integrity": "<1-2 sentence rationale>",
    "capability_debt": "<1-2 sentence rationale>",
    "knowledge_sustainability": "<1-2 sentence rationale>",
    "enterprise_adaptation": "<1-2 sentence rationale>",
    "roi": "<1-2 sentence rationale>",
    "roe": "<1-2 sentence rationale>",
    "rof": "<1-2 sentence rationale>",
    "opportunity_cost": "<1-2 sentence rationale>"
  },
  "recommendations": [
    "<actionable recommendation 1>",
    "<actionable recommendation 2>",
    "<actionable recommendation 3>"
  ]
}"""


def call_nebius(user_message: str) -> str:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("NEBIUS_API_KEY"),
    )
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )
    return response.choices[0].message.content


def extract_json(text: str) -> dict:
    text  = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response:\n{text[:300]}")
    return json.loads(text[start:end])


def score_use_case(
    use_case_name: str,
    use_case_description: str,
    rag_context: str,
    rag_sources: list,
) -> ScoringResult:
    """Score a use case using RAG-grounded LLM scoring."""
    user_message = f"""USE CASE TO SCORE:
Name: {use_case_name}
Description: {use_case_description}

RETRIEVED GOVERNANCE CONTEXT (ARM™, AVRE™, EU AI Act, NIST AI RMF):
{rag_context}

Score this use case now. Return only valid JSON."""

    raw_response = call_nebius(user_message)

    try:
        parsed = extract_json(raw_response)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Failed to parse LLM response: {e}\nRaw: {raw_response[:500]}")

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

    return ScoringResult(
        use_case_name        = use_case_name,
        use_case_description = use_case_description,
        arm                  = arm,
        avre                 = avre,
        rationale            = parsed.get("rationale", {}),
        rag_sources          = rag_sources,
        recommendations      = parsed.get("recommendations", []),
    )
