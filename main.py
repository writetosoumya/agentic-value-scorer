"""
main.py
Agentification Value Scorer — Main Entry Point
Week 2 RAG Project | The Gen Academy Mastering Agentic AI Bootcamp
Author: Agni Sivaraman | Enterprise AI Governance

Usage:
  python main.py                          # Interactive mode
  python main.py --eval                   # Run full 15-question evaluation suite
  python main.py --score "use case desc"  # Score a single use case
  python main.py --rebuild                # Rebuild vector store from corpus
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_engine import build_vectorstore, get_retriever, retrieve_context
from llm_scorer import score_use_case
from evaluator import run_evaluation_suite

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║       AGENTIFICATION VALUE SCORER                                ║
║       ARM™ × AVRE™ Intelligence Engine                           ║
║       Powered by RAG + LangChain + ChromaDB                      ║
║       Week 2 Project | The Gen Academy Mastering Agentic AI      ║
╚══════════════════════════════════════════════════════════════════╝
"""

EXAMPLE_USE_CASES = [
    {
        "name": "HR Resume Screening Agent",
        "description": "An AI agent that automatically screens and ranks resumes against job requirements, scoring candidates and prioritizing the HR review queue. Deployed in Workday, operating across 50+ open requisitions monthly."
    },
    {
        "name": "Enterprise Policy Q&A Bot",
        "description": "A RAG-powered chatbot that answers employee questions about HR, IT, legal, and compliance policies from SharePoint. Deployed in Microsoft Teams for 8,000 employees. Read-only, surfaces answers with citations."
    },
    {
        "name": "Fraud Detection Agent",
        "description": "Real-time transaction monitoring agent that flags suspicious activity and autonomously initiates account holds for transactions above $10,000 showing anomaly signals. Operates 24/7 in a retail banking context."
    },
    {
        "name": "Contract Review Agent",
        "description": "An agent that extracts key terms from vendor contracts, flags non-standard clauses against a legal playbook, and generates a summary for attorney review. The attorney makes all final decisions."
    },
    {
        "name": "IT Service Management Triage Agent",
        "description": "Agent that classifies incoming IT support tickets, auto-resolves low-complexity requests (password resets, VPN access), and routes complex issues to the right team with a suggested resolution path."
    },
    {
        "name": "Project Portfolio Status Agent",
        "description": "Agent that aggregates status from Jira, ServiceNow, and Smartsheet, identifies at-risk projects using earned value metrics, and generates an executive PMO briefing every Monday morning."
    },
]


def run_interactive(retriever):
    """Interactive mode: user enters a use case and gets scored."""
    console.print(Panel(
        "[bold]Enter a business use case to score for agentification readiness.[/bold]\n"
        "The system will assess ARM™ risk dimensions and AVRE™ value lenses\n"
        "grounded in EU AI Act, NIST AI RMF, and proprietary governance frameworks.\n\n"
        "[dim]Type 'examples' to see sample use cases, 'quit' to exit.[/dim]",
        border_style="cyan"
    ))

    while True:
        console.print()
        use_case_name = Prompt.ask("[bold cyan]Use Case Name[/bold cyan]")

        if use_case_name.lower() == "quit":
            console.print("Goodbye!")
            break

        if use_case_name.lower() == "examples":
            for i, ex in enumerate(EXAMPLE_USE_CASES, 1):
                console.print(f"[bold]{i}.[/bold] {ex['name']}")
                console.print(f"   [dim]{ex['description'][:100]}...[/dim]")
            continue

        use_case_desc = Prompt.ask("[bold cyan]Use Case Description[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Retrieving governance context from RAG...", total=None)
            rag_result = retrieve_context(
                f"ARM risk scoring AVRE value assessment: {use_case_name} {use_case_desc}",
                retriever
            )
            progress.update(task, description="Scoring with ARM™ and AVRE™ frameworks...")
            result = score_use_case(
                use_case_name=use_case_name,
                use_case_description=use_case_desc,
                rag_context=rag_result["context"],
                rag_sources=rag_result["sources"],
            )

        result.pretty_print()

        # Save output
        output_file = Path(__file__).parent / "outputs" / f"{use_case_name.replace(' ', '_')[:40]}_score.json"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"[dim]Score saved to {output_file}[/dim]")

        if not Confirm.ask("Score another use case?"):
            break


def score_single(use_case_desc: str, retriever):
    """Score a single use case passed via CLI."""
    name = use_case_desc[:60] + "..." if len(use_case_desc) > 60 else use_case_desc
    rag_result = retrieve_context(f"ARM AVRE scoring: {use_case_desc}", retriever)
    result = score_use_case(
        use_case_name=name,
        use_case_description=use_case_desc,
        rag_context=rag_result["context"],
        rag_sources=rag_result["sources"],
    )
    result.pretty_print()
    return result


def main():
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")

    parser = argparse.ArgumentParser(description="Agentification Value Scorer")
    parser.add_argument("--eval", action="store_true", help="Run 15-question evaluation suite")
    parser.add_argument("--score", type=str, help="Score a single use case description")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of vector store")
    args = parser.parse_args()

    # Build/load RAG
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        t = p.add_task("Loading governance knowledge base...", total=None)
        vectorstore = build_vectorstore(force_rebuild=args.rebuild)
        retriever = get_retriever(vectorstore)
        p.update(t, description="✅ Knowledge base ready")

    if args.eval:
        run_evaluation_suite(retriever)
    elif args.score:
        score_single(args.score, retriever)
    else:
        run_interactive(retriever)


if __name__ == "__main__":
    main()
