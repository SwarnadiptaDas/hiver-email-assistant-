#!/usr/bin/env python3
"""
Hiver AI Email Response System — Interactive Demo

An interactive demo that lets you type an email and get a suggested response.
Usage:
    python demo.py
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

from generator.retriever import EmailRetriever
from generator.responder import EmailResponder
from evaluation.evaluator import ResponseEvaluator

console = Console()


def main():
    console.print(Panel(
        "[bold blue]Hiver AI Email Response System[/bold blue]\n"
        "[dim]Interactive Demo — Type an email, get a suggested response + quality score[/dim]\n\n"
        "[yellow]Type 'quit' or 'exit' to stop. Type 'example' for a sample email.[/yellow]",
        border_style="blue",
        padding=(1, 2)
    ))

    project_root = Path(__file__).resolve().parent
    dataset_path = str(project_root / "dataset" / "email_dataset.json")

    # Initialize components
    console.print("\n[yellow]Loading system components...[/yellow]")

    console.print("  [dim]→ Building retrieval index...[/dim]")
    retriever = EmailRetriever(dataset_path=dataset_path)
    retriever.build_index()
    console.print("  [green]OK: Retriever ready[/green]")

    console.print("  [dim]→ Initializing response generator...[/dim]")
    responder = EmailResponder(retriever=retriever)
    console.print("  [green]OK: Responder ready[/green]")

    console.print("  [dim]→ Initializing evaluator...[/dim]")
    evaluator = ResponseEvaluator()
    console.print("  [green]OK: Evaluator ready[/green]\n")

    EXAMPLE_EMAIL = (
        "Hi,\n\n"
        "I've been trying to set up the integration with our Slack workspace for the past "
        "two hours but keep getting an 'Authentication Failed' error. I've double-checked "
        "my API credentials and they seem correct.\n\n"
        "We need this working before our team standup tomorrow morning. Can someone help "
        "me troubleshoot this urgently?\n\n"
        "Also, is there a way to customize the notification settings once the integration "
        "is working?\n\n"
        "Thanks,\nMichael"
    )

    while True:
        console.print("\n" + "─" * 60)
        email_input = Prompt.ask(
            "\n[bold cyan]Enter incoming email[/bold cyan] (or 'example'/'quit')"
        )

        if email_input.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if email_input.lower() == "example":
            email_input = EXAMPLE_EMAIL
            console.print(Panel(email_input, title="📨 Example Email", border_style="cyan"))

        if not email_input.strip():
            console.print("[red]Please enter an email.[/red]")
            continue

        # Generate response
        console.print("\n[yellow]🔍 Retrieving similar past emails...[/yellow]")
        incoming = {
            "from": "user@example.com",
            "subject": "Customer Inquiry",
            "body": email_input
        }

        similar = retriever.retrieve(
            f"{incoming['subject']} {incoming['body']}", top_k=3
        )
        if similar:
            console.print(f"[dim]  Found {len(similar)} similar past conversations[/dim]")

        console.print("[yellow]✍️  Generating response...[/yellow]")
        response = responder.generate_response(incoming)

        gen_body = response.get("body", str(response)) if isinstance(response, dict) else str(response)

        console.print(Panel(
            gen_body,
            title="✉️  Suggested Response",
            border_style="green",
            padding=(1, 2)
        ))

        # Quick evaluation (semantic similarity only, skip full LLM eval for speed)
        console.print("\n[yellow]📊 Quick quality check...[/yellow]")
        try:
            from evaluation.metrics import EmbeddingMetrics
            embedding_metrics = EmbeddingMetrics()

            # Compare against the best retrieved reference
            if similar:
                best_ref = similar[0].get("reference_response", {})
                ref_body = best_ref.get("body", "") if isinstance(best_ref, dict) else str(best_ref)
                sim_score = embedding_metrics.semantic_similarity(gen_body, ref_body)
                console.print(f"  Semantic similarity to best match: [{_score_color(sim_score)}]{sim_score:.2f}[/]")
        except Exception as e:
            console.print(f"[dim]  (Quick eval skipped: {e})[/dim]")

        console.print("\n[dim]Tip: Run 'python evaluate.py' for full multi-metric evaluation[/dim]")


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "yellow"
    return "red"


if __name__ == "__main__":
    main()
