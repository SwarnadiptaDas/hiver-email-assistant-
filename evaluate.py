#!/usr/bin/env python3
"""
Hiver AI Email Response System — Evaluation Runner

Evaluates generated responses against reference responses using 7 metrics.
Usage:
    python evaluate.py                              # Generate + evaluate all
    python evaluate.py --results results/generated_responses.json  # Evaluate existing results
    python evaluate.py --count 10                   # Evaluate first 10 emails
    python evaluate.py --output evaluation_report.json  # Save report
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

from generator.retriever import EmailRetriever
from generator.responder import EmailResponder
from evaluation.evaluator import ResponseEvaluator

console = Console()


def load_results(results_path: str) -> list[dict] | None:
    """Load pre-generated results."""
    path = Path(results_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data) if isinstance(data, dict) else data


def generate_responses(dataset_path: str, count: int | None = None) -> list[dict]:
    """Generate responses for evaluation."""
    console.print("[yellow]No pre-generated results found. Generating responses first...[/yellow]\n")

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    emails = data.get("emails", [])

    if count:
        emails = emails[:count]

    # Initialize generator
    retriever = EmailRetriever(dataset_path=dataset_path)
    retriever.build_index()
    responder = EmailResponder(retriever=retriever)

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating responses...", total=len(emails))

        for i, email_data in enumerate(emails):
            email_id = email_data.get("id", f"email_{i+1:03d}")
            progress.update(task, description=f"Generating: [cyan]{email_id}[/cyan]")

            try:
                response = responder.generate_response(email_data["incoming_email"])
                results.append({
                    "email_id": email_id,
                    "category": email_data.get("category", "unknown"),
                    "incoming_email": email_data["incoming_email"],
                    "reference_response": email_data.get("reference_response", {}),
                    "generated_response": response,
                    "metadata": email_data.get("metadata", {})
                })
            except Exception as e:
                console.print(f"\n[red]Error for {email_id}: {e}[/red]")
                results.append({
                    "email_id": email_id,
                    "category": email_data.get("category", "unknown"),
                    "incoming_email": email_data["incoming_email"],
                    "reference_response": email_data.get("reference_response", {}),
                    "generated_response": {"body": f"Error: {e}"},
                    "error": str(e),
                    "metadata": email_data.get("metadata", {})
                })

            progress.update(task, advance=1)

    # Save generated results
    out_path = Path(__file__).resolve().parent / "results" / "generated_responses.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Generated responses saved to {out_path}[/green]\n")
    return results


def run_evaluation(results: list[dict], output_path: str | None = None) -> dict:
    """Run the full evaluation pipeline."""
    evaluator = ResponseEvaluator()

    console.print(Panel(
        f"Evaluating [bold]{len(results)}[/bold] generated responses\n"
        "[dim]Using 7 metrics: semantic similarity, intent coverage, tone, "
        "completeness, actionability, fluency, LLM judge[/dim]",
        title="📊 Hiver AI Evaluation System",
        border_style="magenta",
        padding=(1, 2)
    ))

    evaluated_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(results))

        for result in results:
            email_id = result.get("email_id", "unknown")
            progress.update(task, description=f"Evaluating: [cyan]{email_id}[/cyan]")

            incoming = result["incoming_email"]
            generated = result["generated_response"]
            reference = result.get("reference_response", {})

            # Extract body text
            gen_text = generated.get("body", str(generated)) if isinstance(generated, dict) else str(generated)
            ref_text = reference.get("body", str(reference)) if isinstance(reference, dict) else str(reference)
            incoming_text = f"Subject: {incoming.get('subject', '')}\n\n{incoming.get('body', '')}"

            try:
                evaluation = evaluator.evaluate_single(
                    incoming_email=incoming,
                    generated_response=gen_text,
                    reference_response=ref_text
                )
                evaluation["email_id"] = email_id
                evaluation["category"] = result.get("category", "unknown")
                evaluated_results.append(evaluation)
            except Exception as e:
                console.print(f"\n[red]Evaluation error for {email_id}: {e}[/red]")
                evaluated_results.append({
                    "email_id": email_id,
                    "category": result.get("category", "unknown"),
                    "metrics": {},
                    "composite_score": 0.0,
                    "error": str(e)
                })

            progress.update(task, advance=1)

    # Generate and print overall report
    batch_report = evaluator.evaluate_batch(evaluated_results)

    console.print("\n")
    evaluator.print_overall_report(batch_report)

    # Print detailed per-response scores
    console.print("\n")
    detail_table = Table(
        title="Per-Response Scores",
        box=box.ROUNDED,
        show_lines=True
    )
    detail_table.add_column("Email ID", style="cyan", width=12)
    detail_table.add_column("Category", style="blue", width=14)
    detail_table.add_column("Semantic", justify="center", width=9)
    detail_table.add_column("Intent", justify="center", width=9)
    detail_table.add_column("Tone", justify="center", width=9)
    detail_table.add_column("Complete", justify="center", width=9)
    detail_table.add_column("Action", justify="center", width=9)
    detail_table.add_column("Fluency", justify="center", width=9)
    detail_table.add_column("Judge", justify="center", width=9)
    detail_table.add_column("COMPOSITE", justify="center", width=10, style="bold")

    def score_color(score: float) -> str:
        if score >= 0.8:
            return "green"
        elif score >= 0.6:
            return "yellow"
        return "red"

    for ev in evaluated_results:
        if "error" in ev and not ev.get("metrics"):
            continue
        metrics = ev.get("metrics", {})

        def get_score(key):
            m = metrics.get(key, {})
            return m.get("score", 0.0) if isinstance(m, dict) else 0.0

        scores = {
            "semantic": get_score("semantic_similarity"),
            "intent": get_score("intent_coverage"),
            "tone": get_score("tone"),
            "complete": get_score("completeness"),
            "action": get_score("actionability"),
            "fluency": get_score("fluency"),
            "judge": get_score("overall_judge"),
        }
        composite = ev.get("composite_score", 0.0)

        detail_table.add_row(
            ev.get("email_id", "?"),
            ev.get("category", "?"),
            f"[{score_color(scores['semantic'])}]{scores['semantic']:.2f}[/]",
            f"[{score_color(scores['intent'])}]{scores['intent']:.2f}[/]",
            f"[{score_color(scores['tone'])}]{scores['tone']:.2f}[/]",
            f"[{score_color(scores['complete'])}]{scores['complete']:.2f}[/]",
            f"[{score_color(scores['action'])}]{scores['action']:.2f}[/]",
            f"[{score_color(scores['fluency'])}]{scores['fluency']:.2f}[/]",
            f"[{score_color(scores['judge'])}]{scores['judge']:.2f}[/]",
            f"[{score_color(composite)}]{composite:.2f}[/]",
        )

    console.print(detail_table)

    # Save full report
    if output_path:
        out_path = Path(output_path)
    else:
        out_path = Path(__file__).resolve().parent / "results" / "evaluation_report.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(batch_report, f, indent=2, ensure_ascii=False, default=str)

    console.print(f"\n[green]Full evaluation report saved to:[/green] {out_path}")

    return batch_report


def main():
    parser = argparse.ArgumentParser(
        description="Hiver AI Email Response Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=str, default=None,
                        help="Path to pre-generated results JSON")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to email dataset JSON")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of emails to evaluate (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for evaluation report")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_path = args.dataset or str(project_root / "dataset" / "email_dataset.json")

    console.print(Panel(
        "[bold magenta]Hiver AI Email Evaluation System[/bold magenta]\n"
        "[dim]Multi-dimensional response quality assessment[/dim]",
        border_style="magenta",
        padding=(1, 2)
    ))

    # Load or generate results
    if args.results:
        results = load_results(args.results)
        if results is None:
            console.print(f"[red]Results file not found: {args.results}[/red]")
            sys.exit(1)
    else:
        # Check for existing results
        default_results = project_root / "results" / "generated_responses.json"
        results = load_results(str(default_results))

        if results is None:
            results = generate_responses(dataset_path, count=args.count)

    if args.count:
        results = results[:args.count]

    console.print(f"\n[green]Loaded {len(results)} results for evaluation[/green]\n")

    # Run evaluation
    run_evaluation(results, output_path=args.output)


if __name__ == "__main__":
    main()
