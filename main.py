#!/usr/bin/env python3
"""
Hiver AI Email Response System — Main Runner

Generates suggested email responses for incoming emails using RAG + LLM.
Usage:
    python main.py                     # Generate responses for all test emails
    python main.py --count 10          # Generate for first 10 emails
    python main.py --email "Your email text here"  # Generate for a single email
    python main.py --output results.json  # Save results to file
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

console = Console()


def load_dataset(dataset_path: str) -> list[dict]:
    """Load the email dataset."""
    path = Path(dataset_path)
    if not path.exists():
        console.print(f"[red]Error:[/red] Dataset not found at {path}")
        console.print("Run: python dataset/generate_dataset.py")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("emails", [])


def generate_for_single_email(responder: EmailResponder, email_text: str) -> None:
    """Generate a response for a single ad-hoc email."""
    console.print(Panel(
        email_text,
        title="📨 Incoming Email",
        border_style="cyan",
        padding=(1, 2)
    ))

    console.print("\n[yellow]Generating response...[/yellow]\n")

    incoming = {
        "from": "user@example.com",
        "subject": "Customer Inquiry",
        "body": email_text
    }

    start = time.time()
    response = responder.generate_response(incoming)
    elapsed = time.time() - start

    console.print(Panel(
        response.get("body", response) if isinstance(response, dict) else str(response),
        title="✉️  Suggested Response",
        border_style="green",
        padding=(1, 2)
    ))
    console.print(f"\n[dim]Generated in {elapsed:.1f}s[/dim]")


def generate_for_dataset(
    responder: EmailResponder,
    emails: list[dict],
    count: int | None = None,
    output_path: str | None = None
) -> list[dict]:
    """Generate responses for emails from the dataset."""
    if count:
        emails = emails[:count]

    results = []

    console.print(Panel(
        f"Generating responses for [bold]{len(emails)}[/bold] emails",
        title="🚀 Hiver AI Email Response Generator",
        border_style="blue",
        padding=(1, 2)
    ))

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
            incoming = email_data["incoming_email"]
            email_id = email_data.get("id", f"email_{i+1:03d}")
            category = email_data.get("category", "unknown")

            progress.update(task, description=f"[cyan]{email_id}[/cyan] ({category})")

            try:
                start = time.time()
                response = responder.generate_response(incoming)
                elapsed = time.time() - start

                result = {
                    "email_id": email_id,
                    "category": category,
                    "incoming_email": incoming,
                    "reference_response": email_data.get("reference_response", {}),
                    "generated_response": response,
                    "generation_time_seconds": round(elapsed, 2),
                    "metadata": email_data.get("metadata", {})
                }
                results.append(result)

            except Exception as e:
                console.print(f"\n[red]Error generating response for {email_id}: {e}[/red]")
                results.append({
                    "email_id": email_id,
                    "category": category,
                    "incoming_email": incoming,
                    "reference_response": email_data.get("reference_response", {}),
                    "generated_response": {"subject": "Error", "body": f"Generation failed: {str(e)}"},
                    "generation_time_seconds": 0,
                    "error": str(e),
                    "metadata": email_data.get("metadata", {})
                })

            progress.update(task, advance=1)

    # Print summary
    successful = sum(1 for r in results if "error" not in r)
    avg_time = sum(r["generation_time_seconds"] for r in results if "error" not in r) / max(successful, 1)

    summary_table = Table(title="Generation Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Total Emails", str(len(emails)))
    summary_table.add_row("Successful", str(successful))
    summary_table.add_row("Failed", str(len(emails) - successful))
    summary_table.add_row("Avg Generation Time", f"{avg_time:.1f}s")
    console.print(summary_table)

    # Show a few example responses
    console.print("\n[bold]Sample Generated Responses:[/bold]\n")
    for result in results[:3]:
        if "error" in result:
            continue
        incoming = result["incoming_email"]
        generated = result["generated_response"]
        gen_body = generated.get("body", str(generated)) if isinstance(generated, dict) else str(generated)

        console.print(Panel(
            f"[dim]From: {incoming.get('from', 'N/A')}[/dim]\n"
            f"[bold]Subject: {incoming.get('subject', 'N/A')}[/bold]\n\n"
            f"{incoming.get('body', '')[:300]}{'...' if len(incoming.get('body', '')) > 300 else ''}",
            title=f"📨 {result['email_id']} ({result['category']})",
            border_style="cyan",
            padding=(1, 2)
        ))
        console.print(Panel(
            gen_body[:500] + ('...' if len(gen_body) > 500 else ''),
            title="✉️  Generated Response",
            border_style="green",
            padding=(1, 2)
        ))
        console.print()

    # Save results
    if output_path:
        out_path = Path(output_path)
    else:
        out_path = Path(__file__).resolve().parent / "results" / "generated_responses.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": {
            "total": len(emails),
            "successful": successful,
            "failed": len(emails) - successful,
            "avg_generation_time": round(avg_time, 2)
        }}, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Results saved to:[/green] {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hiver AI Email Response Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Generate for all dataset emails
  python main.py --count 5                # Generate for first 5 emails
  python main.py --email "Hi, I need help with billing"
  python main.py --output my_results.json
        """
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to email dataset JSON")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of emails to process (default: all)")
    parser.add_argument("--email", type=str, default=None,
                        help="Single email text to generate a response for")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")

    args = parser.parse_args()

    console.print(Panel(
        "[bold blue]Hiver AI Email Response System[/bold blue]\n"
        "[dim]RAG-powered email reply suggestions using Groq[/dim]",
        border_style="blue",
        padding=(1, 2)
    ))

    # Resolve dataset path
    project_root = Path(__file__).resolve().parent
    dataset_path = args.dataset or str(project_root / "dataset" / "email_dataset.json")

    # Initialize components
    console.print("\n[yellow]Initializing retriever...[/yellow]")
    retriever = EmailRetriever(dataset_path=dataset_path)
    retriever.build_index()
    console.print("[green]OK: Retriever ready[/green]")
    
    console.print("[yellow]Initializing responder...[/yellow]")
    responder = EmailResponder(retriever=retriever)
    console.print("[green]OK: Responder ready[/green]\n")

    if args.email:
        generate_for_single_email(responder, args.email)
    else:
        emails = load_dataset(dataset_path)
        generate_for_dataset(responder, emails, count=args.count, output_path=args.output)


if __name__ == "__main__":
    main()
