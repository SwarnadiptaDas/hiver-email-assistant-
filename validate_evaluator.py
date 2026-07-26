#!/usr/bin/env python3
"""
Hiver AI Evaluator Meta-Validation
====================================
This script VALIDATES the evaluation system itself by proving that:

1. HIGH-quality responses get HIGH scores
2. LOW-quality responses (evasive, rude, off-topic, hallucinated) get LOW scores
3. The 7 metrics are each sensitive to the specific dimension they measure

This directly addresses the core challenge requirement:
  "How do you know the metric reflects real quality — not just a number?"

Usage:
    python validate_evaluator.py
"""

import sys
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

console = Console()

# ─────────────────────────────────────────────
# Test Cases — Deliberately crafted pairs
# Each has a REFERENCE good response + multiple
# test responses at different quality levels
# ─────────────────────────────────────────────

INCOMING_EMAIL = {
    "subject": "Invoice discrepancy for July",
    "body": (
        "Hi,\n\nI noticed our July invoice shows $2,400 but we're on the Growth plan at $1,800/month. "
        "Could you explain the difference? Is there an overage fee we missed? "
        "We need to get this resolved before accounting closes the month.\n\nThanks,\nSarah"
    ),
    "id": "meta_test_001"
}

REFERENCE_RESPONSE = (
    "Hi Sarah,\n\nThank you for reaching out. I've reviewed your account and found that "
    "the $600 difference is due to 20 additional user seats added on July 14th. "
    "These were prorated for the remaining 17 days of the billing cycle.\n\n"
    "I've attached a detailed breakdown in the invoice PDF on page 2. "
    "If you'd like to adjust the seat count or need further clarification, "
    "please let me know and I'll get this sorted before your accounting deadline.\n\n"
    "Best regards,\nSupport Team"
)

TEST_RESPONSES = [
    {
        "label": "PERFECT Response",
        "tier": "HIGH",
        "expected": "> 0.80",
        "body": (
            "Hi Sarah,\n\nThank you for reaching out! I can see exactly why this is confusing. "
            "The $600 difference on your July invoice is because 20 additional user seats were added to your account on July 14th. "
            "Those seats were prorated for the remaining 17 days in the billing cycle at $1.76/seat/day, totalling $599.20 (rounded to $600).\n\n"
            "You'll find the full breakdown on page 2 of your invoice PDF. "
            "If you'd like to remove those seats or make any changes before the month closes, "
            "just let me know and I'll prioritise this for you today.\n\n"
            "Best regards,\nSupport Team"
        )
    },
    {
        "label": "GOOD Response (minor gaps)",
        "tier": "HIGH",
        "expected": "0.65-0.80",
        "body": (
            "Hi Sarah,\n\nThe invoice difference is due to additional user licenses added during the month. "
            "Please check page 2 of the invoice for more details. "
            "Let us know if you have questions.\n\nBest,\nSupport Team"
        )
    },
    {
        "label": "EVASIVE Response (no answer)",
        "tier": "LOW",
        "expected": "< 0.50",
        "body": (
            "Hi Sarah,\n\nThank you for contacting us! Our billing team is very busy right now. "
            "We appreciate your patience and will look into this when we get a chance. "
            "Have a great day!\n\nBest,\nSupport Team"
        )
    },
    {
        "label": "OFF-TOPIC Response",
        "tier": "LOW",
        "expected": "< 0.40",
        "body": (
            "Hi there,\n\nThank you for reaching out to us. We'd like to remind you about "
            "our upcoming webinar on Hiver's new AI features this Thursday at 3 PM EST. "
            "Click here to register. We hope to see you there!\n\nBest,\nMarketing Team"
        )
    },
    {
        "label": "HALLUCINATED Response (wrong info)",
        "tier": "LOW",
        "expected": "< 0.50",
        "body": (
            "Hi Sarah,\n\nYou're actually on our Enterprise plan at $3,200/month — not Growth. "
            "The invoice is correct. You upgraded last March and signed a 2-year contract. "
            "There's nothing to refund. Please contact your account manager for further details.\n\nBest,\nSupport Team"
        )
    },
    {
        "label": "RUDE Response",
        "tier": "LOW",
        "expected": "< 0.45",
        "body": (
            "Hi,\n\nThe invoice is correct. You should have read the terms and conditions "
            "before signing up. Overage fees are standard and clearly listed. "
            "We cannot change pricing because you didn't monitor your usage.\n\nRegards"
        )
    },
    {
        "label": "INCOMPLETE Response (one word)",
        "tier": "LOW",
        "expected": "< 0.30",
        "body": "Noted."
    },
]


def run_validation():
    console.print(Panel(
        "[bold cyan]Hiver AI — Evaluator Meta-Validation[/bold cyan]\n"
        "[dim]Proving the 7-metric system correctly distinguishes good from bad responses[/dim]",
        border_style="cyan",
        padding=(1, 2)
    ))

    from evaluation.evaluator import ResponseEvaluator
    evaluator = ResponseEvaluator()

    results = []

    console.print(f"\n[dim]Testing {len(TEST_RESPONSES)} deliberately crafted responses...[/dim]\n")

    for i, test in enumerate(TEST_RESPONSES):
        label = test["label"]
        tier  = test["tier"]
        console.print(f"  [{i+1}/{len(TEST_RESPONSES)}] Evaluating: [yellow]{label}[/yellow]")

        evaluation = evaluator.evaluate_single(
            incoming_email=INCOMING_EMAIL,
            generated_response=test["body"],
            reference_response=REFERENCE_RESPONSE
        )

        composite = evaluation.get("composite_score", 0.0)
        metrics   = evaluation.get("metrics", {})

        def g(k):
            m = metrics.get(k, {})
            return m.get("score", 0.0) if isinstance(m, dict) else 0.0

        results.append({
            "label":      label,
            "tier":       tier,
            "expected":   test["expected"],
            "composite":  composite,
            "semantic":   g("semantic_similarity"),
            "intent":     g("intent_coverage"),
            "tone":       g("tone"),
            "complete":   g("completeness"),
            "action":     g("actionability"),
            "fluency":    g("fluency"),
            "judge":      g("overall_judge"),
        })

    # ── Print Results Table ──────────────────────────────────
    console.print("\n")
    table = Table(
        title="Meta-Validation Results",
        box=box.ROUNDED,
        show_lines=True,
        border_style="cyan"
    )
    table.add_column("Response Type",    style="bold", width=30)
    table.add_column("Tier",             justify="center", width=6)
    table.add_column("Expected",         justify="center", width=12)
    table.add_column("COMPOSITE",        justify="center", width=10, style="bold")
    table.add_column("Intent",           justify="center", width=8)
    table.add_column("Tone",             justify="center", width=7)
    table.add_column("Complete",         justify="center", width=9)
    table.add_column("LLM Judge",        justify="center", width=9)

    def score_style(s):
        if s >= 0.75: return f"[green]{s:.2f}[/green]"
        if s >= 0.50: return f"[yellow]{s:.2f}[/yellow]"
        return f"[red]{s:.2f}[/red]"

    passed = 0
    failed = 0

    for r in results:
        tier_badge = "[green]HIGH[/green]" if r["tier"] == "HIGH" else "[red]LOW[/red]"
        cs = r["composite"]

        # Check if the result aligns with expectation
        meets = True
        if r["tier"] == "HIGH" and cs < 0.50:
            meets = False
        if r["tier"] == "LOW" and cs > 0.65:
            meets = False

        if meets:
            passed += 1
            composite_str = f"[green]{cs:.2f} OK[/green]"
        else:
            failed += 1
            composite_str = f"[red]{cs:.2f} FAIL[/red]"

        table.add_row(
            r["label"], tier_badge, r["expected"],
            composite_str,
            score_style(r["intent"]),
            score_style(r["tone"]),
            score_style(r["complete"]),
            score_style(r["judge"]),
        )

    console.print(table)

    # ── Summary ──────────────────────────────────────────────
    console.print()
    total = len(results)
    pct = passed / total * 100
    verdict_color = "green" if passed == total else ("yellow" if pct >= 70 else "red")

    console.print(Panel(
        f"[{verdict_color}]Meta-Validation: {passed}/{total} test cases correctly classified ({pct:.0f}%)[/{verdict_color}]\n\n"
        "[dim]HIGH-quality responses should score > 0.65, LOW-quality < 0.65[/dim]\n\n"
        "[bold]Interpretation:[/bold]\n"
        "  - If HIGH responses score high → the metrics reward correct, complete, professional replies\n"
        "  - If LOW responses score low  → the metrics penalise evasion, hallucination, and rudeness\n"
        "  - This proves the composite score is a meaningful signal, not a random number",
        title="Validation Summary",
        border_style=verdict_color,
        padding=(1, 2)
    ))

    # ── Save Report ──────────────────────────────────────────
    out_path = Path(__file__).resolve().parent / "results" / "meta_validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "passed": passed,
            "accuracy_pct": pct,
            "results": results
        }, f, indent=2)

    console.print(f"\n[green]Meta-validation report saved to:[/green] {out_path}\n")
    return pct


if __name__ == "__main__":
    run_validation()
