import os
from typing import List, Dict, Any
from pathlib import Path
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from evaluation.metrics import EmbeddingMetrics, LLMMetrics

class ResponseEvaluator:
    METRIC_WEIGHTS = {
        'semantic_similarity': 0.15,
        'intent_coverage': 0.25,
        'tone': 0.15,
        'completeness': 0.20,
        'actionability': 0.10,
        'fluency': 0.05,
        'overall_judge': 0.10
    }
    
    def __init__(self):
        self.embedding_metrics = EmbeddingMetrics()
        self.llm_metrics = LLMMetrics()
        self.console = Console()
        
    def evaluate_single(self, incoming_email: dict, generated_response: str, reference_response: str) -> dict:
        email_text = incoming_email.get('body', incoming_email.get('text', ''))
        
        results = {
            'email_id': incoming_email.get('id', 'unknown_id'),
            'email_subject': incoming_email.get('subject', 'No Subject'),
            'category': incoming_email.get('category', 'general'),
            'metrics': {}
        }
        
        # 1. Semantic Similarity
        sim_score = self.embedding_metrics.semantic_similarity(generated_response, reference_response)
        results['metrics']['semantic_similarity'] = {'score': sim_score}
        
        # 2. Intent Coverage
        results['metrics']['intent_coverage'] = self.llm_metrics.intent_coverage(email_text, generated_response)
        
        # 3. Tone Score
        results['metrics']['tone'] = self.llm_metrics.tone_score(email_text, generated_response)
        
        # 4. Completeness
        results['metrics']['completeness'] = self.llm_metrics.completeness_score(email_text, generated_response, reference_response)
        
        # 5. Actionability
        results['metrics']['actionability'] = self.llm_metrics.actionability_score(generated_response)
        
        # 6. Fluency
        results['metrics']['fluency'] = self.llm_metrics.fluency_score(generated_response)
        
        # 7. Overall Judge
        results['metrics']['overall_judge'] = self.llm_metrics.overall_judge(email_text, generated_response, reference_response)
        
        # Calculate composite score
        composite = 0.0
        for metric, weight in self.METRIC_WEIGHTS.items():
            score = results['metrics'][metric].get('score', 0.0)
            composite += score * weight
            
        results['composite_score'] = composite
        results['weights_used'] = self.METRIC_WEIGHTS.copy()
        
        return results

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> dict:
        """Aggregate pre-evaluated individual results into an overall report.
        
        Args:
            results: List of dicts, each already containing 'metrics', 'composite_score',
                     'category', 'email_id' — as returned by evaluate_single().
        """
        if not results:
            return {}
            
        # Filter out results with errors and no metrics
        individual_results = [r for r in results if r.get('metrics')]
        
        if not individual_results:
            return {'total_evaluated': 0, 'overall_composite': {}, 'per_metric_stats': {}, 'per_category_stats': {}, 'individual_results': []}
        
        composites = [r['composite_score'] for r in individual_results]
        
        # Per metric stats
        per_metric_stats = {}
        for metric in self.METRIC_WEIGHTS.keys():
            scores = [r['metrics'][metric].get('score', 0.0) for r in individual_results]
            per_metric_stats[metric] = {
                'mean': statistics.mean(scores) if scores else 0.0,
                'median': statistics.median(scores) if scores else 0.0,
                'std': statistics.stdev(scores) if len(scores) > 1 else 0.0
            }
            
        # Per category stats
        category_scores = {}
        for r in individual_results:
            cat = r.get('category', 'general')
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(r['composite_score'])
            
        per_category_stats = {}
        for cat, scores in category_scores.items():
            per_category_stats[cat] = {
                'mean': statistics.mean(scores),
                'count': len(scores)
            }
            
        return {
            'total_evaluated': len(results),
            'overall_composite': {
                'mean': statistics.mean(composites),
                'median': statistics.median(composites),
                'std': statistics.stdev(composites) if len(composites) > 1 else 0.0,
                'min': min(composites),
                'max': max(composites)
            },
            'per_metric_stats': per_metric_stats,
            'per_category_stats': per_category_stats,
            'individual_results': individual_results
        }
        
    def _format_score(self, score: float) -> str:
        color = "red"
        if score >= 0.8:
            color = "green"
        elif score >= 0.6:
            color = "yellow"
        return f"[{color}]{score:.2f}[/{color}]"

    def print_single_report(self, evaluation: dict) -> None:
        table = Table(title=f"Evaluation Report: {evaluation['email_id']}")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Score", justify="right")
        table.add_column("Reasoning/Details")
        
        metrics = evaluation.get('metrics', {})
        
        # Add rows for each metric
        for metric_name, details in metrics.items():
            score_str = self._format_score(details.get('score', 0.0))
            
            # Build reasoning text
            reasoning = details.get('reasoning', '')
            if metric_name == 'intent_coverage':
                reasoning += f"\nFound: {len(details.get('intents_found', []))} | Addressed: {len(details.get('intents_addressed', []))}"
            elif metric_name == 'completeness' and details.get('missing_points'):
                reasoning += f"\nMissing: {', '.join(details.get('missing_points', []))}"
            elif metric_name == 'actionability' and details.get('actions_found'):
                reasoning += f"\nActions: {', '.join(details.get('actions_found', []))}"
                
            table.add_row(metric_name.replace('_', ' ').title(), score_str, reasoning)
            
        self.console.print(table)
        
        composite_score = evaluation.get('composite_score', 0.0)
        self.console.print(Panel(f"Composite Score: {self._format_score(composite_score)}", title="Overall Result", border_style="blue"))

    def print_overall_report(self, batch_evaluation: dict) -> None:
        if not batch_evaluation:
            self.console.print("[red]No batch evaluation data to display.[/red]")
            return
            
        overall_stats = batch_evaluation.get('overall_composite', {})
        
        summary_text = (
            f"Total Evaluated: {batch_evaluation.get('total_evaluated', 0)}\n"
            f"Mean Score: {self._format_score(overall_stats.get('mean', 0.0))}\n"
            f"Median Score: {self._format_score(overall_stats.get('median', 0.0))}\n"
            f"Std Dev: {overall_stats.get('std', 0.0):.2f}\n"
            f"Range: {self._format_score(overall_stats.get('min', 0.0))} - {self._format_score(overall_stats.get('max', 0.0))}"
        )
        self.console.print(Panel(summary_text, title="Batch Summary", border_style="magenta"))
        
        # Metric Table
        metric_table = Table(title="Per-Metric Statistics")
        metric_table.add_column("Metric", style="cyan")
        metric_table.add_column("Mean", justify="right")
        metric_table.add_column("Median", justify="right")
        
        for metric, stats in batch_evaluation.get('per_metric_stats', {}).items():
            metric_table.add_row(
                metric.replace('_', ' ').title(),
                self._format_score(stats.get('mean', 0.0)),
                self._format_score(stats.get('median', 0.0))
            )
            
        self.console.print(metric_table)
        
        # Category Table
        cat_table = Table(title="Category Statistics")
        cat_table.add_column("Category", style="green")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Mean Score", justify="right")
        
        for cat, stats in batch_evaluation.get('per_category_stats', {}).items():
            cat_table.add_row(
                cat.title(),
                str(stats.get('count', 0)),
                self._format_score(stats.get('mean', 0.0))
            )
            
        self.console.print(cat_table)
