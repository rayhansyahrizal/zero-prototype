"""
TTA Analysis and Visualization Tools.

Provides utilities for:
1. Tracking TTA convergence
2. Computing delta metrics
3. Visualizing TTA effectiveness
4. Per-sample TTA impact analysis
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TTAMetrics:
    """Container for TTA adaptation metrics."""
    image_id: str
    step: int
    loss_total: float
    loss_similarity: float
    loss_variance: float
    loss_entropy: float
    embedding_change: float  # L2 norm of change from original
    top_k_similarities: List[float]  # Similarities with top-k prototypes


class TTAAnalyzer:
    """Analyzer for TTA convergence and effectiveness."""

    def __init__(self, save_dir: Path):
        """
        Initialize TTA analyzer.

        Args:
            save_dir: Directory to save analysis results and plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []  # List of TTAMetrics

        logger.info(f"TTAAnalyzer initialized, saving to {self.save_dir}")

    def record_step(self, metrics: TTAMetrics):
        """Record metrics from a single TTA step."""
        self.metrics_history.append(metrics)

    def record_batch(self, metrics_list: List[TTAMetrics]):
        """Record metrics from multiple samples."""
        self.metrics_history.extend(metrics_list)

    def save_metrics(self, filename: str = "tta_metrics.json"):
        """Save all recorded metrics to JSON."""
        metrics_dicts = [asdict(m) for m in self.metrics_history]

        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics_dicts, f, indent=2)

        logger.info(f"Saved {len(metrics_dicts)} TTA metrics to {save_path}")

    def load_metrics(self, filename: str = "tta_metrics.json"):
        """Load metrics from JSON."""
        load_path = self.save_dir / filename

        if not load_path.exists():
            logger.warning(f"Metrics file not found: {load_path}")
            return

        with open(load_path) as f:
            metrics_dicts = json.load(f)

        self.metrics_history = [
            TTAMetrics(**m) for m in metrics_dicts
        ]

        logger.info(f"Loaded {len(self.metrics_history)} TTA metrics from {load_path}")

    def plot_convergence(
        self,
        image_id: Optional[str] = None,
        metric: str = 'loss_total',
        save_name: str = "convergence.png"
    ):
        """
        Plot TTA convergence curve.

        Args:
            image_id: If specified, plot for single image; else plot average
            metric: Which metric to plot (loss_total, loss_similarity, etc.)
            save_name: Filename for saving plot
        """
        if not self.metrics_history:
            logger.warning("No metrics to plot")
            return

        plt.figure(figsize=(10, 6))

        if image_id:
            # Plot for single image
            image_metrics = [m for m in self.metrics_history if m.image_id == image_id]

            if not image_metrics:
                logger.warning(f"No metrics found for image {image_id}")
                return

            steps = [m.step for m in image_metrics]
            values = [getattr(m, metric) for m in image_metrics]

            plt.plot(steps, values, marker='o', linewidth=2, label=image_id)
            plt.title(f"TTA Convergence: {metric} ({image_id})")

        else:
            # Plot average across all images
            # Group by step
            steps_dict = {}
            for m in self.metrics_history:
                if m.step not in steps_dict:
                    steps_dict[m.step] = []
                steps_dict[m.step].append(getattr(m, metric))

            steps = sorted(steps_dict.keys())
            means = [np.mean(steps_dict[s]) for s in steps]
            stds = [np.std(steps_dict[s]) for s in steps]

            plt.plot(steps, means, marker='o', linewidth=2, label='Mean')
            plt.fill_between(
                steps,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.3,
                label='±1 std'
            )
            plt.title(f"TTA Convergence: {metric} (Average across all images)")

        plt.xlabel("Adaptation Step")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved convergence plot to {save_path}")

    def plot_all_convergence_metrics(self, save_prefix: str = "convergence"):
        """Plot convergence for all tracked metrics."""
        metrics_to_plot = [
            'loss_total',
            'loss_similarity',
            'loss_variance',
            'loss_entropy',
            'embedding_change'
        ]

        for metric in metrics_to_plot:
            try:
                self.plot_convergence(
                    metric=metric,
                    save_name=f"{save_prefix}_{metric}.png"
                )
            except Exception as e:
                logger.warning(f"Failed to plot {metric}: {e}")

    def compute_per_sample_delta(
        self,
        pre_tta_scores: Dict[str, float],
        post_tta_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Compute per-sample delta between pre-TTA and post-TTA.

        Args:
            pre_tta_scores: Dict mapping image_id to pre-TTA scores
            post_tta_scores: Dict mapping image_id to post-TTA scores

        Returns:
            DataFrame with per-sample delta analysis
        """
        records = []

        for image_id in pre_tta_scores.keys():
            if image_id not in post_tta_scores:
                continue

            pre = pre_tta_scores[image_id]
            post = post_tta_scores[image_id]

            delta = post - pre
            delta_pct = (delta / pre * 100) if pre != 0 else 0

            records.append({
                'image_id': image_id,
                'pre_tta_score': pre,
                'post_tta_score': post,
                'delta': delta,
                'delta_pct': delta_pct,
                'improved': delta > 0
            })

        df = pd.DataFrame(records)

        # Summary statistics
        logger.info("\n" + "=" * 60)
        logger.info("PER-SAMPLE DELTA ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Improved: {df['improved'].sum()} ({df['improved'].mean()*100:.1f}%)")
        logger.info(f"Degraded: {(~df['improved']).sum()} ({(~df['improved']).mean()*100:.1f}%)")
        logger.info(f"Mean delta: {df['delta'].mean():.4f}")
        logger.info(f"Median delta: {df['delta'].median():.4f}")
        logger.info(f"Std delta: {df['delta'].std():.4f}")

        # Save to CSV
        csv_path = self.save_dir / "per_sample_delta.csv"
        df.to_csv(csv_path, index=False, float_format='%.6f')
        logger.info(f"\n💾 Per-sample delta saved to {csv_path}")

        return df

    def plot_delta_distribution(
        self,
        delta_df: pd.DataFrame,
        metric_name: str = "BLEU-4",
        save_name: str = "delta_distribution.png"
    ):
        """
        Plot distribution of delta scores.

        Args:
            delta_df: DataFrame from compute_per_sample_delta
            metric_name: Name of metric for labeling
            save_name: Filename for saving plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax = axes[0]
        ax.hist(delta_df['delta'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
        ax.axvline(delta_df['delta'].mean(), color='green', linestyle='--',
                   linewidth=2, label=f'Mean: {delta_df["delta"].mean():.4f}')
        ax.set_xlabel(f'Δ {metric_name} (POST-TTA - PRE-TTA)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of TTA Impact on {metric_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Box plot
        ax = axes[1]
        improved = delta_df[delta_df['improved']]['delta']
        degraded = delta_df[~delta_df['improved']]['delta']

        box_data = [improved, degraded]
        ax.boxplot(box_data, labels=['Improved', 'Degraded'])
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_ylabel(f'Δ {metric_name}')
        ax.set_title(f'TTA Impact by Outcome')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved delta distribution plot to {save_path}")

    def plot_performance_vs_adaptation(
        self,
        delta_df: pd.DataFrame,
        embedding_changes: Dict[str, float],
        save_name: str = "performance_vs_adaptation.png"
    ):
        """
        Plot relationship between embedding change and performance delta.

        Args:
            delta_df: DataFrame from compute_per_sample_delta
            embedding_changes: Dict mapping image_id to final embedding L2 change
            save_name: Filename for saving plot
        """
        # Merge embedding changes with delta
        delta_df = delta_df.copy()
        delta_df['embedding_change'] = delta_df['image_id'].map(embedding_changes)

        # Remove samples without embedding change data
        delta_df = delta_df.dropna(subset=['embedding_change'])

        if len(delta_df) == 0:
            logger.warning("No samples with embedding change data")
            return

        plt.figure(figsize=(10, 6))

        # Scatter plot
        colors = delta_df['improved'].map({True: 'green', False: 'red'})
        plt.scatter(
            delta_df['embedding_change'],
            delta_df['delta'],
            c=colors,
            alpha=0.6,
            s=50
        )

        plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.xlabel('Embedding Change (L2 Norm)')
        plt.ylabel('Performance Delta')
        plt.title('Performance Delta vs. Embedding Adaptation')
        plt.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.6, label='Improved'),
            Patch(facecolor='red', alpha=0.6, label='Degraded')
        ]
        plt.legend(handles=legend_elements)

        plt.tight_layout()

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved performance vs adaptation plot to {save_path}")

    def generate_full_report(
        self,
        pre_tta_results: Dict,
        post_tta_results: Dict,
        metric_key: str = 'bleu_4'
    ):
        """
        Generate comprehensive TTA analysis report.

        Args:
            pre_tta_results: Detailed scores before TTA
            post_tta_results: Detailed scores after TTA
            metric_key: Which metric to analyze
        """
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING COMPREHENSIVE TTA ANALYSIS REPORT")
        logger.info("=" * 60)

        # Extract scores for specified metric
        pre_scores = {
            sample['image_id']: sample[metric_key]
            for sample in pre_tta_results
            if metric_key in sample
        }

        post_scores = {
            sample['image_id']: sample[metric_key]
            for sample in post_tta_results
            if metric_key in sample
        }

        # Compute per-sample delta
        delta_df = self.compute_per_sample_delta(pre_scores, post_scores)

        # Plot delta distribution
        self.plot_delta_distribution(
            delta_df,
            metric_name=metric_key.upper().replace('_', '-'),
            save_name=f"delta_dist_{metric_key}.png"
        )

        # Plot convergence if we have metrics history
        if self.metrics_history:
            self.plot_all_convergence_metrics(save_prefix=f"convergence_{metric_key}")

            # Extract final embedding changes
            final_changes = {}
            for image_id in set(m.image_id for m in self.metrics_history):
                image_metrics = [m for m in self.metrics_history if m.image_id == image_id]
                if image_metrics:
                    final_changes[image_id] = image_metrics[-1].embedding_change

            # Plot performance vs adaptation
            self.plot_performance_vs_adaptation(
                delta_df,
                final_changes,
                save_name=f"perf_vs_adapt_{metric_key}.png"
            )

        # Save metrics
        self.save_metrics(filename=f"tta_metrics_{metric_key}.json")

        # Generate summary report
        report_path = self.save_dir / f"tta_report_{metric_key}.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TTA ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Metric: {metric_key.upper()}\n")
            f.write(f"Total samples: {len(delta_df)}\n\n")

            f.write("PERFORMANCE CHANGE:\n")
            f.write(f"  Improved samples: {delta_df['improved'].sum()} ({delta_df['improved'].mean()*100:.1f}%)\n")
            f.write(f"  Degraded samples: {(~delta_df['improved']).sum()} ({(~delta_df['improved']).mean()*100:.1f}%)\n\n")

            f.write("DELTA STATISTICS:\n")
            f.write(f"  Mean:   {delta_df['delta'].mean():.6f}\n")
            f.write(f"  Median: {delta_df['delta'].median():.6f}\n")
            f.write(f"  Std:    {delta_df['delta'].std():.6f}\n")
            f.write(f"  Min:    {delta_df['delta'].min():.6f}\n")
            f.write(f"  Max:    {delta_df['delta'].max():.6f}\n\n")

            f.write("TOP 5 MOST IMPROVED:\n")
            top_improved = delta_df.nlargest(5, 'delta')
            for idx, row in top_improved.iterrows():
                f.write(f"  {row['image_id']}: {row['pre_tta_score']:.4f} → {row['post_tta_score']:.4f} (Δ={row['delta']:.4f})\n")

            f.write("\nTOP 5 MOST DEGRADED:\n")
            top_degraded = delta_df.nsmallest(5, 'delta')
            for idx, row in top_degraded.iterrows():
                f.write(f"  {row['image_id']}: {row['pre_tta_score']:.4f} → {row['post_tta_score']:.4f} (Δ={row['delta']:.4f})\n")

        logger.info(f"\n💾 Full report saved to {report_path}")
        logger.info("=" * 60)


if __name__ == "__main__":
    # Test TTA analyzer
    logging.basicConfig(level=logging.INFO)

    analyzer = TTAAnalyzer(save_dir=Path("results/tta_analysis"))

    # Simulate some metrics
    for step in range(10):
        metrics = TTAMetrics(
            image_id="test_001",
            step=step,
            loss_total=1.0 - step * 0.08,
            loss_similarity=0.5 - step * 0.04,
            loss_variance=0.3 - step * 0.02,
            loss_entropy=0.2 - step * 0.02,
            embedding_change=step * 0.01,
            top_k_similarities=[0.7 + step * 0.02, 0.6 + step * 0.02]
        )
        analyzer.record_step(metrics)

    # Save and plot
    analyzer.save_metrics()
    analyzer.plot_convergence(image_id="test_001")
    analyzer.plot_all_convergence_metrics()

    print("TTA analyzer test complete")
