#!/usr/bin/env python3
"""
Compare results from different TTA improvement experiments.
Analyzes metrics and identifies best configuration.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_experiment_results(results_dir: Path):
    """Load results from an experiment directory."""
    # Find latest comparison CSV
    comparison_files = list(results_dir.glob("comparison_*.csv"))

    if not comparison_files:
        print(f"Warning: No comparison file found in {results_dir}")
        return None

    # Get most recent
    latest_file = max(comparison_files, key=lambda p: p.stat().st_mtime)

    # Load CSV
    df = pd.read_csv(latest_file, index_col=0)

    return df


def load_detailed_scores(results_dir: Path):
    """Load detailed per-sample scores."""
    detailed_files = list(results_dir.glob("detailed_scores_*.json"))

    if not detailed_files:
        return None

    latest_file = max(detailed_files, key=lambda p: p.stat().st_mtime)

    with open(latest_file) as f:
        return json.load(f)


def compare_all_experiments(base_dir: Path):
    """Compare all experiments and generate summary report."""

    experiments = {
        'conservative': base_dir / 'results' / 'conservative',
        'moderate': base_dir / 'results' / 'moderate',
        'no_tta': base_dir / 'results' / 'no_tta',
        'more_prototypes': base_dir / 'results' / 'more_prototypes',
    }

    print("=" * 80)
    print("  TTA IMPROVEMENT EXPERIMENTS - COMPARISON REPORT")
    print("=" * 80)
    print()

    # Collect results
    all_results = {}

    for exp_name, exp_dir in experiments.items():
        if not exp_dir.exists():
            print(f"⚠️  {exp_name}: Directory not found ({exp_dir})")
            continue

        results = load_experiment_results(exp_dir)
        if results is not None:
            all_results[exp_name] = results
            print(f"✅ {exp_name}: Loaded successfully")
        else:
            print(f"❌ {exp_name}: Failed to load results")

    print()

    if not all_results:
        print("No results found! Run experiments first with:")
        print("  bash scripts/run_experiments.sh")
        return

    # Compare prototype method across experiments
    print("=" * 80)
    print("  PROTOTYPE METHOD COMPARISON (Focus: Retrieval+TTA)")
    print("=" * 80)
    print()

    prototype_comparison = []

    for exp_name, df in all_results.items():
        if 'prototype' in df.index:
            row = df.loc['prototype']
            prototype_comparison.append({
                'Experiment': exp_name,
                'BLEU-4': row.get('bleu_4', np.nan),
                'METEOR': row.get('meteor', np.nan),
                'Semantic Sim': row.get('semantic_similarity', np.nan)
            })

    if prototype_comparison:
        proto_df = pd.DataFrame(prototype_comparison)
        proto_df = proto_df.sort_values('BLEU-4', ascending=False)
        print(proto_df.to_string(index=False))
        print()

        # Highlight best
        best_exp = proto_df.iloc[0]['Experiment']
        best_bleu = proto_df.iloc[0]['BLEU-4']
        print(f"🏆 Best Configuration: {best_exp} (BLEU-4: {best_bleu:.4f})")
        print()

    # Full comparison table
    print("=" * 80)
    print("  FULL COMPARISON (All Methods)")
    print("=" * 80)
    print()

    # Build comprehensive comparison
    full_comparison = []

    for exp_name, df in all_results.items():
        for method in df.index:
            row = df.loc[method]
            full_comparison.append({
                'Experiment': exp_name,
                'Method': method,
                'BLEU-4': row.get('bleu_4', np.nan),
                'METEOR': row.get('meteor', np.nan),
                'Semantic Sim': row.get('semantic_similarity', np.nan)
            })

    full_df = pd.DataFrame(full_comparison)

    # Pivot for easier reading
    pivot_bleu = full_df.pivot(index='Method', columns='Experiment', values='BLEU-4')
    pivot_meteor = full_df.pivot(index='Method', columns='Experiment', values='METEOR')

    print("BLEU-4 Scores:")
    print(pivot_bleu.to_string())
    print()

    print("METEOR Scores:")
    print(pivot_meteor.to_string())
    print()

    # Analysis: Did improvements help?
    print("=" * 80)
    print("  ANALYSIS")
    print("=" * 80)
    print()

    # Check if prototype beats baseline
    for exp_name, df in all_results.items():
        if 'baseline' in df.index and 'prototype' in df.index:
            baseline_bleu = df.loc['baseline', 'bleu_4']
            proto_bleu = df.loc['prototype', 'bleu_4']
            diff = proto_bleu - baseline_bleu

            status = "✅" if diff > 0 else "❌"
            print(f"{status} {exp_name:20s}: Prototype vs Baseline = {diff:+.4f}")

    print()

    # Check if prototype beats regular retrieval
    print("Prototype vs Regular Retrieval:")
    for exp_name, df in all_results.items():
        if 'retrieval' in df.index and 'prototype' in df.index:
            retrieval_bleu = df.loc['retrieval', 'bleu_4']
            proto_bleu = df.loc['prototype', 'bleu_4']
            diff = proto_bleu - retrieval_bleu

            status = "✅" if diff > 0 else "❌"
            print(f"{status} {exp_name:20s}: Prototype vs Retrieval = {diff:+.4f}")

    print()

    # Recommendations
    print("=" * 80)
    print("  RECOMMENDATIONS")
    print("=" * 80)
    print()

    if 'no_tta' in all_results and any(k in all_results for k in ['conservative', 'moderate']):
        # Compare TTA vs no TTA
        no_tta_bleu = all_results['no_tta'].loc['prototype', 'bleu_4'] if 'prototype' in all_results['no_tta'].index else 0

        best_tta_exp = None
        best_tta_bleu = -1

        for exp_name in ['conservative', 'moderate', 'more_prototypes']:
            if exp_name in all_results and 'prototype' in all_results[exp_name].index:
                bleu = all_results[exp_name].loc['prototype', 'bleu_4']
                if bleu > best_tta_bleu:
                    best_tta_bleu = bleu
                    best_tta_exp = exp_name

        if best_tta_bleu > no_tta_bleu:
            print(f"✅ TTA HELPS! Best TTA config ({best_tta_exp}) outperforms no-TTA by {best_tta_bleu - no_tta_bleu:.4f}")
            print(f"   → Use config: config_tta_{best_tta_exp}.yaml")
        else:
            print(f"❌ TTA HURTS! No-TTA performs better than all TTA variants")
            print(f"   → Consider disabling TTA or further tuning")

    print()

    # Save summary
    summary_path = base_dir / 'results' / 'experiments_summary.csv'
    full_df.to_csv(summary_path, index=False)
    print(f"📊 Summary saved to: {summary_path}")
    print()


def analyze_failures(base_dir: Path, experiment_name: str):
    """Analyze specific failure cases for an experiment."""
    exp_dir = base_dir / 'results' / experiment_name

    if not exp_dir.exists():
        print(f"Error: {exp_dir} not found")
        return

    detailed = load_detailed_scores(exp_dir)

    if not detailed:
        print(f"No detailed scores found for {experiment_name}")
        return

    print(f"=" * 80)
    print(f"  FAILURE ANALYSIS: {experiment_name}")
    print(f"=" * 80)
    print()

    # Compare baseline vs prototype
    failures = []

    baseline_scores = detailed.get('baseline', {})
    prototype_scores = detailed.get('prototype', {})

    for img_id in baseline_scores.keys():
        if img_id in prototype_scores:
            baseline_bleu = baseline_scores[img_id].get('bleu_4', 0)
            proto_bleu = prototype_scores[img_id].get('bleu_4', 0)

            if proto_bleu < baseline_bleu - 0.05:  # Significant drop
                failures.append({
                    'image_id': img_id,
                    'baseline': baseline_bleu,
                    'prototype': proto_bleu,
                    'drop': baseline_bleu - proto_bleu
                })

    if failures:
        print(f"Found {len(failures)} cases where Prototype+TTA significantly underperforms baseline")
        print()
        print("Top 10 worst failures:")
        failures.sort(key=lambda x: x['drop'], reverse=True)

        for i, f in enumerate(failures[:10], 1):
            print(f"{i:2d}. {f['image_id']:30s} | Baseline: {f['baseline']:.4f} | Prototype: {f['prototype']:.4f} | Drop: {f['drop']:.4f}")
    else:
        print("No significant failures found! 🎉")

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare TTA improvement experiments")
    parser.add_argument(
        '--failures',
        type=str,
        help='Analyze failures for specific experiment (conservative, moderate, no_tta, more_prototypes)'
    )

    args = parser.parse_args()

    # Get project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    if args.failures:
        analyze_failures(project_dir, args.failures)
    else:
        compare_all_experiments(project_dir)


if __name__ == "__main__":
    main()
