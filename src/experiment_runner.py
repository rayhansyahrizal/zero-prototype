"""
Experiment Runner for TTA Ablation Studies.

Supports systematic experiments to evaluate:
1. Different TTA configurations (learning rate, steps, etc.)
2. Different adaptation strategies (1D vector, LoRA, etc.)
3. Different loss functions and regularization weights

Following Section 5 (Studi Ablasi TTA) from the thesis document.
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Any
import itertools
import subprocess

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runner for systematic TTA ablation experiments."""

    def __init__(self, base_config_path: Path, results_dir: Path):
        """
        Initialize experiment runner.

        Args:
            base_config_path: Path to base configuration file
            results_dir: Directory to save all experiment results
        """
        self.base_config_path = Path(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load base config
        with open(self.base_config_path) as f:
            self.base_config = yaml.safe_load(f)

        logger.info(f"ExperimentRunner initialized")
        logger.info(f"  Base config: {self.base_config_path}")
        logger.info(f"  Results dir: {self.results_dir}")

    def create_experiment_config(
        self,
        experiment_name: str,
        modifications: Dict[str, Any]
    ) -> Path:
        """
        Create a modified config file for an experiment.

        Args:
            experiment_name: Name of the experiment
            modifications: Dictionary of config modifications (nested keys supported)

        Returns:
            Path to the created config file
        """
        config = self.base_config.copy()

        # Apply modifications (supports nested dict)
        for key_path, value in modifications.items():
            keys = key_path.split('.')
            current = config

            # Navigate to nested dict
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set value
            current[keys[-1]] = value

        # Save config
        config_path = self.results_dir / f"config_{experiment_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created experiment config: {config_path}")
        return config_path

    def run_single_experiment(
        self,
        experiment_name: str,
        config_path: Path,
        skip_embedding: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            experiment_name: Name of the experiment
            config_path: Path to config file
            skip_embedding: Whether to skip embedding generation (use cached)

        Returns:
            Dictionary with experiment results
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RUNNING EXPERIMENT: {experiment_name}")
        logger.info(f"{'=' * 60}")

        # Construct command
        cmd = [
            sys.executable,
            "-m", "src.main_tta_experiment",
            "--config", str(config_path)
        ]

        if skip_embedding:
            cmd.append("--skip-embedding")

        # Run experiment
        try:
            start_time = datetime.now()

            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if result.returncode != 0:
                logger.error(f"Experiment {experiment_name} failed:")
                logger.error(result.stderr)
                return {
                    'experiment_name': experiment_name,
                    'status': 'failed',
                    'duration': duration,
                    'error': result.stderr
                }

            logger.info(f"✅ Experiment {experiment_name} completed in {duration:.1f}s")

            # Parse results (look for latest metrics file)
            with open(config_path) as f:
                exp_config = yaml.safe_load(f)

            results_dir = Path(exp_config['output']['results_dir'])
            metrics_files = sorted(results_dir.glob("metrics_*.csv"))

            if metrics_files:
                latest_metrics = pd.read_csv(metrics_files[-1])
                metrics_dict = latest_metrics.set_index('method').to_dict('index')

                return {
                    'experiment_name': experiment_name,
                    'status': 'success',
                    'duration': duration,
                    'metrics': metrics_dict,
                    'config_path': str(config_path),
                    'metrics_file': str(metrics_files[-1])
                }
            else:
                logger.warning(f"No metrics file found for {experiment_name}")
                return {
                    'experiment_name': experiment_name,
                    'status': 'success_no_metrics',
                    'duration': duration
                }

        except subprocess.TimeoutExpired:
            logger.error(f"Experiment {experiment_name} timed out")
            return {
                'experiment_name': experiment_name,
                'status': 'timeout',
                'duration': 3600
            }
        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed with exception: {e}")
            return {
                'experiment_name': experiment_name,
                'status': 'error',
                'error': str(e)
            }

    def run_ablation_study(
        self,
        study_name: str,
        param_grid: Dict[str, List[Any]],
        skip_embedding: bool = True
    ) -> pd.DataFrame:
        """
        Run ablation study with parameter grid.

        Args:
            study_name: Name of the ablation study
            param_grid: Dictionary mapping parameter paths to list of values
            skip_embedding: Whether to skip embedding generation

        Returns:
            DataFrame with all experiment results
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"STARTING ABLATION STUDY: {study_name}")
        logger.info(f"{'=' * 70}")

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Total experiments to run: {len(combinations)}")

        # Run all experiments
        all_results = []

        for i, combo in enumerate(combinations, 1):
            # Create modifications dict
            modifications = dict(zip(param_names, combo))

            # Create experiment name
            exp_name = f"{study_name}_exp{i:03d}"
            for param, value in modifications.items():
                param_short = param.split('.')[-1]
                exp_name += f"_{param_short}{value}"

            # Create config
            config_path = self.create_experiment_config(exp_name, modifications)

            # Run experiment
            result = self.run_single_experiment(
                exp_name,
                config_path,
                skip_embedding=skip_embedding
            )

            # Add parameter values to result
            for param, value in modifications.items():
                result[param] = value

            all_results.append(result)

            logger.info(f"Progress: {i}/{len(combinations)} experiments completed")

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        # Save results
        results_csv = self.results_dir / f"{study_name}_results.csv"
        results_df.to_csv(results_csv, index=False)

        results_json = self.results_dir / f"{study_name}_results.json"
        with open(results_json, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"ABLATION STUDY COMPLETE: {study_name}")
        logger.info(f"  Total experiments: {len(all_results)}")
        logger.info(f"  Successful: {sum(1 for r in all_results if r['status'] == 'success')}")
        logger.info(f"  Failed: {sum(1 for r in all_results if r['status'] == 'failed')}")
        logger.info(f"  Results saved to: {results_csv}")
        logger.info(f"{'=' * 70}")

        return results_df

    def analyze_ablation_results(
        self,
        results_df: pd.DataFrame,
        target_metric: str = 'bleu_4',
        method: str = 'prototype_with_tta'
    ):
        """
        Analyze ablation study results.

        Args:
            results_df: DataFrame from run_ablation_study
            target_metric: Which metric to analyze
            method: Which method to extract metrics from
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ABLATION ANALYSIS: {target_metric} for {method}")
        logger.info(f"{'=' * 60}")

        # Filter successful experiments
        success_df = results_df[results_df['status'] == 'success'].copy()

        if len(success_df) == 0:
            logger.warning("No successful experiments to analyze")
            return

        # Extract target metric
        def extract_metric(metrics_dict):
            if pd.isna(metrics_dict):
                return None
            if isinstance(metrics_dict, str):
                import ast
                metrics_dict = ast.literal_eval(metrics_dict)
            if method in metrics_dict and target_metric in metrics_dict[method]:
                return metrics_dict[method][target_metric]
            return None

        success_df['target_metric_value'] = success_df['metrics'].apply(extract_metric)
        success_df = success_df.dropna(subset=['target_metric_value'])

        if len(success_df) == 0:
            logger.warning(f"No valid {target_metric} values found for {method}")
            return

        # Find best configuration
        best_idx = success_df['target_metric_value'].idxmax()
        best_row = success_df.loc[best_idx]

        logger.info(f"\nBEST CONFIGURATION:")
        logger.info(f"  Experiment: {best_row['experiment_name']}")
        logger.info(f"  {target_metric}: {best_row['target_metric_value']:.4f}")

        # Print parameter values
        param_cols = [col for col in success_df.columns if col.startswith('tta.')]
        for col in param_cols:
            logger.info(f"  {col}: {best_row[col]}")

        # Summary statistics
        logger.info(f"\nSUMMARY STATISTICS:")
        logger.info(f"  Mean {target_metric}: {success_df['target_metric_value'].mean():.4f}")
        logger.info(f"  Std {target_metric}: {success_df['target_metric_value'].std():.4f}")
        logger.info(f"  Min {target_metric}: {success_df['target_metric_value'].min():.4f}")
        logger.info(f"  Max {target_metric}: {success_df['target_metric_value'].max():.4f}")

        # Parameter sensitivity (if applicable)
        logger.info(f"\nPARAMETER SENSITIVITY:")
        for col in param_cols:
            if success_df[col].nunique() > 1:
                grouped = success_df.groupby(col)['target_metric_value'].agg(['mean', 'std', 'count'])
                logger.info(f"\n  {col}:")
                logger.info(f"{grouped.to_string()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run TTA ablation experiments"
    )
    parser.add_argument(
        '--base-config',
        type=str,
        default='config.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/ablation_studies',
        help='Directory to save experiment results'
    )
    parser.add_argument(
        '--study',
        type=str,
        required=True,
        choices=['learning_rate', 'num_steps', 'regularization', 'full'],
        help='Which ablation study to run'
    )
    parser.add_argument(
        '--skip-embedding',
        action='store_true',
        default=True,
        help='Skip embedding generation (use cached)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize runner
    runner = ExperimentRunner(
        base_config_path=Path(args.base_config),
        results_dir=Path(args.results_dir)
    )

    # Define ablation studies
    if args.study == 'learning_rate':
        param_grid = {
            'tta.learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'tta.num_steps': [10]  # Fixed
        }
    elif args.study == 'num_steps':
        param_grid = {
            'tta.learning_rate': [0.001],  # Fixed
            'tta.num_steps': [5, 10, 15, 20, 25]
        }
    elif args.study == 'regularization':
        param_grid = {
            'tta.weight_variance': [0.0, 0.05, 0.1, 0.2],
            'tta.weight_entropy': [0.0, 0.005, 0.01, 0.02]
        }
    elif args.study == 'full':
        # Full grid search
        param_grid = {
            'tta.learning_rate': [0.0005, 0.001, 0.005],
            'tta.num_steps': [5, 10, 20],
            'tta.weight_variance': [0.0, 0.1],
            'tta.weight_entropy': [0.0, 0.01]
        }

    # Run ablation study
    results_df = runner.run_ablation_study(
        study_name=args.study,
        param_grid=param_grid,
        skip_embedding=args.skip_embedding
    )

    # Analyze results
    runner.analyze_ablation_results(results_df)


if __name__ == "__main__":
    main()
