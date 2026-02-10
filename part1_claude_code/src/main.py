"""Main pipeline script to orchestrate Wine classification workflow."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from eda import perform_eda
from evaluate_model import evaluate_model
from feature_engineering import perform_feature_engineering
from generate_report import generate_report
from train_model import train_model
from tune_hyperparameters import tune_hyperparameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output"


def _print_separator() -> None:
    """Print a separator line."""
    logging.info("=" * 80)


def _print_step_header(
    step_number: int,
    step_name: str,
) -> None:
    """Print a formatted step header."""
    _print_separator()
    logging.info(f"STEP {step_number}: {step_name}")
    _print_separator()


def run_pipeline(
    steps: list[str] | None = None,
) -> dict[str, Any]:
    """Run the complete Wine classification pipeline."""
    start_time = time.time()

    logging.info("Starting Wine Classification Pipeline")
    _print_separator()

    # Define all available steps
    all_steps = ["eda", "feature_engineering", "tuning", "training", "evaluation", "report"]

    # Use all steps if none specified
    if steps is None or "all" in steps:
        steps = all_steps
    else:
        # Validate step names
        invalid_steps = [s for s in steps if s not in all_steps]
        if invalid_steps:
            logging.error(f"Invalid step names: {invalid_steps}")
            logging.info(f"Valid steps: {all_steps}")
            sys.exit(1)

    results = {}

    try:
        # Step 1: Exploratory Data Analysis
        if "eda" in steps:
            _print_step_header(1, "Exploratory Data Analysis")
            perform_eda()
            results["eda"] = "completed"
            logging.info("EDA completed successfully")

        # Step 2: Feature Engineering
        if "feature_engineering" in steps:
            _print_step_header(2, "Feature Engineering")
            feature_summary = perform_feature_engineering()
            results["feature_engineering"] = feature_summary
            logging.info("Feature engineering completed successfully")

        # Step 3: Hyperparameter Tuning
        if "tuning" in steps:
            _print_step_header(3, "Hyperparameter Tuning")
            tuning_results = tune_hyperparameters()
            results["tuning"] = tuning_results
            logging.info("Hyperparameter tuning completed successfully")

        # Step 4: Model Training
        if "training" in steps:
            _print_step_header(4, "Model Training")
            training_summary = train_model()
            results["training"] = training_summary
            logging.info("Model training completed successfully")

        # Step 5: Model Evaluation
        if "evaluation" in steps:
            _print_step_header(5, "Model Evaluation")
            eval_summary = evaluate_model()
            results["evaluation"] = eval_summary
            logging.info("Model evaluation completed successfully")

        # Step 6: Report Generation
        if "report" in steps:
            _print_step_header(6, "Report Generation")
            generate_report()
            results["report"] = "completed"
            logging.info("Report generation completed successfully")

        # Print summary
        _print_separator()
        elapsed_time = time.time() - start_time
        logging.info("PIPELINE COMPLETED SUCCESSFULLY")
        logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
        logging.info(f"Steps completed: {len(results)}")
        logging.info(f"Output directory: {Path(OUTPUT_DIR).absolute()}")
        _print_separator()

        return results

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Wine Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py

  # Run specific steps
  python main.py --steps eda feature_engineering

  # Run from tuning onwards
  python main.py --steps tuning training evaluation report
        """,
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["all", "eda", "feature_engineering", "tuning", "training", "evaluation", "report"],
        default=["all"],
        help="Specify which steps to run (default: all)",
    )

    args = parser.parse_args()

    # Run pipeline
    run_pipeline(steps=args.steps)


if __name__ == "__main__":
    main()
