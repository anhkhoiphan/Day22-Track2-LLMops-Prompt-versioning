"""
Run all lab steps sequentially (or a specific step with --step N).

Usage:
  python run_all.py           # run all steps 1-4
  python run_all.py --step 3  # run only step 3
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def load_step(filename: str):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEPS = {
    1: "01_langsmith_rag_pipeline.py",
    2: "02_prompt_hub_ab_routing.py",
    3: "03_ragas_evaluation.py",
    4: "04_guardrails_validator.py",
}


def run_step(n: int):
    if n not in STEPS:
        print(f"Unknown step: {n}. Valid steps: 1, 2, 3, 4")
        sys.exit(1)
    module = load_step(STEPS[n])
    module.main()


def main():
    parser = argparse.ArgumentParser(description="Run Day-22 lab steps")
    parser.add_argument("--step", type=int, default=None,
                        help="Run only this step (1-4). Omit to run all.")
    args = parser.parse_args()

    steps = [args.step] if args.step else [1, 2, 3, 4]
    for n in steps:
        print(f"\n{'#' * 60}")
        print(f"#  RUNNING STEP {n}")
        print(f"{'#' * 60}\n")
        run_step(n)


if __name__ == "__main__":
    main()
