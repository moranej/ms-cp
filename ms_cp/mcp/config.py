from __future__ import annotations
import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run marginal conformal prediction for molecular retrieval."
    )
    p.add_argument("--dataset_tsv", type=str, required=True, help="Path to dataset split TSV (Data related to each scenario).")
    p.add_argument("--helper_dir", type=str, required=True, help="Path to helper data directory (MassSpecgGym data).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained retrieval model checkpoint.")
    p.add_argument("--score", type=str, required=True, choices=["lac", "aps", "raps"], help="Nonconformity score.")
    p.add_argument("--alpha", type=float, required=True, help="Miscoverage level, e.g. 0.1 for 90%% coverage.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    # Dataset
    p.add_argument("--max_mz", type=float, default=1005.0)
    p.add_argument("--bin_width", type=float, default=0.1)
    p.add_argument("--fp_size", type=int, default=4096)

    # RAPS-specific options
    p.add_argument("--tune_n", type=int, default=1000)
    p.add_argument(
        "--lambda_grid",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 0.2, 0.5],
        help="Grid for RAPS lambda tuning.",
    )
    p.add_argument("--randomized", action="store_true", help="Use randomized RAPS.")
    p.add_argument("--allow_zero_sets", action="store_true", help="Allow empty prediction sets.")
    return p


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    args.output_dir = str(Path(args.output_dir))
    return args
