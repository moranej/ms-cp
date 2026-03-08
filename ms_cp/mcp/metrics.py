from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

def summarize_masked(df: pd.DataFrame, mask: np.ndarray) -> Dict[str, float]:
    if mask.sum() == 0:
        return {
            "coverage": float("nan"),
            "avg_set_size": float("nan"),
            "n": 0,
            "covered_n": 0,
            "not_covered_n": 0,
            "avg_candidates": float("nan"),
            "avg_set_size_covered": float("nan"),
            "avg_set_size_not_covered": float("nan"),
            "hit1": float("nan"),
            "hit5": float("nan"),
            "hit20": float("nan"),
            "avg_true_rank": float("nan"),
            "avg_true_rank_covered": float("nan"),
            "avg_true_rank_not_covered": float("nan"),
        }

    sub = df.loc[mask].copy()
    cov = sub["covered"].astype(bool)

    def _mean(s):
        return float(np.nanmean(s.astype(float))) if len(s) else float("nan")

    out = {
        "coverage": _mean(cov),
        "avg_set_size": _mean(sub["set_size"]),
        "n": int(len(sub)),
        "covered_n": int(cov.sum()),
        "not_covered_n": int((~cov).sum()),
        "avg_candidates": _mean(sub["num_candidates"]),
        "avg_set_size_covered": _mean(sub.loc[cov, "set_size"]) if cov.sum() > 0 else float("nan"),
        "avg_set_size_not_covered": _mean(sub.loc[~cov, "set_size"]) if (~cov).sum() > 0 else float("nan"),
        "hit1": _mean(sub["top1"]),
        "hit5": _mean(sub["top5"]),
        "hit20": _mean(sub["top20"]),
        "avg_true_rank": _mean(sub["true_rank"]),
        "avg_true_rank_covered": _mean(sub.loc[cov, "true_rank"]) if cov.sum() > 0 else float("nan"),
        "avg_true_rank_not_covered": _mean(sub.loc[~cov, "true_rank"]) if (~cov).sum() > 0 else float("nan"),
    }
    return out


def summarize_results(per_sample_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    seen_mask = per_sample_df["seen_in_train"].astype(bool).to_numpy()
    unseen_mask = ~seen_mask
    all_mask = np.ones(len(per_sample_df), dtype=bool)
    return {
        "overall": summarize_masked(per_sample_df, all_mask),
        "seen": summarize_masked(per_sample_df, seen_mask),
        "unseen": summarize_masked(per_sample_df, unseen_mask),
    }


def format_summary_text(summary: Dict[str, Dict[str, float]]) -> str:
    lines: List[str] = []
    for tag in ["overall", "seen", "unseen"]:
        s = summary[tag]
        lines.append(tag.upper())
        for key, val in s.items():
            lines.append(f"{key}: {val}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
