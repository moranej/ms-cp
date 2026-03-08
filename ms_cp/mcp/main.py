from __future__ import annotations
import numpy as np
import pandas as pd
from .config import parse_args
from .dataset import load_retrieval_data
from .inference import collect_logits_and_true_indices
from .metrics import summarize_results
from .model import load_retrieval_model, resolve_device
from .reporting import save_calibration, save_config, save_per_sample_csv, save_summary
from .scores import get_score_method
from .utils import ensure_dir, set_seed


def _evaluate_test_set(
    dataset,
    kept_test_idx,
    logits_list,
    true_idx_list,
    score_method,
    calibration_state,
    train_inchikeys,
    seed: int,
):
    rows = []

    for global_i, (ds_idx, logits, true_idx) in enumerate(
        zip(kept_test_idx, logits_list, true_idx_list)
    ):
        meta = dataset.metadata.iloc[ds_idx]
        identifier = str(meta["identifier"])
        inchikey = str(meta["inchikey"])
        seen = inchikey in train_inchikeys

        pred_set, probs = score_method.predict_set(
            logits, calibration_state, seed=seed + global_i
        )
        pred_set = np.asarray(pred_set, dtype=int)
        probs = np.asarray(probs)

        ranked_indices = np.argsort(-probs)
        rank_pos = np.where(ranked_indices == true_idx)[0]
        true_rank = int(rank_pos[0]) + 1 if len(rank_pos) else np.nan

        set_pos = np.where(pred_set == true_idx)[0]
        covered = int(len(set_pos) > 0)
        true_set_pos = int(set_pos[0]) + 1 if covered else np.nan

        rows.append(
            {
                "identifier": identifier,
                "inchikey": inchikey,
                "seen_in_train": int(seen),
                "covered": covered,
                "set_size": int(len(pred_set)),
                "num_candidates": int(len(probs)),
                "true_idx": int(true_idx),
                "true_rank": true_rank,
                "true_set_pos": true_set_pos,
                "top1": int(true_rank == 1),
                "top5": int(true_rank <= 5) if not np.isnan(true_rank) else 0,
                "top20": int(true_rank <= 20) if not np.isnan(true_rank) else 0,
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = ensure_dir(args.output_dir)

    device = resolve_device(args.device)

    data = load_retrieval_data(
        tsv_path=args.dataset_tsv,
        helper_dir=args.helper_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_mz=args.max_mz,
        bin_width=args.bin_width,
        fp_size=args.fp_size,
    )

    model = load_retrieval_model(args.checkpoint, device)
    score_method = get_score_method(args.score)

    calib_logits, calib_true = collect_logits_and_true_indices(
        model, data.calib_loader, device
    )

    test_logits, test_true, kept_test_idx = collect_logits_and_true_indices(
        model,
        data.test_loader,
        device,
        sample_indices=data.test_idx,
    )

    calibration_state = score_method.calibrate(
        calib_logits,
        calib_true,
        alpha=args.alpha,
        seed=args.seed,
        tune_n=args.tune_n,
        lambda_grid=args.lambda_grid,
        randomized=args.randomized,
        allow_zero_sets=args.allow_zero_sets,
        device=device,
    )

    per_sample_df = _evaluate_test_set(
        dataset=data.dataset,
        kept_test_idx=kept_test_idx,
        logits_list=test_logits,
        true_idx_list=test_true,
        score_method=score_method,
        calibration_state=calibration_state,
        train_inchikeys=data.train_inchikeys,
        seed=args.seed,
    )

    summary = summarize_results(per_sample_df)

    save_config(vars(args), outdir)
    save_calibration(calibration_state, outdir)
    save_per_sample_csv(per_sample_df, outdir)
    save_summary(summary, outdir)

    print(f"Saved outputs to: {outdir}")
    print(f"Overall coverage: {summary['overall']['coverage']:.4f}")
    print(f"Average set size: {summary['overall']['avg_set_size']:.4f}")


if __name__ == "__main__":
    main()