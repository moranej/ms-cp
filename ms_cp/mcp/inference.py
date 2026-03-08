from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import torch
import massspecgym.utils as utils
from torch.utils.data import DataLoader
from torch_geometric.utils import unbatch

def forward_candidate_scores(
    model,
    batch,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    x = batch["spec"].to(device)
    cands = batch["candidates"].int().to(device)
    labels = batch["labels"].to(device)
    batch_ptr = batch["batch_ptr"].to(device)

    with torch.no_grad():
        fp_pred = torch.sigmoid(model.loss.fp_pred_head(model(x)))
        scores = model.loss.ranker(fp_pred.repeat_interleave(batch_ptr, 0), cands)

    bidx = utils.batch_ptr_to_batch_idx(batch_ptr)
    scores_list = unbatch(scores, bidx)
    labels_list = unbatch(labels, bidx)
    return scores_list, labels_list


def _resolve_true_idx(labels: torch.Tensor, scores: torch.Tensor) -> Optional[int]:
    pos = torch.where(labels.bool())[0]
    if pos.numel() == 0:
        return None
    if pos.numel() == 1:
        return int(pos[0].item())

    probs = torch.softmax(scores, dim=0)
    best_local = torch.argmax(probs[pos]).item()
    return int(pos[best_local].item())


def collect_logits_and_true_indices(
    model,
    loader: DataLoader,
    device: torch.device,
    sample_indices: Optional[Sequence[int]] = None,
):
    logits_list = []
    true_idx_list = []
    kept_sample_indices = []

    global_ptr = 0
    for batch in loader:
        scores_list, labels_list = forward_candidate_scores(model, batch, device)

        for scores, labels in zip(scores_list, labels_list):
            ds_idx = None
            if sample_indices is not None:
                ds_idx = int(sample_indices[global_ptr])

            true_idx = _resolve_true_idx(labels, scores)
            global_ptr += 1

            if true_idx is None:
                continue

            logits_list.append(scores.detach().cpu())
            true_idx_list.append(true_idx)
            if ds_idx is not None:
                kept_sample_indices.append(ds_idx)

    if sample_indices is None:
        return logits_list, true_idx_list
    return logits_list, true_idx_list, kept_sample_indices