from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import torch
from .base import MCPScore

class APSScore(MCPScore):
    def calibrate(self, logits_list: List[torch.Tensor], true_idx_list: List[int], alpha: float, **kwargs) -> Dict[str, Any]:
        scores = []
        for logits, true_idx in zip(logits_list, true_idx_list):
            probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
            order = np.argsort(-probs)
            sorted_probs = probs[order]
            cumsum = np.cumsum(sorted_probs)
            rank_pos = int(np.where(order == true_idx)[0][0])
            scores.append(float(cumsum[rank_pos]))
        tau = float(torch.quantile(torch.tensor(scores, dtype=torch.float32), 1 - alpha).item())
        return {
            "method": "aps",
            "alpha": alpha,
            "threshold": tau,
            "n_calibration": len(scores),
            "calibration_scores": scores,
        }

    def predict_set(self, logits: torch.Tensor, calibration_state: Dict[str, Any], **kwargs):
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        tau = float(calibration_state["threshold"])
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumsum = np.cumsum(sorted_probs)
        k = int(np.searchsorted(cumsum, tau, side="left") + 1)
        keep = order[: min(k, len(order))]
        return keep, probs
