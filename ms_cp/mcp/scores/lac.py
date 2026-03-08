from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import torch
from .base import MCPScore


class LACScore(MCPScore):
    def calibrate(self, logits_list: List[torch.Tensor], true_idx_list: List[int], alpha: float, **kwargs) -> Dict[str, Any]:
        scores = []
        for logits, true_idx in zip(logits_list, true_idx_list):
            probs = torch.softmax(logits, dim=0)
            scores.append(float(1.0 - probs[true_idx].item()))
        tau = float(torch.quantile(torch.tensor(scores, dtype=torch.float32), 1 - alpha).item())
        return {
            "method": "lac",
            "alpha": alpha,
            "threshold": tau,
            "n_calibration": len(scores),
            "calibration_scores": scores,
        }

    def predict_set(self, logits: torch.Tensor, calibration_state: Dict[str, Any], **kwargs):
        probs = torch.softmax(logits, dim=0)
        threshold = float(calibration_state["threshold"])
        keep = torch.where((1.0 - probs) <= threshold)[0]
        if keep.numel() == 0:
            keep = torch.tensor([int(torch.argmax(probs).item())], dtype=torch.long)
        return keep.cpu().numpy(), probs.detach().cpu().numpy()
