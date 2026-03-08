from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch

class MCPScore(ABC):
    @abstractmethod
    def calibrate(self, logits_list: List[torch.Tensor], true_idx_list: List[int], alpha: float, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict_set(self, logits: torch.Tensor, calibration_state: Dict[str, Any], **kwargs):
        raise NotImplementedError
