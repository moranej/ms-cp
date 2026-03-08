from __future__ import annotations
import torch
from models import FingerprintPredicter

def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_retrieval_model(checkpoint_path: str, device: torch.device) -> FingerprintPredicter:
    model = FingerprintPredicter.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model
