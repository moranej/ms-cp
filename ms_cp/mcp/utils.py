from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def dump_json(obj: Any, path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_builtin(obj), fh, indent=2, sort_keys=True)
