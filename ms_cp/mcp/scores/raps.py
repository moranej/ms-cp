from __future__ import annotations
import math
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from .base import MCPScore

def _q_level(n: int, alpha: float) -> float:
    if n <= 0:
        return 1.0
    return min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / float(n))


def _sort_sum_vec(scores: np.ndarray):
    order = np.argsort(scores)[::-1]
    ordered = np.sort(scores)[::-1]
    cumsum = np.cumsum(ordered)
    return order, ordered, cumsum


def _get_tau(scores: np.ndarray, true_idx: int, penalties: np.ndarray, randomized: bool, allow_zero_sets: bool, rng: np.random.RandomState) -> float:
    order, ordered, cumsum = _sort_sum_vec(scores)
    pos = int(np.where(order == true_idx)[0][0])
    tau_nonrandom = cumsum[pos]
    if not randomized:
        return float(tau_nonrandom + penalties[: pos + 1].sum())
    u = rng.rand()
    if pos == 0:
        if not allow_zero_sets:
            return float(tau_nonrandom + penalties[0])
        return float(u * tau_nonrandom + penalties[0])
    return float(u * ordered[pos] + cumsum[pos - 1] + penalties[: pos + 1].sum())


def _gcq(scores: np.ndarray, tau: float, penalties: np.ndarray, randomized: bool, allow_zero_sets: bool, rng: np.random.RandomState) -> np.ndarray:
    order, ordered, cumsum = _sort_sum_vec(scores)
    pen_cumsum = np.cumsum(penalties)
    sizes_base = ((cumsum + pen_cumsum) <= tau).sum() + 1
    sizes_base = min(int(sizes_base), int(scores.shape[0]))

    if randomized:
        k = sizes_base
        if k <= 0:
            sizes = 0
        else:
            v = (1.0 / ordered[k - 1]) * (tau - (cumsum[k - 1] - ordered[k - 1]) - pen_cumsum[k - 1])
            sizes = k - int(rng.rand() >= v)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes = int(scores.shape[0])
    if (not allow_zero_sets) and sizes == 0:
        sizes = 1
    return order[:sizes]


def _split_paramtune(logits_list: List[torch.Tensor], true_idx_list: List[int], tune_n: int, seed: int):
    n = len(logits_list)
    if n <= 1:
        return [], [], logits_list, true_idx_list
    n_param = int(min(max(1, tune_n), n - 1))
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    param_idx = set(idx[:n_param])
    param_logits, param_true, calib_logits, calib_true = [], [], [], []
    for i in range(n):
        if i in param_idx:
            param_logits.append(logits_list[i])
            param_true.append(true_idx_list[i])
        else:
            calib_logits.append(logits_list[i])
            calib_true.append(true_idx_list[i])
    return param_logits, param_true, calib_logits, calib_true


def _pick_kreg(param_logits: List[torch.Tensor], param_true: List[int], alpha: float) -> int:
    ranks = []
    for logits, true_idx in zip(param_logits, param_true):
        order = torch.argsort(logits, descending=True)
        pos0 = int((order == true_idx).nonzero(as_tuple=False).item())
        ranks.append(pos0 + 1)
    if not ranks:
        return 1
    q = _q_level(len(ranks), alpha)
    return max(1, int(np.quantile(np.asarray(ranks), q, method="higher")))


def _compute_qhat(logits_list: List[torch.Tensor], true_idx_list: List[int], T: float, k_reg: int, lamda: float, alpha: float, randomized: bool, allow_zero_sets: bool, seed: int):
    rng = np.random.RandomState(seed)
    E = []
    for logits, true_idx in zip(logits_list, true_idx_list):
        probs = F.softmax(logits / T, dim=0).cpu().numpy()
        n = probs.shape[0]
        penalties = np.zeros(n, dtype=np.float64)
        if k_reg < n:
            penalties[k_reg:] = lamda
        E.append(_get_tau(probs, true_idx, penalties, randomized, allow_zero_sets, rng))
    q = _q_level(len(E), alpha)
    qhat = float(np.quantile(np.asarray(E, dtype=np.float64), q, method="higher"))
    return qhat, E


def _avg_set_size(logits_list: List[torch.Tensor], T: float, k_reg: int, lamda: float, tau: float, randomized: bool, allow_zero_sets: bool, seed: int) -> float:
    rng = np.random.RandomState(seed)
    sizes = []
    for logits in logits_list:
        probs = F.softmax(logits / T, dim=0).cpu().numpy()
        n = probs.shape[0]
        penalties = np.zeros(n, dtype=np.float64)
        if k_reg < n:
            penalties[k_reg:] = lamda
        S = _gcq(probs, tau, penalties, randomized, allow_zero_sets, rng)
        sizes.append(int(S.shape[0]))
    return float(np.mean(sizes)) if sizes else float("nan")


def _pick_lambda(param_logits: List[torch.Tensor], param_true: List[int], T: float, k_reg: int, alpha: float, lamda_grid: List[float], randomized: bool, allow_zero_sets: bool, seed: int) -> float:
    best_lam = float(lamda_grid[0])
    best_size = float("inf")
    for lam in lamda_grid:
        tau, _ = _compute_qhat(param_logits, param_true, T, k_reg, lam, alpha, randomized, allow_zero_sets, seed)
        avg_size = _avg_set_size(param_logits, T, k_reg, lam, tau, randomized, allow_zero_sets, seed)
        if avg_size < best_size:
            best_size = avg_size
            best_lam = float(lam)
    return best_lam


def _optimize_temperature(logits_list: List[torch.Tensor], true_idx_list: List[int], device: torch.device, max_iters: int = 100, lr: float = 0.01, eps: float = 1e-3) -> float:
    if len(logits_list) == 0:
        return 1.0
    T = torch.nn.Parameter(torch.tensor([1.3], dtype=torch.float32, device=device))
    opt = torch.optim.SGD([T], lr=lr)

    def nll_once() -> torch.Tensor:
        loss = 0.0
        for logits, true_idx in zip(logits_list, true_idx_list):
            logits = logits.to(device).detach()
            loss = loss + (torch.logsumexp(logits / T, dim=0) - (logits[true_idx] / T))
        return loss / len(logits_list)

    with torch.enable_grad():
        prev = float("inf")
        for _ in range(max_iters):
            opt.zero_grad()
            loss = nll_once()
            loss.backward()
            opt.step()
            T.data.clamp_(min=1e-4)
            cur = float(loss.item())
            if abs(prev - cur) < eps:
                break
            prev = cur
    return float(T.detach().item())


class RAPSScore(MCPScore):
    def calibrate(self, logits_list: List[torch.Tensor], true_idx_list: List[int], alpha: float, **kwargs) -> Dict[str, Any]:
        seed = int(kwargs.get("seed", 0))
        tune_n = int(kwargs.get("tune_n", 1000))
        lambda_grid = list(kwargs.get("lambda_grid", [0.001, 0.01, 0.1, 0.2, 0.5]))
        randomized = bool(kwargs.get("randomized", False))
        allow_zero_sets = bool(kwargs.get("allow_zero_sets", False))
        device = kwargs.get("device", torch.device("cpu"))

        param_logits, param_true, calib_logits, calib_true = _split_paramtune(logits_list, true_idx_list, tune_n=tune_n, seed=seed)
        T = _optimize_temperature(param_logits if param_logits else logits_list, param_true if param_true else true_idx_list, device=device)
        k_reg = _pick_kreg(param_logits if param_logits else logits_list, param_true if param_true else true_idx_list, alpha=alpha)
        lamda = _pick_lambda(param_logits if param_logits else logits_list, param_true if param_true else true_idx_list, T=T, k_reg=k_reg, alpha=alpha, lamda_grid=lambda_grid, randomized=randomized, allow_zero_sets=allow_zero_sets, seed=seed)
        qhat, calib_scores = _compute_qhat(calib_logits if calib_logits else logits_list, calib_true if calib_true else true_idx_list, T=T, k_reg=k_reg, lamda=lamda, alpha=alpha, randomized=randomized, allow_zero_sets=allow_zero_sets, seed=seed)

        return {
            "method": "raps",
            "alpha": alpha,
            "temperature": T,
            "k_reg": k_reg,
            "lambda": lamda,
            "threshold": qhat,
            "n_calibration": len(calib_scores),
            "calibration_scores": calib_scores,
            "randomized": randomized,
            "allow_zero_sets": allow_zero_sets,
        }

    def predict_set(self, logits: torch.Tensor, calibration_state: Dict[str, Any], **kwargs):
        seed = int(kwargs.get("seed", 0))
        rng = np.random.RandomState(seed)
        T = float(calibration_state["temperature"])
        k_reg = int(calibration_state["k_reg"])
        lamda = float(calibration_state["lambda"])
        tau = float(calibration_state["threshold"])
        randomized = bool(calibration_state.get("randomized", False))
        allow_zero_sets = bool(calibration_state.get("allow_zero_sets", False))

        probs = F.softmax(logits / T, dim=0).detach().cpu().numpy()
        n = probs.shape[0]
        penalties = np.zeros(n, dtype=np.float64)
        if k_reg < n:
            penalties[k_reg:] = lamda
        keep = _gcq(probs, tau, penalties, randomized, allow_zero_sets, rng)
        return keep, probs
