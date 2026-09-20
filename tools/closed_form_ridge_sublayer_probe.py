#!/usr/bin/env python3
"""Closed-form ridge probe for one frozen transformer sublayer.

This is a safe, offline experiment tool for AGILLM DiffusionBlocks.
It does not touch a live trainer unless you explicitly feed its output back in.

Target objective for frozen activations X and residual target Y:
    min_W ||X W - Y||_F^2 + l2 * ||W - W0||_F^2

Closed form:
    W* = (X.T @ X + l2 * I)^(-1) @ (X.T @ Y + l2 * W0)

Expected capture format, saved with torch.save:
    {
      "X": tensor[..., din],
      "Y": tensor[..., dout],
      "W0": tensor[din, dout]        # optional if --checkpoint/--weight-key supplied
    }

For torch.nn.Linear weights, PyTorch stores [dout, din], so use
--orientation dout_din if loading from a state_dict key.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def _flatten_last_dim(t: torch.Tensor) -> torch.Tensor:
    if t.ndim < 2:
        raise ValueError(f"expected at least 2D tensor, got shape {tuple(t.shape)}")
    return t.reshape(-1, t.shape[-1]).contiguous()


def _load_pt(path: str | os.PathLike[str]) -> Any:
    return torch.load(Path(path), map_location="cpu")


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "module"):
            val = obj.get(key)
            if isinstance(val, dict):
                return val
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError("could not find a state_dict-like mapping in checkpoint")


def _state_weight_to_w(weight: torch.Tensor, orientation: str) -> torch.Tensor:
    if orientation == "dout_din":
        return weight.detach().to(torch.float64).t().contiguous()
    if orientation == "din_dout":
        return weight.detach().to(torch.float64).contiguous()
    raise ValueError(f"unknown orientation: {orientation}")


def _w_to_state_weight(w: torch.Tensor, orientation: str, dtype: torch.dtype) -> torch.Tensor:
    if orientation == "dout_din":
        return w.t().contiguous().to(dtype)
    if orientation == "din_dout":
        return w.contiguous().to(dtype)
    raise ValueError(f"unknown orientation: {orientation}")


def _sample_rows(x: torch.Tensor, y: torch.Tensor, max_rows: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x, y
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:max_rows]
    return x.index_select(0, idx), y.index_select(0, idx)


def _split_holdout(x: torch.Tensor, y: torch.Tensor, holdout_frac: float, seed: int):
    if holdout_frac <= 0.0:
        return x, y, x, y
    n = x.shape[0]
    h = max(1, int(round(n * holdout_frac)))
    h = min(h, n - 1) if n > 1 else 0
    if h <= 0:
        return x, y, x, y
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 17)
    perm = torch.randperm(n, generator=g)
    hold = perm[:h]
    train = perm[h:]
    return x.index_select(0, train), y.index_select(0, train), x.index_select(0, hold), y.index_select(0, hold)


def mse(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    pred = x @ w
    return torch.mean((pred - y) ** 2)


def solve_ridge_closed_form(
    x: torch.Tensor,
    y: torch.Tensor,
    w0: torch.Tensor,
    l2: float,
    jitter: float = 1e-7,
) -> torch.Tensor:
    """Solve (X.T X + l2 I) W = X.T Y + l2 W0 in float64 on CPU."""
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    x = x.to(torch.float64).contiguous()
    y = y.to(torch.float64).contiguous()
    w0 = w0.to(torch.float64).contiguous()
    if x.ndim != 2 or y.ndim != 2 or w0.ndim != 2:
        raise ValueError("x, y, w0 must all be 2D after flattening")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row mismatch X/Y: {x.shape} vs {y.shape}")
    if x.shape[1] != w0.shape[0] or y.shape[1] != w0.shape[1]:
        raise ValueError(f"shape mismatch X={x.shape} Y={y.shape} W0={w0.shape}")

    din = x.shape[1]
    gram = x.t().matmul(x)
    rhs = x.t().matmul(y) + float(l2) * w0
    gram.diagonal().add_(float(l2))

    # Cholesky is fastest when the ridge term makes the Gram positive definite.
    # If a capture is rank-deficient and l2 is tiny, add tiny jitter or fall back.
    eye = torch.eye(din, dtype=gram.dtype, device=gram.device)
    last_error = None
    for scale in (0.0, jitter, jitter * 10.0, jitter * 100.0):
        try:
            chol = torch.linalg.cholesky(gram + scale * eye)
            return torch.cholesky_solve(rhs, chol)
        except RuntimeError as exc:
            last_error = exc
    try:
        return torch.linalg.solve(gram + jitter * 1000.0 * eye, rhs)
    except RuntimeError as exc:
        raise RuntimeError(f"ridge solve failed; last cholesky error={last_error}; solve error={exc}") from exc


def load_capture(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    cap = _load_pt(args.capture)
    if not isinstance(cap, dict):
        raise ValueError("capture must be a torch-saved dict")
    x = cap.get("X", cap.get("x"))
    y = cap.get("Y", cap.get("y"))
    w0 = cap.get("W0", cap.get("w0", cap.get("weight")))
    if x is None or y is None:
        raise ValueError("capture must contain X/Y or x/y")
    x = _flatten_last_dim(x.detach().cpu())
    y = _flatten_last_dim(y.detach().cpu())
    if w0 is not None:
        w0 = w0.detach().cpu().to(torch.float64)
    return x, y, w0


def maybe_load_w0_from_checkpoint(args: argparse.Namespace, w0: torch.Tensor | None):
    original_dtype = torch.float32
    checkpoint_obj = None
    state = None
    if args.checkpoint:
        checkpoint_obj = _load_pt(args.checkpoint)
        state = _extract_state_dict(checkpoint_obj)
        if args.weight_key is None:
            raise ValueError("--weight-key is required with --checkpoint")
        if args.weight_key not in state:
            matches = [k for k in state.keys() if args.weight_key in k]
            raise KeyError(f"weight key {args.weight_key!r} not found; partial matches={matches[:20]}")
        raw = state[args.weight_key]
        original_dtype = raw.dtype
        w0 = _state_weight_to_w(raw, args.orientation)
    if w0 is None:
        raise ValueError("W0 missing: provide W0 in capture or pass --checkpoint and --weight-key")
    return w0, checkpoint_obj, state, original_dtype


def save_updated_checkpoint(
    args: argparse.Namespace,
    checkpoint_obj: Any,
    state: Dict[str, torch.Tensor],
    w_new: torch.Tensor,
    original_dtype: torch.dtype,
) -> None:
    if not args.update_out:
        return
    if checkpoint_obj is None or state is None or args.weight_key is None:
        raise ValueError("--update-out requires --checkpoint and --weight-key")
    state[args.weight_key] = _w_to_state_weight(w_new, args.orientation, original_dtype)
    torch.save(checkpoint_obj, args.update_out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", required=True, help="torch .pt capture with X/Y and optional W0")
    p.add_argument("--checkpoint", help="optional checkpoint/state_dict to load and/or update")
    p.add_argument("--weight-key", help="state_dict key for the target sublayer weight")
    p.add_argument("--orientation", choices=("dout_din", "din_dout"), default="dout_din")
    p.add_argument("--l2", type=float, default=1e-2, help="ridge λ; keep >0 for stable unique solve")
    p.add_argument("--max-rows", type=int, default=65536, help="subsample rows before solve; <=0 disables")
    p.add_argument("--holdout-frac", type=float, default=0.10, help="held-out capture rows for rejection gate")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--accept-ratio", type=float, default=1.000, help="accept if holdout new_mse <= base_mse * ratio")
    p.add_argument("--update-out", help="write updated checkpoint only if accepted")
    p.add_argument("--report", default="ridge_sublayer_report.json")
    args = p.parse_args()

    x, y, w0 = load_capture(args)
    x, y = _sample_rows(x, y, args.max_rows, args.seed)
    train_x, train_y, hold_x, hold_y = _split_holdout(x, y, args.holdout_frac, args.seed)
    w0, checkpoint_obj, state, original_dtype = maybe_load_w0_from_checkpoint(args, w0)

    if train_x.shape[1] != w0.shape[0] or train_y.shape[1] != w0.shape[1]:
        raise ValueError(f"capture shape incompatible with W0: X={train_x.shape}, Y={train_y.shape}, W0={w0.shape}")

    w_new = solve_ridge_closed_form(train_x, train_y, w0, args.l2)
    base_train = mse(train_x, train_y, w0).item()
    new_train = mse(train_x, train_y, w_new).item()
    base_hold = mse(hold_x, hold_y, w0).item()
    new_hold = mse(hold_x, hold_y, w_new).item()

    accepted = bool(math.isfinite(new_hold) and new_hold <= base_hold * float(args.accept_ratio))
    report = {
        "subproblem": "closed_form_ridge_frozen_sublayer",
        "objective": "min_W ||XW-Y||_F^2 + l2 ||W-W0||_F^2",
        "closed_form": "W=(X^T X + l2 I)^-1 (X^T Y + l2 W0)",
        "capture": str(args.capture),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "weight_key": args.weight_key,
        "orientation": args.orientation,
        "rows_total_after_sample": int(x.shape[0]),
        "rows_train": int(train_x.shape[0]),
        "rows_holdout": int(hold_x.shape[0]),
        "din": int(w0.shape[0]),
        "dout": int(w0.shape[1]),
        "l2": float(args.l2),
        "base_train_mse": base_train,
        "new_train_mse": new_train,
        "base_holdout_mse": base_hold,
        "new_holdout_mse": new_hold,
        "holdout_delta": new_hold - base_hold,
        "holdout_ratio": new_hold / base_hold if base_hold != 0.0 else None,
        "accepted": accepted,
        "update_out": str(args.update_out) if args.update_out and accepted else None,
    }

    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if accepted and args.update_out:
        save_updated_checkpoint(args, checkpoint_obj, state, w_new, original_dtype)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
