"""DiffusionBlocks training mode folded into AGILLM-4 (gated by --dblock).

Block-wise EDM denoising on the real Encoder blocks, supervising AR + SAT(fixed+var)
+ NAT each step on ONE block, with grad-checkpointed layers and fused vocab-streaming
CE. Reuses the live data stream / optimizer / checkpointing of nB300_agillm4.
Lazy-imports nB300 inside functions to avoid a circular import.
"""
import math
import random
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ck
from fused_ce import fused_ce

SD = 0.5




def _profile_active(state, args):
    limit = int(getattr(args, "profile_steps", 0) or 0)
    return limit > 0 and int(state.get("profile_n", 0)) < limit


def _profile_add(state, name, seconds):
    if seconds is None:
        return
    prof = state.setdefault("profile_times", defaultdict(float))
    prof[name] += float(seconds)


def _profile_tic(enabled):
    if not enabled:
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _profile_toc(state, name, start):
    if start is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _profile_add(state, name, time.perf_counter() - start)


def _profile_step_done(state, args):
    limit = int(getattr(args, "profile_steps", 0) or 0)
    if limit <= 0:
        return
    n_prev = int(state.get("profile_n", 0))
    if n_prev >= limit:
        return
    state["profile_n"] = n_prev + 1
    n = int(state["profile_n"])
    log_every = max(1, int(getattr(args, "profile_log_every", 25) or 25))
    if n % log_every != 0 and n != limit:
        return
    times = state.get("profile_times", {})
    keys = [
        "data_stream", "tensor", "setup",
        "ar_forward", "ar_ce", "ar_backward",
        "sat_forward", "sat_ce", "sat_backward",
        "nat_forward", "nat_ce", "nat_backward",
        "opt_step", "step_total",
    ]
    parts = []
    for key in keys:
        val = float(times.get(key, 0.0)) * 1000.0 / max(1, n)
        if val > 0.01:
            parts.append(f"{key}={val:.2f}ms")
    print(f"[profile] n={n}/{limit} avg " + " ".join(parts), flush=True)

def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ppf(p):
    return float(torch.erfinv(torch.tensor(2 * p - 1.0)) * math.sqrt(2))


def _block_sigmas(B, smin=0.002, smax=80.0, pm=-1.2, ps=1.2):
    a, b = _cdf((math.log(smin) - pm) / ps), _cdf((math.log(smax) - pm) / ps)
    return [float(np.exp(pm + ps * _ppf(a + (b - a) * (i / B)))) for i in range(B + 1)]


def _edm_pre(s):
    s = s[:, None, None]
    return SD**2 / (s**2 + SD**2), s * SD / (s**2 + SD**2) ** 0.5, 1 / (s**2 + SD**2) ** 0.5


def _edm_w(s, wmax=5.0):
    return float(((s**2 + SD**2) / (s * SD) ** 2).clamp(max=wmax).mean())


def _dblock_init(core, args):
    B = int(getattr(args, "dblock_blocks", 4))
    L = len(core.blocks)
    sp = max(1, L // B)
    asg = [list(range(i * sp, (i + 1) * sp)) for i in range(B)]
    asg[-1] = list(range((B - 1) * sp, L))
    bsig = _block_sigmas(B)
    schedule = getattr(args, "dblock_schedule", "loss_balanced")
    print(f"[dblock] DiffusionBlocks mode: {L} layers -> {B} blocks {asg}")
    print(f"[dblock] schedule={schedule} sigma boundaries: {[round(x, 3) for x in bsig]}")
    return {
        "B": B,
        "assign": asg,
        "bsig": bsig,
        "step": 0,
        "counts": [0 for _ in range(B)],
        "loss_ema": [None for _ in range(B)],
    }


def _choose_block(state, args):
    B = state["B"]
    schedule = str(getattr(args, "dblock_schedule", "loss_balanced") or "loss_balanced").lower()
    step = int(state.get("step", 0))
    counts = state.setdefault("counts", [0 for _ in range(B)])
    emas = state.setdefault("loss_ema", [None for _ in range(B)])
    if schedule == "random":
        return random.randrange(B)
    if schedule == "roundrobin":
        return step % B
    explore = float(getattr(args, "dblock_explore", 0.05))
    warmup = int(getattr(args, "dblock_warmup_steps", max(8, B * 2)))
    if step < warmup or any(c == 0 for c in counts):
        return min(range(B), key=lambda i: (counts[i], i))
    if explore > 0.0 and random.random() < explore:
        return min(range(B), key=lambda i: (counts[i], i))
    return max(range(B), key=lambda i: (-1.0 if emas[i] is None else emas[i], -counts[i]))


def _sample_sigma(ids, lo, hi, args, state):
    cur_step = int(state.get("step", 0))
    curriculum = int(getattr(args, "dblock_sigma_curriculum_steps", 0))
    if curriculum > 0:
        frac = min(1.0, max(0.05, (cur_step + 1) / float(curriculum)))
        hi = lo * ((hi / max(lo, 1e-8)) ** frac)
    sig_np = np.exp(
        np.random.uniform(
            math.log(max(lo, 1e-4)),
            math.log(max(hi, lo + 1e-4)),
            ids.size(0),
        ).astype("float32")
    )
    return torch.from_numpy(sig_np).to(ids.device)


def _maybe_log(state, args, bi, layers, ar_val, sat_val, nat_val, total_val, peak_alloc, peak_reserved, objective=None):
    log_every = int(getattr(args, "dblock_log_every", 50))
    step = int(state.get("step", 0))
    if log_every <= 0 or step % log_every != 0:
        return
    counts = ",".join(str(x) for x in state.get("counts", []))
    emas = ",".join("nan" if x is None else f"{x:.2f}" for x in state.get("loss_ema", []))
    mem = ""
    if peak_alloc is not None:
        mem = f" peak_alloc={peak_alloc:.2f}GB peak_reserved={peak_reserved:.2f}GB"
    print(
        f"[dblock] step={step} block={bi} obj={objective or 'mixed'} layers={layers} "
        f"loss={total_val:.3f} ar={ar_val:.3f} sat={sat_val:.3f} nat={nat_val:.3f} "
        f"counts=[{counts}] ema=[{emas}]{mem}",
        flush=True,
    )


def _update_stats(state, bi, loss_value):
    B = state["B"]
    counts = state.setdefault("counts", [0 for _ in range(B)])
    emas = state.setdefault("loss_ema", [None for _ in range(B)])
    counts[bi] += 1
    prev = emas[bi]
    beta = 0.96
    emas[bi] = float(loss_value) if prev is None else beta * float(prev) + (1.0 - beta) * float(loss_value)
    state["step"] = int(state.get("step", 0)) + 1


def _run_block(block, x, mask, use_checkpoint):
    if use_checkpoint:
        return _ck.checkpoint(lambda y, block=block: block(y, mask), x, use_reentrant=False)
    return block(x, mask)


def _dblock_checkpoint_this_layer(args, base_enabled, layer_pos):
    if not base_enabled:
        return False
    stride = int(getattr(args, "dblock_checkpoint_stride", 1) or 1)
    if stride <= 0:
        return False
    if stride == 1:
        return True
    return (int(layer_pos) % stride) == 0


def _sample_token_loss_inputs(hidden, targets, max_tokens):
    max_tokens = int(max_tokens or 0)
    if max_tokens <= 0:
        return hidden.contiguous(), targets.contiguous(), int(targets.numel()), int(targets.numel())
    flat_targets = targets.reshape(-1)
    total = int(flat_targets.numel())
    if total <= max_tokens:
        return hidden.contiguous(), targets.contiguous(), total, total
    # With-replacement sampling avoids building a full randperm each step; the sampled
    # mean remains an unbiased estimator of the dense token CE mean.
    idx = torch.randint(total, (max_tokens,), device=targets.device)
    flat_hidden = hidden.reshape(total, hidden.size(-1))
    return flat_hidden.index_select(0, idx).contiguous(), flat_targets.index_select(0, idx).contiguous(), int(max_tokens), total


def _choose_objectives(state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic):
    mode = str(getattr(args, "dblock_objective_mode", "periodic") or "periodic").lower()
    if mode != "stochastic":
        return ar_weight > 0.0, sat_weight > 0.0 and do_sat_periodic, nat_weight > 0.0 and do_nat_periodic, "periodic"
    choices = []
    probs = []
    if ar_weight > 0.0:
        choices.append("ar")
        probs.append(max(0.0, float(getattr(args, "dblock_ar_prob", 0.80))))
    if sat_weight > 0.0 and not getattr(args, "ar_only", False):
        choices.append("sat")
        probs.append(max(0.0, float(getattr(args, "dblock_sat_prob", 0.10))))
    if nat_weight > 0.0 and not getattr(args, "ar_only", False):
        choices.append("nat")
        probs.append(max(0.0, float(getattr(args, "dblock_nat_prob", 0.10))))
    if not choices:
        return False, False, False, "none"
    total = sum(probs)
    if total <= 0.0:
        probs = [1.0 / len(choices) for _ in choices]
    else:
        probs = [p / total for p in probs]
    picked = random.choices(choices, weights=probs, k=1)[0]
    return picked == "ar", picked == "sat", picked == "nat", picked


def _dblock_step(core, ar_h, sat_h, nat_h, opt, scaler, args, ids, state):
    import nB300_agillm4 as M

    prof = _profile_active(state, args)
    _step_t = _profile_tic(prof)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _setup_t = _profile_tic(prof)
    B = state["B"]
    asg = state["assign"]
    bs = state["bsig"]
    T = ids.size(1)
    use_layer_checkpoint = bool(getattr(args, "grad_checkpoint", False))
    bi = _choose_block(state, args)
    lo, hi = sorted([bs[bi], bs[bi + 1]])
    layers = asg[bi]
    sig = _sample_sigma(ids, lo, hi, args, state)
    cs, co, ci = _edm_pre(sig)
    w = _edm_w(sig, float(getattr(args, "dblock_edm_wmax", 5.0)))
    SATB = M.SAT_BLOCK
    ar_weight = float(getattr(args, "dblock_ar_weight", 1.0))
    sat_weight = float(getattr(args, "dblock_sat_weight", 1.0))
    nat_weight = float(getattr(args, "dblock_nat_weight", 1.0)) * float(getattr(args, "nat_loss_weight", 1.0))
    do_sat_periodic = (not getattr(args, "ar_only", False)) and (
        int(getattr(args, "sat_every", 1)) <= 1 or ((int(state.get("step", 0)) + 1) % int(getattr(args, "sat_every", 1)) == 0)
    )
    do_nat_periodic = (
        nat_h is not None
        and (not getattr(args, "ar_only", False))
        and int(getattr(args, "nat_every", 1)) > 0
        and (
            int(getattr(args, "nat_every", 1)) <= 1
            or ((int(state.get("step", 0)) + 1) % int(getattr(args, "nat_every", 1)) == 0)
        )
    )
    run_ar, run_sat, run_nat, objective = _choose_objectives(
        state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic
    )
    _profile_toc(state, "setup", _setup_t)

    ar_val = 0.0
    sat_val = 0.0
    nat_val = 0.0

    if run_ar:
        causal = M.causal_mask(T, structured=M.use_structured_masks(args))
        _t = _profile_tic(prof)
        with M.amp(args.amp):
            emb = core.emb(ids)
            zt = emb + sig[:, None, None] * torch.randn_like(emb)
            h = ci * zt
            for lpos, li in enumerate(layers):
                h = _run_block(core.blocks[li], h, causal, _dblock_checkpoint_this_layer(args, use_layer_checkpoint, lpos))
            Dn = core.ln(cs * zt + co * h)
        _profile_toc(state, "ar_forward", _t)
        _t = _profile_tic(prof)
        ar_hidden, ar_targets, ar_used, ar_total = _sample_token_loss_inputs(
            Dn[:, :-1], ids[:, 1:], int(getattr(args, "dblock_ar_loss_tokens", 0))
        )
        ar = ar_weight * w * fused_ce(ar_hidden, ar_h.proj.weight, ar_targets)
        ar_val = float(ar.detach())
        _profile_toc(state, "ar_ce", _t)
        _t = _profile_tic(prof)
        scaler.scale(ar).backward()
        _profile_toc(state, "ar_backward", _t)
        del causal, emb, zt, h, Dn, ar_hidden, ar_targets, ar, ar_used, ar_total

    if run_sat:
        smask = M.sat_mask(T, structured=M.use_structured_masks(args))
        _t = _profile_tic(prof)
        with M.amp(args.amp):
            emb2 = core.emb(ids)
            zt2 = emb2 + sig[:, None, None] * torch.randn_like(emb2)
            h2 = ci * zt2
            for lpos, li in enumerate(layers):
                h2 = _run_block(core.blocks[li], h2, smask, _dblock_checkpoint_this_layer(args, use_layer_checkpoint, lpos))
            Ds = core.ln(cs * zt2 + co * h2)
            last = Ds[:, -SATB:]
        _profile_toc(state, "sat_forward", _t)
        _t = _profile_tic(prof)
        sat_hidden, sat_targets, sat_used, sat_total = _sample_token_loss_inputs(
            last, ids[:, 1 : SATB + 1], int(getattr(args, "dblock_sat_loss_tokens", 0))
        )
        with M.amp(args.amp):
            satf = fused_ce(sat_hidden, sat_h.proj.weight, sat_targets)
            satv = (
                M.EMIT_LAMBDA
                * F.cross_entropy(
                    sat_h.gate(Ds[:, 0].float()),
                    torch.ones(ids.size(0), dtype=torch.long, device=ids.device),
                )
                if sat_h.gate is not None
                else 0.0
            )
            sat = sat_weight * w * (satf + satv)
        _profile_toc(state, "sat_ce", _t)
        sat_val = float(sat.detach())
        _t = _profile_tic(prof)
        scaler.scale(sat).backward()
        _profile_toc(state, "sat_backward", _t)
        del smask, emb2, zt2, h2, Ds, last, sat_hidden, sat_targets, satf, satv, sat

    if run_nat:
        ratio = min(max(float(getattr(args, "nat_mask_ratio", 0.5)), 0.05), 0.95)
        nat_ids = M._nat_ids_for_training(ids, int(getattr(args, "nat_max_tokens", 0)))
        _t = _profile_tic(prof)
        with M.amp(args.amp):
            nat_in = nat_ids.clone()
            m = torch.rand(nat_ids.shape, device=nat_ids.device) < ratio
            if not bool(m.any()):
                m[..., -1] = True
            nat_in[m] = M.BLANK
            hn = core.emb(nat_in)
            for lpos, li in enumerate(layers):
                hn = _run_block(core.blocks[li], hn, None, _dblock_checkpoint_this_layer(args, use_layer_checkpoint, lpos))
            Dnat = core.ln(hn)
        _profile_toc(state, "nat_forward", _t)
        _t = _profile_tic(prof)
        nat_hidden = Dnat[m]
        nat_targets = nat_ids[m]
        nat_hidden, nat_targets, nat_used, nat_total = _sample_token_loss_inputs(
            nat_hidden.unsqueeze(0), nat_targets.unsqueeze(0), int(getattr(args, "dblock_nat_loss_tokens", 0))
        )
        nat = nat_weight * fused_ce(nat_hidden, nat_h.proj.weight, nat_targets)
        nat_val = float(nat.detach())
        _profile_toc(state, "nat_ce", _t)
        _t = _profile_tic(prof)
        scaler.scale(nat).backward()
        _profile_toc(state, "nat_backward", _t)
        del nat_ids, nat_in, m, hn, Dnat, nat_hidden, nat_targets, nat, nat_used, nat_total

    total_val = ar_val + sat_val + nat_val
    if not math.isfinite(total_val):
        opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[dblock] non-finite loss {total_val}; skipped optimizer step", flush=True)
        _profile_toc(state, "step_total", _step_t)
        _profile_step_done(state, args)
        _update_stats(state, bi, total_val)
        return total_val

    _t = _profile_tic(prof)
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    _profile_toc(state, "opt_step", _t)

    peak_alloc = None
    peak_reserved = None
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
    _profile_toc(state, "step_total", _step_t)
    _profile_step_done(state, args)
    _update_stats(state, bi, total_val)
    _maybe_log(state, args, bi, layers, ar_val, sat_val, nat_val, total_val, peak_alloc, peak_reserved, objective=objective)
    return total_val
