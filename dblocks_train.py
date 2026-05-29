"""DiffusionBlocks training mode folded into AGILLM-4 (gated by --dblock).

Block-wise EDM denoising on the real Encoder blocks, supervising AR + SAT(fixed+var)
+ NAT each step on ONE block, with grad-checkpointed layers and fused vocab-streaming
CE. Reuses the live data stream / optimizer / checkpointing of nB300_agillm4.
Lazy-imports nB300 inside functions to avoid a circular import.
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ck
from fused_ce import fused_ce

SD = 0.5


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


def _maybe_log(state, args, bi, layers, ar_val, sat_val, nat_val, total_val, peak_alloc, peak_reserved):
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
        f"[dblock] step={step} block={bi} layers={layers} "
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


def _dblock_step(core, ar_h, sat_h, nat_h, opt, scaler, args, ids, state):
    import nB300_agillm4 as M

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    B = state["B"]
    asg = state["assign"]
    bs = state["bsig"]
    T = ids.size(1)
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

    ar_val = 0.0
    sat_val = 0.0
    nat_val = 0.0

    if ar_weight > 0.0:
        causal = M.causal_mask(T)
        with M.amp(args.amp):
            emb = core.emb(ids)
            zt = emb + sig[:, None, None] * torch.randn_like(emb)
            h = ci * zt
            for li in layers:
                h = _ck.checkpoint(core.blocks[li], h, causal, use_reentrant=False)
            Dn = core.ln(cs * zt + co * h)
        ar = ar_weight * w * fused_ce(Dn[:, :-1].contiguous(), ar_h.proj.weight, ids[:, 1:].contiguous())
        ar_val = float(ar.detach())
        scaler.scale(ar).backward()
        del causal, emb, zt, h, Dn, ar

    do_sat = (not getattr(args, "ar_only", False)) and (
        int(getattr(args, "sat_every", 1)) <= 1 or ((int(state.get("step", 0)) + 1) % int(getattr(args, "sat_every", 1)) == 0)
    )
    if sat_weight > 0.0 and do_sat:
        smask = M.sat_mask(T)
        with M.amp(args.amp):
            emb2 = core.emb(ids)
            zt2 = emb2 + sig[:, None, None] * torch.randn_like(emb2)
            h2 = ci * zt2
            for li in layers:
                h2 = _ck.checkpoint(core.blocks[li], h2, smask, use_reentrant=False)
            Ds = core.ln(cs * zt2 + co * h2)
            last = Ds[:, -SATB:]
            satf = fused_ce(last.contiguous(), sat_h.proj.weight, ids[:, 1 : SATB + 1].contiguous())
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
        sat_val = float(sat.detach())
        scaler.scale(sat).backward()
        del smask, emb2, zt2, h2, Ds, last, satf, satv, sat

    do_nat = (
        nat_h is not None
        and nat_weight > 0.0
        and (not getattr(args, "ar_only", False))
        and int(getattr(args, "nat_every", 1)) > 0
        and (
            int(getattr(args, "nat_every", 1)) <= 1
            or ((int(state.get("step", 0)) + 1) % int(getattr(args, "nat_every", 1)) == 0)
        )
    )
    if do_nat:
        ratio = min(max(float(getattr(args, "nat_mask_ratio", 0.5)), 0.05), 0.95)
        nat_ids = M._nat_ids_for_training(ids, int(getattr(args, "nat_max_tokens", 0)))
        with M.amp(args.amp):
            nat_in = nat_ids.clone()
            m = torch.rand(nat_ids.shape, device=nat_ids.device) < ratio
            if not bool(m.any()):
                m[..., -1] = True
            nat_in[m] = M.BLANK
            hn = core.emb(nat_in)
            for li in layers:
                hn = _ck.checkpoint(core.blocks[li], hn, None, use_reentrant=False)
            Dnat = core.ln(hn)
        nat = nat_weight * fused_ce(Dnat[m], nat_h.proj.weight, nat_ids[m])
        nat_val = float(nat.detach())
        scaler.scale(nat).backward()
        del nat_ids, nat_in, m, hn, Dnat, nat

    total_val = ar_val + sat_val + nat_val
    if not math.isfinite(total_val):
        opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[dblock] non-finite loss {total_val}; skipped optimizer step", flush=True)
        _update_stats(state, bi, total_val)
        return total_val

    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)

    peak_alloc = None
    peak_reserved = None
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
    _update_stats(state, bi, total_val)
    _maybe_log(state, args, bi, layers, ar_val, sat_val, nat_val, total_val, peak_alloc, peak_reserved)
    return total_val
