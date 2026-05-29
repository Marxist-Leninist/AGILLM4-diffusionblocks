"""DiffusionBlocks training mode folded into AGILLM-4 (gated by --dblock).

Block-wise EDM denoising on the real Encoder blocks, supervising AR + SAT(fixed+var)
+ NAT each step on ONE block, with grad-checkpointed layers and fused vocab-streaming
CE. Reuses the live data stream / optimizer / checkpointing of nB300_agillm4.
Lazy-imports nB300 inside functions to avoid a circular import.
"""
import math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as _ck
from fused_ce import fused_ce
SD=0.5
def _cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def _ppf(p): return float(torch.erfinv(torch.tensor(2*p-1.0))*math.sqrt(2))
def _block_sigmas(B,smin=0.002,smax=80.0,pm=-1.2,ps=1.2):
    a,b=_cdf((math.log(smin)-pm)/ps),_cdf((math.log(smax)-pm)/ps)
    return [float(np.exp(pm+ps*_ppf(a+(b-a)*(i/B)))) for i in range(B+1)]
def _edm_pre(s): s=s[:,None,None]; return SD**2/(s**2+SD**2), s*SD/(s**2+SD**2)**0.5, 1/(s**2+SD**2)**0.5
def _edm_w(s,wmax=5.0): return float(((s**2+SD**2)/(s*SD)**2).clamp(max=wmax).mean())

def _dblock_init(core, args):
    B=int(getattr(args,"dblock_blocks",4)); L=len(core.blocks); sp=max(1,L//B)
    asg=[list(range(i*sp,(i+1)*sp)) for i in range(B)]; asg[-1]=list(range((B-1)*sp,L))
    print(f"[dblock] DiffusionBlocks mode: {L} layers -> {B} blocks {asg}")
    print(f"[dblock] equi-prob sigma boundaries: {[round(x,3) for x in _block_sigmas(B)]}")
    return {"B":B,"assign":asg,"bsig":_block_sigmas(B)}

def _dblock_step(core, ar_h, sat_h, nat_h, opt, scaler, args, ids, state):
    import nB300_agillm4 as M
    B=state["B"]; asg=state["assign"]; bs=state["bsig"]; T=ids.size(1)
    bi=random.randrange(B); lo,hi=sorted([bs[bi],bs[bi+1]]); layers=asg[bi]
    sig=torch.from_numpy(np.exp(np.random.uniform(math.log(max(lo,1e-4)),math.log(hi),ids.size(0))).astype("float32")).to(ids.device)
    cs,co,ci=_edm_pre(sig); w=_edm_w(sig); SATB=M.SAT_BLOCK
    # ---- AR: causal diffusion denoise ----
    with M.amp(args.amp):
        emb=core.emb(ids); zt=emb+sig[:,None,None]*torch.randn_like(emb); h=ci*zt
        for li in layers: h=_ck.checkpoint(core.blocks[li], h, M.causal_mask(T), use_reentrant=False)
        Dn=core.ln(cs*zt+co*h)
    ar=w*fused_ce(Dn[:,:-1].contiguous(), ar_h.proj.weight, ids[:,1:].contiguous())
    scaler.scale(ar).backward()
    ar_val=float(ar.detach())
    del emb, zt, h, Dn, ar
    # ---- SAT: block-causal diffusion; fixed proj + variable gate ----
    with M.amp(args.amp):
        emb2=core.emb(ids); zt2=emb2+sig[:,None,None]*torch.randn_like(emb2); h2=ci*zt2
        for li in layers: h2=_ck.checkpoint(core.blocks[li], h2, M.sat_mask(T), use_reentrant=False)
        Ds=core.ln(cs*zt2+co*h2); last=Ds[:,-SATB:]
        satf=fused_ce(last.contiguous(), sat_h.proj.weight, ids[:,1:SATB+1].contiguous())
        satv=(M.EMIT_LAMBDA*F.cross_entropy(sat_h.gate(Ds[:,0].float()), torch.ones(ids.size(0),dtype=torch.long,device=ids.device))) if sat_h.gate is not None else 0.0
        sat=w*(satf+satv)
    scaler.scale(sat).backward()
    sat_val=float(sat.detach())
    del emb2, zt2, h2, Ds, last, satf, satv, sat
    # ---- NAT: bidirectional mask-predict ----
    nat_val=0.0
    if nat_h is not None:
        ratio=min(max(float(getattr(args,"nat_mask_ratio",0.5)),0.05),0.95)
        with M.amp(args.amp):
            nat_ids=ids.clone(); m=torch.rand(ids.shape,device=ids.device)<ratio
            if not bool(m.any()): m[...,-1]=True
            nat_ids[m]=M.BLANK; hn=core.emb(nat_ids)
            for li in layers: hn=_ck.checkpoint(core.blocks[li], hn, None, use_reentrant=False)
            Dnat=core.ln(hn)
        nat=fused_ce(Dnat[m], nat_h.proj.weight, ids[m]); scaler.scale(nat).backward(); nat_val=float(nat.detach()); del nat_ids, m, hn, Dnat, nat
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    return ar_val+sat_val+nat_val
