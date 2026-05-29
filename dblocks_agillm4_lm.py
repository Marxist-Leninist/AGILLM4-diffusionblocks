"""AR + SAT(fixed+variable) + NAT DiffusionBlocks for AGILLM-4 (v3, long-context).

Upgrades over v2:
  * SAT fixed (proj CE over SAT_BLOCK, block-causal sat_mask) AND variable (gate CE)
  * tied shared vocab projection (emb.weight) across AR/SAT/NAT -> drops ~0.5B params
  * chunked cut-CE: logits computed per token-chunk on a detached hidden, backward
    incrementally, then ONE block backward -> [T x 129280] never fully materialized
  * sequential per-objective backward (one block-forward graph live at a time)
  * AMP bf16 + clamped EDM weight (stable)
"""
import sys, math, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as _ck
from fused_ce import fused_ce
sys.path.insert(0,"/workspace/agillm-4"); import nB300_agillm4 as M
V=M.VOCAB; SATB=M.SAT_BLOCK; EMIT=getattr(M,"EMIT_LAMBDA",0.1); SD=0.5
def _cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def _ppf(p): return float(torch.erfinv(torch.tensor(2*p-1.0))*math.sqrt(2))
def bsig(B,smin=0.002,smax=80.0,pm=-1.2,ps=1.2):
    a,b=_cdf((math.log(smin)-pm)/ps),_cdf((math.log(smax)-pm)/ps)
    return [float(np.exp(pm+ps*_ppf(a+(b-a)*(i/B)))) for i in range(B+1)]
def edm_pre(s): s=s[:,None,None]; return SD**2/(s**2+SD**2), s*SD/(s**2+SD**2)**0.5, 1/(s**2+SD**2)**0.5
def edm_w(s,wmax=5.0): return float(((s**2+SD**2)/(s*SD)**2).clamp(max=wmax).mean())

def make(B):
    cfg=M.PRESETS["agillm4_floor"].copy()
    core=M.Encoder(cfg, attn_backend="sdpa", grad_checkpoint=False).to(M.DEV)
    sat_gate=nn.Linear(cfg["d"],2).to(M.DEV)          # SAT *variable* gate (tiny)
    L=cfg["layers"]; sp=L//B
    asg=[list(range(i*sp,(i+1)*sp)) for i in range(B)]; asg[-1]=list(range((B-1)*sp,L))
    return core,sat_gate,asg,bsig(B)

def runblk(core,h,layers,mask):
    for li in layers: h=_ck.checkpoint(core.blocks[li], h, mask, use_reentrant=False)  # recompute in backward
    return h

def cut_ce_backward(core, hidden, targets, scale, chunk=1024, mask_pos=None):
    """chunked CE on detached hidden -> grad; caller backprops block once."""
    hd=hidden.detach().requires_grad_(True); T=hd.size(1)
    if mask_pos is None: N=targets.numel()
    else: N=int(mask_pos.sum().item()) or 1
    tot=0.0
    for s in range(0,T,chunk):
        lg=F.linear(hd[:,s:s+chunk], core.emb.weight).float()      # tied proj, transient
        tg=targets[:,s:s+chunk]
        if mask_pos is None:
            l=F.cross_entropy(lg.reshape(-1,V),tg.reshape(-1),reduction="sum")*scale/N
        else:
            mm=mask_pos[:,s:s+chunk]
            if not bool(mm.any()): continue
            l=F.cross_entropy(lg[mm],tg[mm],reduction="sum")*scale/N
        l.backward(); tot+=float(l.detach())
    return tot, hd.grad

def step(core,sat_gate,ids,layers,lo,hi,nat_ratio=0.5,bw=True):
    T=ids.size(1); ar=sat=nat=0.0
    sig=torch.from_numpy(np.exp(np.random.uniform(math.log(lo),math.log(hi),ids.size(0))).astype("float32")).to(M.DEV)
    cs,co,ci=edm_pre(sig); w=edm_w(sig)
    # ---- AR: causal diffusion denoise ----
    with torch.autocast("cuda",dtype=torch.bfloat16):
        emb=core.emb(ids); zt=emb+sig[:,None,None]*torch.randn_like(emb)
        Dn=core.ln(cs*zt+co*runblk(core,ci*zt,layers,M.causal_mask(T)))
    ar_loss=w*fused_ce(Dn[:,:-1].contiguous(), core.emb.weight, ids[:,1:].contiguous())
    if bw: ar_loss.backward()
    ar=float(ar_loss.detach())
    # ---- SAT: block-causal diffusion; fixed proj + variable gate ----
    with torch.autocast("cuda",dtype=torch.bfloat16):
        zt2=core.emb(ids)+sig[:,None,None]*torch.randn_like(emb)
        Ds=core.ln(cs*zt2+co*runblk(core,ci*zt2,layers,M.sat_mask(T)))
        last=Ds[:,-SATB:]
        satf=F.cross_entropy(F.linear(last,core.emb.weight).float().reshape(-1,V), ids[:,1:SATB+1].reshape(-1))
        satv=EMIT*F.cross_entropy(sat_gate(Ds[:,0].float()), torch.ones(ids.size(0),dtype=torch.long,device=M.DEV))
        sat_loss=w*(satf+satv)
    if bw: sat_loss.backward()
    sat=float(sat_loss)
    # ---- NAT: bidirectional mask-predict ----
    with torch.autocast("cuda",dtype=torch.bfloat16):
        nat_ids=ids.clone(); m=torch.rand(ids.shape,device=M.DEV)<nat_ratio
        if not bool(m.any()): m[...,-1]=True
        nat_ids[m]=M.BLANK
        Dnat=core.ln(runblk(core,core.emb(nat_ids),layers,None))
    nat_loss=fused_ce(Dnat[m], core.emb.weight, ids[m])
    if bw: nat_loss.backward()
    nat=float(nat_loss.detach())
    return ar,sat,nat

def peak(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()/1e9

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--blocks",type=int,default=4); a=ap.parse_args()
    core,sg,asg,bs=make(a.blocks); nL=len(core.blocks); bi=a.blocks//2
    P=sum(p.numel() for p in core.parameters())+sum(p.numel() for p in sg.parameters())
    print(f"AGILLM-4 floor {nL}L, {P/1e9:.2f}B params (tied heads) | blocks={a.blocks} assign={asg}")
    def full(ctx): step(core,sg,torch.randint(0,V,(1,ctx),device=M.DEV),list(range(nL)),bs[0],bs[-1]); core.zero_grad(set_to_none=True); sg.zero_grad(set_to_none=True)
    def blk(ctx):  step(core,sg,torch.randint(0,V,(1,ctx),device=M.DEV),asg[bi],bs[bi],bs[bi+1]); core.zero_grad(set_to_none=True); sg.zero_grad(set_to_none=True)
    print("\\n=== v3: tied-heads + cut-CE + AMP + SAT(fixed+var)+NAT ===")
    for ctx in (1280,4096,8192,16384):
        try: mf=peak(lambda:full(ctx)); sf=f"{mf:.2f} GB"
        except RuntimeError as e: torch.cuda.empty_cache(); sf=("OOM" if "out of memory" in str(e).lower() else "ERR")
        try: mb=peak(lambda:blk(ctx)); sb=f"{mb:.2f} GB"
        except RuntimeError as e: torch.cuda.empty_cache(); sb=("OOM" if "out of memory" in str(e).lower() else "ERR")
        print(f"  ctx={ctx:>6}: full(28L)={sf:>9}   one block({len(asg[bi])}L)={sb:>9}")
    print("\\n--- smoke train (block-wise AR+SAT(fixed+var)+NAT) ---")
    for st in range(4):
        b=np.random.randint(a.blocks)
        ar,sat,nat=step(core,sg,torch.randint(0,V,(1,1280),device=M.DEV),asg[b],bs[b],bs[b+1])
        opt=torch.optim.SGD([p for li in asg[b] for p in core.blocks[li].parameters()]+list(sg.parameters()),1e-3)
        opt.step(); opt.zero_grad(); core.zero_grad(set_to_none=True)
        print(f"  step {st} block {b}: AR={ar:.2f} SAT={sat:.2f} NAT={nat:.2f}")
