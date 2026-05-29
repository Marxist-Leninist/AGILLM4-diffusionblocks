"""DiffusionBlocks prototype for AGILLM-4 (block-wise denoising training).

Faithful to SakanaAI/DiffusionBlocks: EDM (Karras) diffusion, equi-probability
block sigma partitioning, and the key property -- each step trains ONE block as a
denoiser for its noise range, so backprop touches only that block => training
memory ~ 1/B of end-to-end.  Here we reuse a transformer-block stack (stand-in for
AGILLM-4's Encoder.blocks: same pre-norm residual Block API x = blk(x, mask)).
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
def _cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def _ppf(p): return float(torch.erfinv(torch.tensor(2*p-1.0))*math.sqrt(2))

# ---- EDM block partitioning (verbatim mechanism from dblock_modules.py) ----
def get_block_sigmas(num_blocks, sigma_min=0.002, sigma_max=80.0, p_mean=-1.2, p_std=1.2):
    cdf_min = _cdf((np.log(sigma_min)-p_mean)/p_std)
    cdf_max = _cdf((np.log(sigma_max)-p_mean)/p_std)
    out=[]
    for i in range(num_blocks+1):
        p = cdf_min + (cdf_max-cdf_min)*(i/num_blocks)
        out.append(float(np.exp(p_mean+p_std*_ppf(p))))
    return out   # descending->ascending sigma boundaries, equal CDF mass per block

class DBlockLM(nn.Module):
    """Shared token emb + output proj; n_layers split into B independent blocks."""
    def __init__(self, vocab=2048, d=256, n_layers=8, heads=4, num_blocks=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4*d,
                          batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, vocab)
        self.num_blocks = num_blocks
        split = n_layers // num_blocks
        self.assign = [list(range(i*split,(i+1)*split)) for i in range(num_blocks)]
        self.block_sigmas = get_block_sigmas(num_blocks)
        self.sigma_data = 0.5

    def run_block(self, b, zt, sigma):
        # EDM preconditioning (Karras 2022), exactly as DiffusionBlocks.denoise()
        s = sigma[:,None,None]
        c_skip = self.sigma_data**2/(s**2+self.sigma_data**2)
        c_out  = s*self.sigma_data/(s**2+self.sigma_data**2)**0.5
        c_in   = 1/(s**2+self.sigma_data**2)**0.5
        h = zt*c_in
        for li in self.assign[b]:          # <-- ONLY this block's layers run
            h = self.blocks[li](h)
        return c_skip*zt + c_out*h         # denoiser D_theta(zt,sigma) -> predicts z0

def edm_weight(sigma, sigma_data=0.5):
    return (sigma**2+sigma_data**2)/(sigma*sigma_data)**2

def train_step(model, opt, ids):
    z0 = F.normalize(model.emb(ids), p=2, dim=-1).detach()      # clean target embeds
    b = np.random.randint(model.num_blocks)                     # pick one block
    lo, hi = model.block_sigmas[b], model.block_sigmas[b+1]
    lo, hi = min(lo,hi), max(lo,hi)
    sigma = torch.from_numpy(np.exp(np.random.uniform(np.log(lo),np.log(hi),
                              size=ids.shape[0])).astype('float32'))
    zt = z0 + sigma[:,None,None]*torch.randn_like(z0)
    D = model.run_block(b, zt, sigma)                           # backprop only block b
    loss = (edm_weight(sigma)[:,None,None]*(D-z0)**2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return b, loss.item()

if __name__ == "__main__":
    torch.manual_seed(0)
    NB=4; m=DBlockLM(num_blocks=NB); opt=torch.optim.AdamW(m.parameters(),1e-3)
    ids=torch.randint(0,2048,(8,64))
    print("layers:",len(m.blocks),"| blocks:",NB,"| layer assignment:",m.assign)
    print("block sigma boundaries:",[round(s,3) for s in m.block_sigmas])
    # one step, then verify ONLY the chosen block's layer params got gradients
    b,loss=train_step(m,opt,ids)
    def has_grad(mod): return any(p.grad is not None and p.grad.abs().sum()>0 for p in mod.parameters())
    grad_blocks=[i for i in range(NB) if any(has_grad(m.blocks[li]) for li in m.assign[i])]
    tot=sum(p.numel() for p in m.parameters())
    blk_params=sum(p.numel() for li in m.assign[b] for p in m.blocks[li].parameters())
    print(f"\\nstep trained block {b}  loss={loss:.4f}")
    print("blocks whose layers received gradients:",grad_blocks,"(expect just",[b],")")
    print(f"layer-params with grad this step: {blk_params:,} / {sum(p.numel() for blk in m.blocks for p in blk.parameters()):,}"
          f"  ({100*blk_params/sum(p.numel() for blk in m.blocks for p in blk.parameters()):.0f}% = ~1/{NB})")
    print("\\n--- a few more steps (each trains one block independently) ---")
    for _ in range(6):
        b,loss=train_step(m,opt,ids); print(f"  block {b}: loss={loss:.4f}")
