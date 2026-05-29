"""Fused cross-entropy: streams over the VOCAB dimension (online-softmax) so the
[N x V] logit matrix is NEVER materialized -- only [N x vchunk]. Custom backward
recomputes softmax per vocab-chunk (grad = softmax - onehot). This is the
DiffusionBlocks 'process in chunks, don't hold the whole thing' idea applied to
the output head instead of network depth."""
import torch

class FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, tgt, vchunk=16384):
        with torch.cuda.amp.autocast(enabled=False):
            hf = h.float()
            Wf = W.float()
            N, d = h.shape
            V = W.shape[0]
            m = torch.full((N,), -1e30, device=h.device, dtype=torch.float32)
            s = torch.zeros(N, device=h.device, dtype=torch.float32)
            zt = torch.zeros(N, device=h.device, dtype=torch.float32)
            for c in range(0, V, vchunk):
                lg = hf @ Wf[c:c+vchunk].T                    # [N,vchunk] transient only
                cm = lg.max(1).values
                nm = torch.maximum(m, cm)
                s = s * torch.exp(m - nm) + torch.exp(lg - nm[:, None]).sum(1)
                m = nm
                ic = (tgt >= c) & (tgt < c+vchunk)
                if ic.any():
                    zt[ic] = lg[ic, tgt[ic] - c].float()
            lse = m + torch.log(s)
            ctx.save_for_backward(h, W, tgt, lse)
            ctx.vchunk = vchunk
            return (lse - zt).mean()

    @staticmethod
    def backward(ctx, go):
        h, W, tgt, lse = ctx.saved_tensors
        vc = ctx.vchunk
        N, d = h.shape
        V = W.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            hf = h.float()
            Wc_all = W.float()
            gh = torch.zeros_like(hf)
            gW = torch.zeros(W.shape, device=W.device, dtype=torch.float32)
            sc = float(go) / N
            for c in range(0, V, vc):
                Wc = Wc_all[c:c+vc]
                p = torch.exp(hf @ Wc.T - lse[:, None])     # softmax chunk [N,vchunk]
                ic = (tgt >= c) & (tgt < c+vc)
                if ic.any():
                    p[ic, tgt[ic] - c] -= 1.0
                p *= sc
                gh += p @ Wc
                gW[c:c+vc] += p.T @ hf
            return gh.to(h.dtype), gW.to(W.dtype), None, None

def fused_ce(h, W, tgt, vchunk=16384):
    return FusedCE.apply(h.reshape(-1, h.size(-1)), W, tgt.reshape(-1), vchunk)
