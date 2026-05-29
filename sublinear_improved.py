"""Improved sublinear-attention anchor selection for AGILLM-4.

AGILLM-4's `sublinear` attention = local sliding window + strided "landmark"
anchors. The original capped the anchor set with `anchors[-max_anchors:]`, which
DROPS the entire deep past once N > max_anchors*stride (the trainer goes blind to
everything older than the recent tail). This patch keeps whole-sequence coverage at
the SAME key budget, plus StreamingLLM-style attention sinks.

Drop-in replacement for the anchor-cap block inside MHA._sublinear_attention.
"""
import torch

def select_anchors(k_len, stride, max_anchors, sinks, device):
    """Even-coverage strided landmarks over the FULL past + attention sinks."""
    start = stride - 1
    if stride > 0 and max_anchors > 0 and start < k_len:
        anchors = torch.arange(start, k_len, stride, device=device, dtype=torch.long)
        if anchors.numel() > max_anchors:
            # even-coverage subsample across the whole sequence (NOT the recent tail)
            sel = torch.linspace(0, anchors.numel() - 1, max_anchors, device=device).round().long().unique()
            anchors = anchors[sel]
    else:
        anchors = torch.empty(0, device=device, dtype=torch.long)
    if sinks > 0 and k_len > 0:                       # always keep the first few tokens
        anchors = torch.cat([torch.arange(min(sinks, k_len), device=device, dtype=torch.long), anchors]).unique()
    return anchors

# --- exact patch applied to nB300_agillm4.py MHA._sublinear_attention ---
PATCH = '''
# replace:
#     if anchors.numel() > self.sublinear_max_anchors:
#         anchors = anchors[-self.sublinear_max_anchors :]
# with:
    if anchors.numel() > self.sublinear_max_anchors:
        _sel = torch.linspace(0, anchors.numel() - 1, self.sublinear_max_anchors, device=device).round().long().unique()
        anchors = anchors[_sel]
# ... and after the else branch add:
_sink = int(getattr(self, "sublinear_sinks", 4))
if _sink > 0 and k_len > 0:
    anchors = torch.cat([torch.arange(min(_sink, k_len), device=device, dtype=torch.long), anchors]).unique()
'''

if __name__ == "__main__":
    # Structural coverage demo at the live config, N beyond max_anchors*stride.
    N, W, stride, maxA, sinks = 32768, 128, 128, 128, 4
    i = N - 1
    loc = set(range(i - W, i + 1))
    allA = list(range(stride - 1, i + 1, stride))
    OLD = sorted(loc | set(allA[-maxA:]))
    sel = torch.linspace(0, len(allA) - 1, maxA).round().long().tolist()
    NEW = sorted(loc | {allA[s] for s in sel} | set(range(sinks)))
    print(f"N={N} (cap bites: {len(allA)} anchors > {maxA})")
    print(f"OLD covers {min(OLD)}..{max(OLD)}  -> blind to 0..{min(OLD)-1}; first-half keys={sum(x<N//2 for x in OLD)}")
    print(f"NEW covers {min(NEW)}..{max(NEW)}  -> full span; first-half keys={sum(x<N//2 for x in NEW)}")
