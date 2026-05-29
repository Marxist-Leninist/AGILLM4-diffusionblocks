"""Minimal AGILLM-4 sublinear attention V2 snippets.

These are the core blocks now folded into `nB300_agillm4_vram_dblock.py`.
"""

# Anchor selection: full-span + recent-tail + sinks.
anchors = self._sublinear_anchor_positions(k_len, device)

# Optional pooled landmarks are available behind --sublinear_pooled_landmarks.
if anchors.numel() and self.sublinear_pooled_landmarks and self.sublinear_stride > 1:
    ends = anchors + 1
    starts = (ends - self.sublinear_stride).clamp_min(0)
    zero_k = k.new_zeros(k.size(0), k.size(1), 1, k.size(3))
    zero_v = v.new_zeros(v.size(0), v.size(1), 1, v.size(3))
    prefix_k = torch.cat([zero_k, k.cumsum(dim=2)], dim=2)
    prefix_v = torch.cat([zero_v, v.cumsum(dim=2)], dim=2)
    denom = (ends - starts).to(dtype=k.dtype).view(1, 1, -1, 1).clamp_min(1)
    anchor_k = (prefix_k[:, :, ends, :] - prefix_k[:, :, starts, :]) / denom
    anchor_v = (prefix_v[:, :, ends, :] - prefix_v[:, :, starts, :]) / denom

# Duplicate suppression: do not let an anchor double-count a key already in local attention.
anchor_idx = anchors.view(1, -1).expand(cur, -1)
local_lo = (q_pos - self.sublinear_window).clamp_min(0).view(-1, 1)
local_hi = (q_pos + self.sublinear_window).clamp_max(max(0, k_len - 1)).view(-1, 1)
anchor_valid = (anchor_idx < local_lo) | (anchor_idx > local_hi)

# Gathered ALiBi distance: distance from query to selected key, not future-only clamp.
dist = (q_pos.view(1, 1, cur, 1) - idx.view(1, 1, cur, -1)).abs().to(torch.float32)
scores = scores + (-slopes * dist).to(scores.dtype)
