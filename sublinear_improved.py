"""Improved sublinear-attention anchor selection for AGILLM-4.

V2 used by the live AGILLM-4 DBlock line:
- fixes gathered ALiBi distance so past causal keys receive distance penalty
- suppresses local/anchor duplicate candidates before softmax
- uses hybrid full-span + recent-tail anchors under the same max-anchor budget
- exposes first-token attention sinks as `--sublinear_sinks`
- includes optional pooled landmark K/V summaries behind `--sublinear_pooled_landmarks`
"""
import torch


def select_hybrid_anchors(k_len, stride, max_anchors, sinks=4, recent_anchors=-1, device='cpu'):
    """Full-span + recent-tail landmark positions, plus attention sinks."""
    device = torch.device(device)
    start = stride - 1
    if stride <= 0 or max_anchors <= 0 or start >= k_len:
        anchors = torch.empty(0, device=device, dtype=torch.long)
    else:
        all_anchors = torch.arange(start, k_len, stride, device=device, dtype=torch.long)
        if all_anchors.numel() <= max_anchors:
            anchors = all_anchors
        else:
            if recent_anchors < 0:
                recent_anchors = max_anchors // 2
            recent_budget = min(max(0, int(recent_anchors)), max_anchors)
            span_budget = max(0, max_anchors - recent_budget)
            parts = []
            if span_budget > 0:
                sel = torch.linspace(0, all_anchors.numel() - 1, span_budget, device=device).round().long().unique()
                parts.append(all_anchors[sel])
            if recent_budget > 0:
                parts.append(all_anchors[-recent_budget:])
            anchors = torch.cat(parts).unique() if parts else torch.empty(0, device=device, dtype=torch.long)
    if sinks > 0 and k_len > 0:
        sink_idx = torch.arange(min(int(sinks), k_len), device=device, dtype=torch.long)
        anchors = torch.cat([sink_idx, anchors]).unique() if anchors.numel() else sink_idx
    return anchors


def local_anchor_valid(q_pos, anchors, window, k_len):
    """False where an anchor is already inside that query's local window."""
    anchor_idx = anchors.view(1, -1).expand(q_pos.numel(), -1)
    local_lo = (q_pos - window).clamp_min(0).view(-1, 1)
    local_hi = (q_pos + window).clamp_max(max(0, k_len - 1)).view(-1, 1)
    return (anchor_idx < local_lo) | (anchor_idx > local_hi)


if __name__ == '__main__':
    N, window, stride, maxA, sinks, recent = 32768, 128, 128, 128, 4, 64
    old_all = list(range(stride - 1, N, stride))
    old = sorted(set(range(N - window - 1, N)) | set(old_all[-maxA:]))
    new = select_hybrid_anchors(N, stride, maxA, sinks, recent).tolist()
    print(f'N={N} stride={stride} maxA={maxA} sinks={sinks} recent={recent}')
    print(f'OLD anchor coverage: {min(old)}..{max(old)} first_half={sum(x < N//2 for x in old)}')
    print(f'NEW anchor coverage: {min(new)}..{max(new)} first_half={sum(x < N//2 for x in new)} recent_tail={sum(x >= N-8192 for x in new)}')
    q = torch.tensor([N - 1])
    print(f'duplicate-suppressed anchors for final query: {int(local_anchor_valid(q, torch.tensor(new), window, N).sum())}/{len(new)}')
