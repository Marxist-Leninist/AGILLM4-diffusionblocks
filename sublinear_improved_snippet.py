"""Improved AGILLM-4 sublinear attention anchor selection.

Drop this block into `_sublinear_attention` in place of the recent-tail anchor
selection. It keeps the same local-window + anchor key budget shape, but avoids
losing the deep past once `num_anchors > sublinear_max_anchors`.
"""

anchor_start = self.sublinear_stride - 1
if self.sublinear_stride > 0 and self.sublinear_max_anchors > 0 and anchor_start < k_len:
    anchors = torch.arange(
        anchor_start,
        k_len,
        self.sublinear_stride,
        device=device,
        dtype=torch.long,
    )
    if anchors.numel() > self.sublinear_max_anchors:
        # Span the whole sequence instead of keeping only the recent tail.
        sel = torch.linspace(
            0,
            anchors.numel() - 1,
            self.sublinear_max_anchors,
            device=device,
        ).round().long().unique()
        anchors = anchors[sel]
else:
    anchors = torch.empty(0, device=device, dtype=torch.long)

# StreamingLLM-style attention sinks: preserve the first tokens as stable global memory.
sink = int(getattr(self, "sublinear_sinks", 4))
if sink > 0 and k_len > 0:
    anchors = torch.cat([
        torch.arange(min(sink, k_len), device=device, dtype=torch.long),
        anchors,
    ]).unique()
