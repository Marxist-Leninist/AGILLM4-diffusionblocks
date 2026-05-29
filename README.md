# AGILLM4-DiffusionBlocks

Block-wise (DiffusionBlocks-style) training adapted to the **AGILLM-4** decoder-only
LLM, with all three AGILLM-4 heads (AR + SAT fixed/variable + NAT) and a set of
long-context memory upgrades. Adapts & improves on
[SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks) (ICLR 2026),
whose released code is ViT/classification only.

## What's here
- `dblocks_agillm4.py` — self-contained DiffusionBlocks prototype (proves the
  per-step 1/B gradient locality on a small transformer).
- `dblocks_agillm4_lm.py` — real-architecture trainer: reuses AGILLM-4's `Encoder`
  blocks (ALiBi, causal `Block`), partitions 28 layers into B EDM-noise blocks,
  trains one block per step as an EDM denoiser, supervised by:
    * **AR** — causal next-token CE
    * **SAT** — fixed (proj CE over SAT_BLOCK, block-causal mask) **and** variable (gate CE)
    * **NAT** — bidirectional mask-predict CE
  Memory upgrades: EDM-weight clamp, AMP bf16, **tied shared vocab projection**
  (1.21B -> 0.72B params), grad-checkpointed blocks, and `fused_ce`.
- `fused_ce.py` — fused cross-entropy that streams over the 129k vocab via
  online-softmax with a custom backward, never materializing the `[T x 129280]`
  logit matrix (the DiffusionBlocks "process in chunks" idea applied to the head).

## Measured (real AGILLM-4 floor, 28L, 0.72B tied)
| ctx | full (28L) | one block (7L) |
|----:|-----------:|---------------:|
| 1280 | 7.48 GB | 5.53 GB |
| 4096 | 11.08 GB | 8.68 GB |
| 8192 | 22.11 GB | 19.37 GB |

## Honest findings
- DiffusionBlocks and gradient-checkpointing are **substitutes** for activation
  memory; with checkpointing on, the 28->7 layer saving is only ~1.1-1.3x.
- The big fixed-cost win was **tied heads + fused CE** (ctx-1280 ~24 GB -> ~5.5 GB).
- The long-context ceiling (16k+) is **attention-bound**, so the next lever is the
  sublinear attention backend, not more block/CE work.

Status: validated research prototype. The official AGILLM-4 training line remains the
proven AR/SAT/NAT end-to-end trainer; these wins (tied heads, fused-CE, sublinear
attention) are intended to be folded into it for long context.

License: Apache-2.0 (matching the upstream method).
