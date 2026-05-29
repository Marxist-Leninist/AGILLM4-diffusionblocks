---
license: apache-2.0
tags:
- agillm
- diffusionblocks
- memory-efficient-training
- block-wise-training
---

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

## Official-line integration snapshot (2026-05-29)
- `nB300_agillm4_vram_dblock.py` is the patched official trainer snapshot from the Vast box.
- `dblocks_train.py` is the folded-in low-VRAM training step used by `--dblock`.
- `relaunch_agillm4_dblock.sh` restarts the official line with `--dblock --tie_weights --attn_backend sublinear`.
- `--tie_weights` now means AR, SAT, and NAT share the embedding projection tensor. This drops the live parameter count from 1,213,418,242 to 716,595,202.
- Old untied checkpoint head matrices are intentionally skipped under tied mode; core weights still warm-start and the optimizer can rebuild.
- SAT now uses fused vocab-streaming CE in the dblock path, and the dblock step releases AR/SAT activations before moving to the next objective.
- DBlock now uses loss-balanced block scheduling after warmup, per-block EMA diagnostics, sigma-range curriculum, objective weights, and peak VRAM logging.
- The folded-in DBlock path now builds the dense causal/SAT masks once per objective instead of once per layer, and NAT obeys `--nat_max_tokens` so long-context AR does not force full-context NAT memory.
- Sublinear attention now supports structured causal, SAT block-causal, and unrestricted/NAT rules directly, and computes ALiBi only for gathered local/anchor candidates instead of allocating dense `[H x T x T]` bias.
- The trainer prints lightweight heartbeat lines and clears the CUDA cache after checkpoint load so reserved VRAM does not stay inflated by transient load tensors.

## Honest findings
- DiffusionBlocks and gradient-checkpointing are **substitutes** for activation
  memory; with checkpointing on, the 28->7 layer saving is only ~1.1-1.3x.
- The big fixed-cost win was **tied heads + fused CE** (ctx-1280 ~24 GB -> ~5.5 GB).
- The long-context ceiling (16k+) is **attention-bound**, so the next lever is the
  sublinear attention backend, not more block/CE work.

Status update 2026-05-29: Scott chose the VRAM-first route. The official AGILLM-4
training line is now the DiffusionBlocks mode folded into `nB300_agillm4.py`, with
`--dblock --tie_weights --attn_backend sublinear`. Checkpoint compatibility is best-effort
only: old untied AR/SAT/NAT head tensors are skipped when tied heads are active, and the
optimizer state is allowed to reset. The priority is lower VRAM over preserving every
old training assumption.

Upgrade update 2026-05-29: DBlock is no longer just a random-block prototype. The live
path now has loss-balanced scheduling, sigma curriculum, DBlock objective weights,
per-block loss/VRAM logging, single-build masks per objective, and NAT token capping.
These are meant to preserve the VRAM breakthrough while making block-wise training
less brittle over long runs.

Structured-mask update 2026-05-29: the sublinear backend now accepts symbolic causal,
SAT block-causal, and unrestricted/NAT mask rules. This removes dense O(T^2) mask
allocation for long context, and also gathers ALiBi bias directly for selected
local/anchor keys instead of materializing dense `[heads x T x T]` bias tensors.
A trainer heartbeat, post-checkpoint CUDA cache clear, and optional `--empty_cache_every_steps` hook were added for easier long-running Vast monitoring and VRAM-first allocator behavior.

Speed update 2026-05-29: the live Vast line now uses algorithmic speedups rather
than only hardware-style knobs: stochastic DBlock objective sampling (one sampled
AR/SAT/NAT objective per step), sampled token-level CE for the large vocab head,
and a tighter structured-sublinear attention profile (`window=128`, `stride=128`,
`max_anchors=128`). The first stable live window reached about 2.49k tok/s with
an ETA around 326 days, under the 1y+90d target, while keeping ctx=1280, B=2,
DiffusionBlocks, gradient-checkpointed blocks, tied heads, and structured masks.

Sublinear coverage update 2026-05-29: the saved AGILLM-4 trainer snapshot now matches the live v2 sparse global memory path. It fixes gathered ALiBi distance, suppresses duplicate local/anchor candidates before softmax, uses hybrid full-span + recent-tail anchors with explicit `--sublinear_sinks` and `--sublinear_recent_anchors`, and includes optional pooled K/V landmark summaries behind `--sublinear_pooled_landmarks`. At the live 128/128/128 profile it keeps deep-past coverage while preserving recent anchors and the same VRAM-first key budget. See `sublinear_improved_snippet.py` for the minimal blocks and `sublinear_improved.py` for the coverage demo/standalone selector.

License: Apache-2.0 (matching the upstream method).
