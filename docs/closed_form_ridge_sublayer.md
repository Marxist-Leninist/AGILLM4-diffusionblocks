# Closed-form ridge sublayer experiment

## Picked transformer subproblem

Full transformer training has no useful end-to-end closed form because attention, softmax, layernorm, gating, residual recursion, and CE all couple together. The useful tractable cut is narrower:

> Freeze a micro-window of transformer activations and fit one linear residual sublayer exactly.

For an expert FFN down-projection, DBlock residual sublayer, or any `nn.Linear`-like map, collect:

- `X`: input activations into the chosen sublayer, shape `[tokens, din]`
- `Y`: target residual delta/output for that sublayer, shape `[tokens, dout]`
- `W0`: current weight in math orientation `[din, dout]`

Then solve:

```text
min_W ||XW - Y||_F^2 + λ||W - W0||_F^2
```

Closed form:

```text
W* = (XᵀX + λI)^(-1)(XᵀY + λW0)
```

This is the right level: it gives a real analytical solve for one frozen slice without lying that the whole transformer has turned into a friendly undergraduate exercise. Humanity survives another day.

## Lean status

File:

```text
formal/ClosedFormRidgeSublayer.lean
```

Current proof target:

- scalar ridge normal equation: implemented
- scalar objective expansion: implemented
- full finite matrix minimizer theorem: left as the next proof obligation, to be lifted column-wise over finite indices

Local note: this branch was authored from ChatGPT connector access, not from a machine with Lean installed, so run `lake env lean formal/ClosedFormRidgeSublayer.lean` on SG before marking it fully checked.

## Trainer application path

The guarded probe is:

```text
tools/closed_form_ridge_sublayer_probe.py
```

It is intentionally offline and rejection-gated. It will not mutate a live run. It only writes an updated checkpoint if:

1. a capture is supplied,
2. the closed-form solve finishes,
3. held-out MSE is no worse than the configured `--accept-ratio`, and
4. `--update-out` is explicitly provided.

Example capture hook for a `torch.nn.Linear` module:

```python
# Save math-orientation W0 = [din, dout]. PyTorch Linear weight is [dout, din].
torch.save({
    "X": x_in.detach().float().cpu(),
    "Y": y_target.detach().float().cpu(),
    "W0": module.weight.detach().float().cpu().t().contiguous(),
}, "/tmp/agillm_ridge_capture.pt")
```

Example dry-run solve:

```bash
python tools/closed_form_ridge_sublayer_probe.py \
  --capture /tmp/agillm_ridge_capture.pt \
  --l2 0.01 \
  --holdout-frac 0.10 \
  --max-rows 65536 \
  --report /tmp/agillm_ridge_report.json
```

Example checkpoint update, only if accepted:

```bash
python tools/closed_form_ridge_sublayer_probe.py \
  --capture /tmp/agillm_ridge_capture.pt \
  --checkpoint /path/to/checkpoint.pt \
  --weight-key core.blocks.12.ffn.down.weight \
  --orientation dout_din \
  --l2 0.01 \
  --accept-ratio 1.000 \
  --update-out /tmp/checkpoint.ridge-candidate.pt \
  --report /tmp/agillm_ridge_report.json
```

## SG MCP progress record

If SG MCP memory/coordination tooling is available on SG1/SG2, create or update this record:

```json
{
  "name": "transformer_closed_form_ridge_sublayer",
  "kind": "research_progress",
  "status": "branch_ready_not_live",
  "repo": "Marxist-Leninist/AGILLM4-diffusionblocks",
  "branch": "closed-form-ridge-sublayer",
  "subproblem": "Frozen-input ridge closed-form update for one transformer residual linear sublayer",
  "formula": "W=(X^T X + lambda I)^-1 (X^T Y + lambda W0)",
  "lean_file": "formal/ClosedFormRidgeSublayer.lean",
  "lean_status": "scalar normal equation + expansion authored; run Lean on SG; matrix minimizer proof pending",
  "trainer_tool": "tools/closed_form_ridge_sublayer_probe.py",
  "trainer_status": "offline guarded probe; no production default; rejects unless holdout improves/non-regresses",
  "next_actions": [
    "Run lake env lean formal/ClosedFormRidgeSublayer.lean on SG",
    "Capture X/Y/W0 for one MoE down-projection or DBlock residual sublayer",
    "Run the probe with lambda sweep: 1e-4, 1e-3, 1e-2, 1e-1",
    "Accept only if held-out MSE and downstream CE are neutral or better",
    "If accepted, add a trainer flag for periodic closed-form refresh behind default-off guard"
  ]
}
```

## Acceptance rule

Do **not** production-cutover from regression MSE alone. The promotion rule should be:

1. capture-level held-out MSE non-regression,
2. one-step downstream CE non-regression,
3. no SAT-var/NAT/AR shape or masking regression,
4. no exact-resume/checkpoint layout break,
5. measurable time-to-quality improvement or recovery from a bad sublayer.

The closed form is a repair/calibration primitive, not a replacement for SGD. Tiny slice, sharp knife.
