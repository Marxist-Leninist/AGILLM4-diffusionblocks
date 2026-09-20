/-
Closed-form ridge sublayer target for AGILLM DiffusionBlocks.

Transformer subproblem:
  freeze a micro-window of input activations X and target residual deltas Y;
  solve the best linear sublayer update W in closed form:

    argmin_W ||X W - Y||_F^2 + λ ||W - W₀||_F^2

Matrix normal equation:
    (XᵀX + λI) W = XᵀY + λW₀

This file keeps the Lean target deliberately tiny: prove the scalar ridge
normal equation and objective expansion first. The matrix theorem is the same
statement lifted column-wise over finite indices, which is the next proof task.
Yes, the whole transformer is not going to roll over and confess a closed form;
this sublayer does, because linear algebra still has some dignity left.
-/

import Mathlib

namespace AGILLM
namespace ClosedFormRidgeSublayer

noncomputable section

open scoped BigOperators

/-- One-coordinate ridge objective for a frozen linearized sublayer. -/
def scalarObjective (x y w0 lam w : ℝ) : ℝ :=
  (x * w - y)^2 + lam * (w - w0)^2

/-- Closed-form one-coordinate ridge solution. -/
def scalarSolution (x y w0 lam : ℝ) : ℝ :=
  (x * y + lam * w0) / (x * x + lam)

/-- The scalar closed-form solution satisfies the normal equation. -/
theorem scalar_solution_normal_equation
    (x y w0 lam : ℝ)
    (hden : x * x + lam ≠ 0) :
    (x * x + lam) * scalarSolution x y w0 lam = x * y + lam * w0 := by
  unfold scalarSolution
  field_simp [hden]

/-- Expanding the scalar ridge objective gives the quadratic normal-equation form. -/
theorem scalar_objective_expansion
    (x y w0 lam w : ℝ) :
    scalarObjective x y w0 lam w =
      (x * x + lam) * w^2 - 2 * (x * y + lam * w0) * w +
        (y^2 + lam * w0^2) := by
  unfold scalarObjective
  ring

/-- Column-wise matrix statement to finish in Lean.

For finite row index `r`, input index `i`, and output index `o`, let:
  X : r → i → ℝ
  Y : r → o → ℝ
  W₀ W : i → o → ℝ

If λ > 0 then `(XᵀX + λI)` is positive definite up to the ridge term and the
unique minimizer of `∑ r o (∑ i X r i * W i o - Y r o)^2 + λ∑ i o(W i o-W₀ i o)^2`
satisfies `(XᵀX + λI)W = XᵀY + λW₀`.

Implementation note: the trainer probe in `tools/closed_form_ridge_sublayer_probe.py`
uses the exact normal equation above, solved by Cholesky/solve with rejection gating.
-/

end

end ClosedFormRidgeSublayer
end AGILLM
