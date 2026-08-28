from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.optimize import least_squares

from .model import TEMPERATURES, ZeroOrderAbsorptionModel
from .problem import Problem


_POSITIVE_PARAMETERS = {
    "Q_h", "Q_c", "Q_e",
    "UA_h", "UA_c", "UA_e",
    "U_h", "U_c", "U_e",
    "A_h", "A_c", "A_e",
    "R_h", "R_c", "R_e",
}


@dataclass(frozen=True)
class Solution:
    values: Dict[str, float]
    residuals: Dict[str, float]
    scaled_residual_norm: float
    jacobian_rank: int
    success: bool
    message: str

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    @property
    def cop(self) -> float:
        """Cooling COP implied by equation 5: Q_e / Q_h."""
        return self.values["Q_e"] / self.values["Q_h"]


class AbsorptionSolver:
    """Solve any well-specified case containing 1..6 unknowns."""

    def __init__(self, model: ZeroOrderAbsorptionModel | None = None):
        self.model = model or ZeroOrderAbsorptionModel()

    def _default_guess(self, name: str, known: Dict[str, float]) -> float:
        if name == "T_hi" and "T_h" in known:
            return known["T_h"] - 10.0
        if name == "T_ci" and "T_c" in known:
            return known["T_c"] + 10.0
        if name == "T_ei" and "T_e" in known:
            return known["T_e"] - 10.0
        if name in TEMPERATURES:
            known_t = [known[k] for k in TEMPERATURES if k in known]
            return float(np.mean(known_t)) if known_t else 300.0
        if name.startswith("Q_"):
            known_q = [abs(v) for k, v in known.items() if k.startswith("Q_")]
            return max(known_q, default=100.0)
        if name.startswith("R_"):
            return 0.1
        if name.startswith("UA_"):
            return 10.0
        if name.startswith("U_") or name.startswith("A_"):
            return 1.0
        return 1.0

    @staticmethod
    def _default_bounds(name: str) -> tuple[float, float]:
        if name in TEMPERATURES:
            return np.nextafter(0.0, 1.0), np.inf
        if name in _POSITIVE_PARAMETERS:
            return np.nextafter(0.0, 1.0), np.inf
        return -np.inf, np.inf

    def solve(self, problem: Problem, *, residual_tolerance: float = 1e-8) -> Solution:
        known = problem.known_internal()
        unknowns = list(problem.unknowns)

        x0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []

        for name in unknowns:
            guess_external = problem.initial_guesses.get(name)
            if guess_external is None:
                guess_internal = self._default_guess(name, known)
            else:
                guess_internal = problem.to_internal_value(name, float(guess_external))
            x0.append(float(guess_internal))

            if name in problem.bounds:
                lo, hi = problem.bounds[name]
                lo_i = -np.inf if lo is None else problem.to_internal_value(name, float(lo))
                hi_i = np.inf if hi is None else problem.to_internal_value(name, float(hi))
            else:
                lo_i, hi_i = self._default_bounds(name)
            lower.append(float(lo_i))
            upper.append(float(hi_i))

        x0_arr = np.asarray(x0, dtype=float)
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)

        if np.any(x0_arr <= lower_arr) or np.any(x0_arr >= upper_arr):
            raise ValueError(
                "An initial guess lies outside its bounds. Supply an explicit "
                "initial_guesses/bounds entry for that variable."
            )

        def assemble(x: np.ndarray) -> Dict[str, float]:
            values = dict(known)
            values.update(dict(zip(unknowns, x)))
            return values

        # Fail early with a useful error if a required variable or conductance
        # representation has not been supplied.
        try:
            self.model.residuals(assemble(x0_arr))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Problem specification is incomplete or inconsistent: {exc}") from exc

        def fun(x: np.ndarray) -> np.ndarray:
            return self.model.residuals(assemble(x)).scaled

        result = least_squares(
            fun,
            x0=x0_arr,
            bounds=(lower_arr, upper_arr),
            method="trf",
            x_scale="jac",
            max_nfev=5000,
        )

        values_internal = assemble(result.x)
        evaluation = self.model.residuals(values_internal)

        rank = int(np.linalg.matrix_rank(result.jac))
        n = len(unknowns)
        residual_norm = float(np.linalg.norm(evaluation.scaled, ord=np.inf))

        messages: list[str] = [result.message]
        success = bool(result.success)

        if rank < n:
            success = False
            messages.append(
                f"Jacobian rank is {rank}, below {n} unknowns; the solution is not locally unique."
            )
        if residual_norm > residual_tolerance:
            success = False
            messages.append(
                f"Maximum scaled residual {residual_norm:.3e} exceeds tolerance "
                f"{residual_tolerance:.3e}."
            )

        # Eq. 5 is undefined when T_ci == T_ei. Guard against a numerically
        # converged point at that singular boundary.
        internal_gap = abs(values_internal["T_ci"] - values_internal["T_ei"])
        if internal_gap < 1e-8:
            success = False
            messages.append(
                "T_ci and T_ei are effectively equal, where equation 5 is undefined."
            )

        output_values: Dict[str, float] = {}
        for name, value in values_internal.items():
            output_values[name] = problem.from_internal_value(name, float(value))

        residuals = {
            name: float(value)
            for name, value in zip(self.model.residual_names, evaluation.raw)
        }

        return Solution(
            values=output_values,
            residuals=residuals,
            scaled_residual_norm=residual_norm,
            jacobian_rank=rank,
            success=success,
            message=" ".join(messages),
        )
