from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from absorplab.analysis import SweepResult, sweep
from absorplab.models.zero_order import AbsorptionSolver, Problem


THI_C = 90.0
TCI_C = 60.0
TEI_C = 30.0
TH_C = 100.0
TC_C = 40.0
TE_C = 40.0
QH = 100.0
QE = QH * ((TEI_C + 273.15) / (THI_C + 273.15))
QC = QH + QE
UA_H = QH / (TH_C - THI_C)
UA_C = QC / (TCI_C - TC_C)
UA_E = QE / (TE_C - TEI_C)


def zero_order_problem() -> Problem:
    return Problem(
        known={
            "T_h": TH_C,
            "T_c": TC_C,
            "T_e": TE_C,
            "UA_h": UA_H,
            "UA_c": UA_C,
            "UA_e": UA_E,
        },
        unknowns=["T_hi", "T_ci", "T_ei", "Q_h", "Q_c", "Q_e"],
        temperature_unit="C",
        initial_guesses={
            "T_hi": THI_C,
            "T_ci": TCI_C,
            "T_ei": TEI_C,
            "Q_h": QH,
            "Q_c": QC,
            "Q_e": QE,
        },
        bounds={"Q_h": (1.0, None)},
    )


@dataclass
class ToyProblem:
    known: Mapping[str, float]
    unknowns: Sequence[str]
    initial_guesses: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.known = dict(self.known)
        self.unknowns = tuple(self.unknowns)
        self.initial_guesses = dict(self.initial_guesses)


@dataclass(frozen=True)
class ToySolution:
    values: Mapping[str, float]
    success: bool
    message: str


class CapturingToySolver:
    def __init__(
        self,
        failing_values: set[float] | None = None,
        soft_failing_values: set[float] | None = None,
    ) -> None:
        self.failing_values = set() if failing_values is None else failing_values
        self.soft_failing_values = set() if soft_failing_values is None else soft_failing_values
        self.solved_values: list[float] = []
        self.initial_guesses_seen: list[dict[str, float]] = []

    def solve(self, problem: ToyProblem, *, residual_tolerance: float = 1e-8) -> ToySolution:
        value = float(problem.known["x"])
        self.solved_values.append(value)
        self.initial_guesses_seen.append(dict(problem.initial_guesses))
        if value in self.failing_values:
            raise ValueError(f"x={value} is outside the toy model domain")
        if value in self.soft_failing_values:
            return ToySolution(
                values={"x": value, "y": value * 10.0},
                success=False,
                message=f"x={value} did not converge",
            )
        return ToySolution(
            values={"x": value, "y": value * 10.0},
            success=True,
            message=f"solved at tolerance {residual_tolerance}",
        )


class NoCallToySolver:
    def solve(self, problem: ToyProblem, *, residual_tolerance: float = 1e-8) -> ToySolution:
        raise AssertionError("invalid sweep configuration should fail before solving")


class SweepTests(unittest.TestCase):
    def assertClose(self, actual: float, expected: float, tol: float = 1e-7) -> None:
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=tol, abs_tol=tol),
            (actual, expected),
        )

    def assertSweepMatchesManualSolve(
        self,
        results: SweepResult,
        values: Sequence[float],
        *,
        continuation: bool,
    ) -> None:
        solver = AbsorptionSolver()
        previous_success: dict[str, float] | None = None
        base = zero_order_problem()

        for index, value in enumerate(values):
            initial_guesses = dict(base.initial_guesses)
            if continuation and previous_success is not None:
                initial_guesses.update(previous_success)
            manual_problem = Problem(
                known={**base.known, "UA_e": value},
                unknowns=base.unknowns,
                temperature_unit=base.temperature_unit,
                initial_guesses=initial_guesses,
                bounds=base.bounds,
            )
            manual = solver.solve(manual_problem)
            self.assertTrue(manual.success, manual.message)

            for name in ["T_hi", "T_ci", "T_ei", "Q_h", "Q_c", "Q_e"]:
                self.assertClose(results[name][index], manual.values[name])
            self.assertClose(results["COP"][index], manual.cop)

            previous_success = {
                name: manual.values[name]
                for name in base.unknowns
                if name in manual.values
            }

    def test_basic_zero_order_sweep(self) -> None:
        values = [5.0, UA_E, 10.0]

        results = sweep(
            problem=zero_order_problem(),
            parameter="UA_e",
            values=values,
            solver=AbsorptionSolver(),
        )

        self.assertEqual(len(results), len(values))
        self.assertEqual(results["UA_e"], tuple(values))
        self.assertEqual(results.successes, (True, True, True))
        self.assertIn("Q_e", results.column_names)
        self.assertIn("COP", results.column_names)
        self.assertEqual(len(results["Q_e"]), len(values))

    def test_sweep_numerically_matches_manual_solves(self) -> None:
        values = [5.0, UA_E, 10.0]

        results = sweep(
            problem=zero_order_problem(),
            parameter="UA_e",
            values=values,
            solver=AbsorptionSolver(),
        )

        self.assertSweepMatchesManualSolve(results, values, continuation=False)

    def test_sweep_does_not_mutate_original_problem(self) -> None:
        problem = zero_order_problem()
        known_before = dict(problem.known)
        unknowns_before = tuple(problem.unknowns)
        guesses_before = dict(problem.initial_guesses)
        bounds_before = dict(problem.bounds)
        unit_before = problem.temperature_unit

        sweep(
            problem=problem,
            parameter="UA_e",
            values=[5.0, 10.0],
            solver=AbsorptionSolver(),
            continuation=True,
        )

        self.assertEqual(problem.known, known_before)
        self.assertEqual(problem.unknowns, unknowns_before)
        self.assertEqual(problem.initial_guesses, guesses_before)
        self.assertEqual(problem.bounds, bounds_before)
        self.assertEqual(problem.temperature_unit, unit_before)

    def test_failed_point_is_recorded_and_later_points_are_attempted(self) -> None:
        problem = ToyProblem(known={"x": 0.0}, unknowns=["y"], initial_guesses={"y": 1.0})
        solver = CapturingToySolver(failing_values={2.0})

        results = sweep(
            problem=problem,
            parameter="x",
            values=[1.0, 2.0, 3.0],
            solver=solver,
        )

        self.assertEqual(solver.solved_values, [1.0, 2.0, 3.0])
        self.assertEqual(results["x"], (1.0, 2.0, 3.0))
        self.assertEqual(results.successes, (True, False, True))
        self.assertTrue(math.isnan(results["y"][1]))
        self.assertIn("outside the toy model domain", results.messages[1])
        self.assertClose(results["y"][2], 30.0)

    def test_unsuccessful_solver_result_is_recorded_as_failed_point(self) -> None:
        problem = ToyProblem(known={"x": 0.0}, unknowns=["y"], initial_guesses={"y": 1.0})
        solver = CapturingToySolver(soft_failing_values={2.0})

        results = sweep(
            problem=problem,
            parameter="x",
            values=[1.0, 2.0, 3.0],
            solver=solver,
        )

        self.assertEqual(solver.solved_values, [1.0, 2.0, 3.0])
        self.assertEqual(results.successes, (True, False, True))
        self.assertTrue(math.isnan(results["y"][1]))
        self.assertIn("did not converge", results.messages[1])
        self.assertClose(results["y"][2], 30.0)

    def test_invalid_parameter_fails_before_solving(self) -> None:
        problem = ToyProblem(known={"x": 0.0}, unknowns=["y"], initial_guesses={"y": 1.0})

        with self.assertRaisesRegex(ValueError, "known inputs"):
            sweep(
                problem=problem,
                parameter="z",
                values=[1.0],
                solver=NoCallToySolver(),
            )

    def test_continuation_uses_previous_successful_solution(self) -> None:
        problem = ToyProblem(known={"x": 0.0}, unknowns=["y"], initial_guesses={"y": 1.0})
        solver = CapturingToySolver(failing_values={2.0})

        results = sweep(
            problem=problem,
            parameter="x",
            values=[1.0, 2.0, 3.0],
            solver=solver,
            continuation=True,
        )

        self.assertEqual(results.successes, (True, False, True))
        self.assertEqual(
            solver.initial_guesses_seen,
            [
                {"y": 1.0},
                {"y": 10.0},
                {"y": 10.0},
            ],
        )

    def test_continuation_can_be_disabled(self) -> None:
        problem = ToyProblem(known={"x": 0.0}, unknowns=["y"], initial_guesses={"y": 1.0})
        solver = CapturingToySolver()

        results = sweep(
            problem=problem,
            parameter="x",
            values=[1.0, 2.0],
            solver=solver,
            continuation=False,
        )

        self.assertEqual(results.successes, (True, True))
        self.assertEqual(solver.initial_guesses_seen, [{"y": 1.0}, {"y": 1.0}])

    def test_zero_order_continuation_remains_numerically_equivalent(self) -> None:
        values = [5.0, UA_E, 10.0]

        results = sweep(
            problem=zero_order_problem(),
            parameter="UA_e",
            values=values,
            solver=AbsorptionSolver(),
            continuation=True,
        )

        self.assertSweepMatchesManualSolve(results, values, continuation=True)

    def test_result_access(self) -> None:
        problem = ToyProblem(known={"x": 0.0}, unknowns=["y"], initial_guesses={"y": 1.0})

        results = sweep(
            problem=problem,
            parameter="x",
            values=[1.0, 2.0],
            solver=CapturingToySolver(),
        )

        self.assertEqual(results["x"], (1.0, 2.0))
        self.assertEqual(results["y"], (10.0, 20.0))
        self.assertEqual(
            results.to_rows(),
            [
                {
                    "x": 1.0,
                    "y": 10.0,
                    "success": True,
                    "message": "solved at tolerance 1e-08",
                },
                {
                    "x": 2.0,
                    "y": 20.0,
                    "success": True,
                    "message": "solved at tolerance 1e-08",
                },
            ],
        )
        with self.assertRaises(KeyError):
            results["not_a_column"]


if __name__ == "__main__":
    unittest.main()
