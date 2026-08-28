from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass, replace
from types import MappingProxyType
from typing import Callable, Iterable, Iterator, Mapping, Protocol, Sequence, TypeVar, cast, runtime_checkable


P = TypeVar("P", bound="SweepProblem")

ProblemBuilder = Callable[[P, Mapping[str, float], Mapping[str, float]], P]
FailureExceptions = tuple[type[Exception], ...]


class SweepProblem(Protocol):
    """Problem data required by one-dimensional parameter sweeps."""

    known: Mapping[str, float]
    unknowns: Sequence[str]
    initial_guesses: Mapping[str, float]


class SolveResult(Protocol):
    """Minimal solve result contract consumed by the analysis layer."""

    values: Mapping[str, float]
    success: bool
    message: str


class Solver(Protocol[P]):
    """Solver contract used by model-independent analysis routines."""

    def solve(self, problem: P, *, residual_tolerance: float = 1e-8) -> SolveResult:
        """Solve one model problem."""


@runtime_checkable
class SupportsCop(Protocol):
    @property
    def cop(self) -> float:
        """Cooling coefficient of performance, when a model defines one."""


@dataclass(frozen=True)
class SweepPoint:
    """Result for one parameter value in a one-dimensional sweep."""

    parameter: str
    value: float
    outputs: Mapping[str, float]
    success: bool
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    def __getitem__(self, name: str) -> float:
        if name == self.parameter:
            return self.value
        return self.outputs[name]


SweepRowValue = float | bool | str


@dataclass(frozen=True)
class SweepResult:
    """Structured collection of one-dimensional parameter sweep results."""

    parameter: str
    points: Sequence[SweepPoint]
    _output_names: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        points = tuple(self.points)
        object.__setattr__(self, "points", points)

        output_names: list[str] = []
        seen: set[str] = set()
        for point in points:
            if point.parameter != self.parameter:
                raise ValueError("All sweep points must use the same swept parameter.")
            for name in point.outputs:
                if name not in seen and name != self.parameter:
                    seen.add(name)
                    output_names.append(name)

        object.__setattr__(self, "_output_names", tuple(output_names))

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[SweepPoint]:
        return iter(self.points)

    def __getitem__(self, name: str) -> tuple[float, ...]:
        if name == self.parameter:
            return tuple(point.value for point in self.points)
        if name not in self._output_names:
            raise KeyError(name)
        return tuple(point.outputs.get(name, math.nan) for point in self.points)

    @property
    def column_names(self) -> tuple[str, ...]:
        return (self.parameter, *self._output_names)

    @property
    def successes(self) -> tuple[bool, ...]:
        return tuple(point.success for point in self.points)

    @property
    def messages(self) -> tuple[str, ...]:
        return tuple(point.message for point in self.points)

    def to_rows(self) -> list[dict[str, SweepRowValue]]:
        rows: list[dict[str, SweepRowValue]] = []
        for point in self.points:
            row: dict[str, SweepRowValue] = {self.parameter: point.value}
            for name in self._output_names:
                row[name] = point.outputs.get(name, math.nan)
            row["success"] = point.success
            row["message"] = point.message
            rows.append(row)
        return rows


def sweep(
    *,
    problem: P,
    parameter: str,
    values: Iterable[float],
    solver: Solver[P],
    continuation: bool = False,
    residual_tolerance: float = 1e-8,
    problem_builder: ProblemBuilder[P] | None = None,
    failure_exceptions: FailureExceptions = (ValueError, RuntimeError, ArithmeticError),
) -> SweepResult:
    """Solve a model repeatedly while varying one known input parameter.

    Continuation uses the most recent successful solution as the next problem's
    initial guesses for variables listed in ``problem.unknowns``. If a point
    fails, the previous successful continuation state is retained; if there has
    not been a success yet, the original problem guesses are used.
    """
    sweep_values = _materialize_values(values)
    _validate_configuration(problem=problem, parameter=parameter, values=sweep_values)

    points: list[SweepPoint] = []
    base_initial_guesses = dict(problem.initial_guesses)
    continuation_guesses: dict[str, float] | None = None

    for value in sweep_values:
        known = dict(problem.known)
        known[parameter] = value

        initial_guesses = dict(base_initial_guesses)
        if continuation and continuation_guesses is not None:
            initial_guesses.update(continuation_guesses)

        point_problem = _build_problem(
            problem=problem,
            known=known,
            initial_guesses=initial_guesses,
            problem_builder=problem_builder,
        )

        try:
            solution = solver.solve(point_problem, residual_tolerance=residual_tolerance)
        except failure_exceptions as exc:
            points.append(_failed_point_from_exception(problem, parameter, value, exc))
            continue

        point = _point_from_solution(parameter=parameter, value=value, solution=solution)
        points.append(point)

        if continuation and solution.success:
            continuation_guesses = {
                name: float(solution.values[name])
                for name in problem.unknowns
                if name in solution.values
            }

    return SweepResult(parameter=parameter, points=tuple(points))


def _materialize_values(values: Iterable[float]) -> tuple[float, ...]:
    sweep_values = tuple(float(value) for value in values)
    if not sweep_values:
        raise ValueError("values must contain at least one sweep point.")

    invalid_values = [value for value in sweep_values if not math.isfinite(value)]
    if invalid_values:
        raise ValueError("values must contain only finite numeric sweep points.")

    return sweep_values


def _validate_configuration(
    *,
    problem: SweepProblem,
    parameter: str,
    values: Sequence[float],
) -> None:
    if not parameter:
        raise ValueError("parameter must be a non-empty variable name.")
    if parameter not in problem.known:
        raise ValueError(
            f"Cannot sweep {parameter!r}; one-dimensional sweeps currently vary known inputs."
        )
    if parameter in set(problem.unknowns):
        raise ValueError(f"Cannot sweep {parameter!r}; it is listed as an unknown variable.")
    if not values:
        raise ValueError("values must contain at least one sweep point.")


def _build_problem(
    *,
    problem: P,
    known: Mapping[str, float],
    initial_guesses: Mapping[str, float],
    problem_builder: ProblemBuilder[P] | None,
) -> P:
    if problem_builder is not None:
        return problem_builder(problem, known, initial_guesses)

    if not is_dataclass(problem):
        raise TypeError(
            "problem must be a dataclass instance or a problem_builder must be supplied."
        )

    return cast(P, replace(problem, known=known, initial_guesses=initial_guesses))


def _point_from_solution(*, parameter: str, value: float, solution: SolveResult) -> SweepPoint:
    outputs = _outputs_from_solution(solution)
    if not solution.success:
        outputs = {name: math.nan for name in outputs}

    return SweepPoint(
        parameter=parameter,
        value=value,
        outputs=outputs,
        success=solution.success,
        message=solution.message,
    )


def _outputs_from_solution(solution: SolveResult) -> dict[str, float]:
    outputs = {name: float(value) for name, value in solution.values.items()}
    if isinstance(solution, SupportsCop):
        try:
            outputs["COP"] = float(solution.cop)
        except (KeyError, ZeroDivisionError, ArithmeticError):
            pass
    return outputs


def _failed_point_from_exception(
    problem: SweepProblem,
    parameter: str,
    value: float,
    exc: Exception,
) -> SweepPoint:
    outputs = {name: math.nan for name in problem.unknowns}
    return SweepPoint(
        parameter=parameter,
        value=value,
        outputs=outputs,
        success=False,
        message=f"{type(exc).__name__}: {exc}",
    )
