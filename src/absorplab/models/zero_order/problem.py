from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

from .model import SUPPORTED_VARIABLES, TEMPERATURES


@dataclass
class Problem:
    """Definition of one problem instance.

    Known values and unknown names are intentionally data, not solver logic. This
    is the principal volatility boundary: a new homework/design case should
    normally require changing only a Problem instance.
    """

    known: Mapping[str, float]
    unknowns: Sequence[str]
    temperature_unit: str = "C"
    initial_guesses: Mapping[str, float] = field(default_factory=dict)
    bounds: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.known = dict(self.known)
        self.unknowns = tuple(self.unknowns)
        self.initial_guesses = dict(self.initial_guesses)
        self.bounds = dict(self.bounds)

        unit = self.temperature_unit.upper()
        if unit not in {"C", "K"}:
            raise ValueError("temperature_unit must be 'C' or 'K'.")
        self.temperature_unit = unit

        if not 1 <= len(self.unknowns) <= 6:
            raise ValueError("The six-equation model requires between 1 and 6 unknowns.")
        if len(set(self.unknowns)) != len(self.unknowns):
            raise ValueError("unknowns contains duplicate variable names.")

        names = set(self.known) | set(self.unknowns)
        unsupported = names - SUPPORTED_VARIABLES
        if unsupported:
            raise ValueError(f"Unsupported variable(s): {sorted(unsupported)}")

        overlap = set(self.known) & set(self.unknowns)
        if overlap:
            raise ValueError(f"Variables cannot be both known and unknown: {sorted(overlap)}")

        for name in self.initial_guesses:
            if name not in self.unknowns:
                raise ValueError(f"Initial guess supplied for non-unknown variable {name!r}.")
        for name in self.bounds:
            if name not in self.unknowns:
                raise ValueError(f"Bounds supplied for non-unknown variable {name!r}.")

    def known_internal(self) -> Dict[str, float]:
        values = dict(self.known)
        if self.temperature_unit == "C":
            for name in TEMPERATURES:
                if name in values:
                    values[name] = values[name] + 273.15
        return values

    def to_internal_value(self, name: str, value: float) -> float:
        if name in TEMPERATURES and self.temperature_unit == "C":
            return value + 273.15
        return value

    def from_internal_value(self, name: str, value: float) -> float:
        if name in TEMPERATURES and self.temperature_unit == "C":
            return value - 273.15
        return value
