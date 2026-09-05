# AbsorpLab

AbsorpLab provides reusable absorption-cycle model components and generic
analysis tools. The current implementation includes the six-equation zero-order
absorption model plus a model-independent one-dimensional parameter sweep.

## Setup

```bash
uv sync
uv run pytest
```

## Zero-order convenience API

```python
from absorplab.models.zero_order import solve

solution = solve(
    known={
        "T_h": 200.0,
        "T_c": 50.0,
        "T_e": -20.0,
        "UA_h": 1.35,
        "UA_c": 2.50,
        "UA_e": 1.14,
    },
    unknowns=["T_hi", "T_ci", "T_ei", "Q_h", "Q_c", "Q_e"],
    temperature_unit="C",
)

print(solution["Q_e"])
print(solution.cop)
```

When `temperature_unit="C"`, all temperature inputs, initial guesses, and
bounds are interpreted as degrees Celsius. Temperatures are converted to Kelvin
internally before the model equations are evaluated, then converted back to
Celsius in the returned solution. Use `temperature_unit="K"` when supplying
absolute temperatures directly.

For each temperature level, choose exactly one conductance representation:

- `UA_h`, `UA_c`, or `UA_e`
- `R_h`, `R_c`, or `R_e` where `UA = 1/R`
- a matching `U_*` and `A_*` pair where `UA = U*A`

## Parameter sweeps

```python
import numpy as np

from absorplab.analysis import sweep
from absorplab.models.zero_order import AbsorptionSolver, Problem

problem = Problem(
    known={
        "T_h": 120.0,
        "T_c": 30.0,
        "T_e": 10.0,
        "UA_h": 3.5,
        "UA_c": 4.2,
        "UA_e": 2.8,
    },
    unknowns=["T_hi", "T_ci", "T_ei", "Q_h", "Q_c", "Q_e"],
    temperature_unit="C",
)

results = sweep(
    problem=problem,
    parameter="UA_e",
    values=np.linspace(2.0, 10.0, 20),
    solver=AbsorptionSolver(),
    continuation=True,
)

print(results["UA_e"])
print(results["COP"])
print(results["Q_e"])
```

`SweepResult` stores one `SweepPoint` per requested value. Failed individual
points are retained with `success=False`, a diagnostic message, and `NaN`
numerical outputs rather than aborting the entire sweep.

With continuation enabled, the most recent successful solution supplies initial
guesses for the next sweep point.

## Repository layout

```text
src/absorplab/
├── analysis/
│   └── sweep.py
├── common/
└── models/
    └── zero_order/
        ├── model.py
        ├── problem.py
        └── solver.py

tests/
├── analysis/
└── zero_order/
```
