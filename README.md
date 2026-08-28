# AbsorpLab

AbsorpLab provides reusable absorption-cycle model components and analysis tools.

## Parameter Sweeps

Use `absorplab.analysis.sweep` to repeatedly solve a model while varying one
known input parameter. The sweep layer is model-independent: callers provide the
model problem, the parameter values, and a solver object with a public
`solve(problem, residual_tolerance=...)` method.

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
    unknowns=[
        "T_hi",
        "T_ci",
        "T_ei",
        "Q_h",
        "Q_c",
        "Q_e",
    ],
    temperature_unit="C",
)

results = sweep(
    problem=problem,
    parameter="UA_e",
    values=np.linspace(2.0, 10.0, 20),
    solver=AbsorptionSolver(),
)

print(results["UA_e"])
print(results["COP"])
print(results["Q_e"])
```

`SweepResult` stores one `SweepPoint` per requested value and supports numeric
column access with `results["column_name"]`. The swept parameter is always
available as a column. Solver outputs are collected from successful points, and
derived `COP` is included when the model's solution exposes it.

Individual solve failures are recorded without aborting the sweep. Failed points
retain their parameter value, set `success` to `False`, store the diagnostic in
`message`, and expose missing numeric outputs as `NaN`.

```python
for row in results.to_rows():
    print(row["UA_e"], row["COP"], row["success"], row["message"])
```

Continuation can be enabled for nonlinear sweeps:

```python
results = sweep(
    problem=problem,
    parameter="UA_e",
    values=np.linspace(2.0, 10.0, 20),
    solver=AbsorptionSolver(),
    continuation=True,
)
```

With continuation enabled, the most recent successful solution supplies initial
guesses for the next sweep point's unknown variables. If a point fails, the last
successful continuation state is retained; if no point has succeeded yet, the
original problem guesses are used.
