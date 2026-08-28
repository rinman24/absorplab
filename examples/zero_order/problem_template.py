"""Copy this file for each new zero-order absorption problem."""

from absorplab.models.zero_order import AbsorptionSolver, Problem


problem = Problem(
    known={
        # Temperatures (degC here because temperature_unit="C")
        # "T_h": ...,
        # "T_c": ...,
        # "T_e": ...,
        #
        # Heat rates, if known:
        # "Q_h": ...,
        # "Q_c": ...,
        # "Q_e": ...,
        #
        # For EACH heat exchanger, choose ONE representation:
        # "UA_h": ...,
        # OR "R_h": ...,
        # OR both "U_h": ... and "A_h": ...,
        #
        # Repeat for c and e sides.
    },
    unknowns=[
        # Put 1 to 6 unknown variable names here.
        # Example: "T_hi", "T_ci", "T_ei", "Q_h", "Q_c", "Q_e"
    ],
    temperature_unit="C",
    initial_guesses={
        # Strongly recommended for nonlinear multi-unknown cases.
        # "T_hi": ...,
    },
    bounds={
        # Optional overrides. Example:
        # "Q_e": (0.0, None),
    },
)

solution = AbsorptionSolver().solve(problem)

print("Success:", solution.success)
print("Message:", solution.message)
print("Jacobian rank:", solution.jacobian_rank)
print("Max scaled residual:", solution.scaled_residual_norm)
print()

for name in problem.unknowns:
    print(f"{name:8s} = {solution[name]:.10g}")

if "Q_h" in solution.values and "Q_e" in solution.values:
    print(f"COP      = {solution.cop:.10g}")
