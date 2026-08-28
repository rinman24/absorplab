from absorplab.models.zero_order import AbsorptionSolver, Problem


# A self-consistent demonstration case. Temperatures are entered in degC.
# Each side uses a different allowed conductance representation to demonstrate
# that parameterization is isolated from the six-equation model.
problem = Problem(
    known={
        "T_h": 100.0,
        "T_c": 40.0,
        "T_e": 40.0,
        "UA_h": 10.0,
        "R_c": 0.10900495272399821,
        "U_e": 2.0,
        "A_e": 4.1738950846757535,
    },
    unknowns=["T_hi", "T_ci", "T_ei", "Q_h", "Q_c", "Q_e"],
    temperature_unit="C",
    initial_guesses={
        "T_hi": 90.0,
        "T_ci": 60.0,
        "T_ei": 30.0,
        "Q_h": 100.0,
        "Q_c": 183.0,
        "Q_e": 83.0,
    },
)

solution = AbsorptionSolver().solve(problem)

print("success:", solution.success)
print("message:", solution.message)
print("max scaled residual:", solution.scaled_residual_norm)
print("Jacobian rank:", solution.jacobian_rank)
print(f"COP = Q_e/Q_h = {solution.cop:.8f}")
print()
for name in problem.unknowns:
    print(f"{name:5s} = {solution[name]:.8g}")
