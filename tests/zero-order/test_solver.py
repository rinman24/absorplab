import math
import unittest

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


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.solver = AbsorptionSolver()

    def assertClose(self, actual, expected, tol=1e-5):
        self.assertTrue(math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), (actual, expected))

    def test_six_unknowns_mixed_conductance_representations(self):
        p = Problem(
            known={
                "T_h": TH_C,
                "T_c": TC_C,
                "T_e": TE_C,
                "UA_h": UA_H,
                "R_c": 1.0 / UA_C,
                "U_e": 2.0,
                "A_e": UA_E / 2.0,
            },
            unknowns=["T_hi", "T_ci", "T_ei", "Q_h", "Q_c", "Q_e"],
            temperature_unit="C",
            initial_guesses={
                "T_hi": 88.0,
                "T_ci": 58.0,
                "T_ei": 28.0,
                "Q_h": 95.0,
                "Q_c": 180.0,
                "Q_e": 80.0,
            },
        )
        s = self.solver.solve(p)
        self.assertTrue(s.success, s.message)
        self.assertClose(s["T_hi"], THI_C)
        self.assertClose(s["T_ci"], TCI_C)
        self.assertClose(s["T_ei"], TEI_C)
        self.assertClose(s["Q_h"], QH)
        self.assertClose(s["Q_c"], QC)
        self.assertClose(s["Q_e"], QE)

    def test_single_unknown_ua(self):
        p = Problem(
            known={
                "T_h": TH_C, "T_hi": THI_C,
                "T_c": TC_C, "T_ci": TCI_C,
                "T_e": TE_C, "T_ei": TEI_C,
                "Q_h": QH, "Q_c": QC, "Q_e": QE,
                "UA_h": UA_H, "UA_c": UA_C,
            },
            unknowns=["UA_e"],
            temperature_unit="C",
            initial_guesses={"UA_e": 5.0},
        )
        s = self.solver.solve(p)
        self.assertTrue(s.success, s.message)
        self.assertClose(s["UA_e"], UA_E)

    def test_single_unknown_resistance(self):
        p = Problem(
            known={
                "T_h": TH_C, "T_hi": THI_C,
                "T_c": TC_C, "T_ci": TCI_C,
                "T_e": TE_C, "T_ei": TEI_C,
                "Q_h": QH, "Q_c": QC, "Q_e": QE,
                "UA_h": UA_H, "UA_c": UA_C,
            },
            unknowns=["R_e"],
            temperature_unit="C",
            initial_guesses={"R_e": 0.1},
        )
        s = self.solver.solve(p)
        self.assertTrue(s.success, s.message)
        self.assertClose(s["R_e"], 1.0 / UA_E)

    def test_unknown_u_with_known_area(self):
        area = 4.0
        p = Problem(
            known={
                "T_h": TH_C, "T_hi": THI_C,
                "T_c": TC_C, "T_ci": TCI_C,
                "T_e": TE_C, "T_ei": TEI_C,
                "Q_h": QH, "Q_c": QC, "Q_e": QE,
                "UA_h": UA_H, "UA_c": UA_C,
                "A_e": area,
            },
            unknowns=["U_e"],
            temperature_unit="C",
            initial_guesses={"U_e": 2.0},
        )
        s = self.solver.solve(p)
        self.assertTrue(s.success, s.message)
        self.assertClose(s["U_e"], UA_E / area)

    def test_rank_deficiency_when_u_and_area_both_unknown(self):
        p = Problem(
            known={
                "T_h": TH_C, "T_hi": THI_C,
                "T_c": TC_C, "T_ci": TCI_C,
                "T_e": TE_C, "T_ei": TEI_C,
                "Q_h": QH, "Q_c": QC, "Q_e": QE,
                "UA_h": UA_H, "UA_c": UA_C,
            },
            unknowns=["U_e", "A_e"],
            temperature_unit="C",
            initial_guesses={"U_e": 2.0, "A_e": UA_E / 2.0},
        )
        s = self.solver.solve(p)
        self.assertFalse(s.success)
        self.assertLess(s.jacobian_rank, 2)

    def test_kelvin_and_celsius_inputs_agree(self):
        known_c = {
            "T_h": TH_C, "T_c": TC_C, "T_e": TE_C,
            "UA_h": UA_H, "UA_c": UA_C, "UA_e": UA_E,
        }
        p_c = Problem(
            known=known_c,
            unknowns=["T_hi", "T_ci", "T_ei", "Q_h", "Q_c", "Q_e"],
            temperature_unit="C",
            initial_guesses={
                "T_hi": 90, "T_ci": 60, "T_ei": 30,
                "Q_h": 100, "Q_c": QC, "Q_e": QE,
            },
        )
        s_c = self.solver.solve(p_c)

        p_k = Problem(
            known={
                "T_h": TH_C + 273.15,
                "T_c": TC_C + 273.15,
                "T_e": TE_C + 273.15,
                "UA_h": UA_H, "UA_c": UA_C, "UA_e": UA_E,
            },
            unknowns=["T_hi", "T_ci", "T_ei", "Q_h", "Q_c", "Q_e"],
            temperature_unit="K",
            initial_guesses={
                "T_hi": THI_C + 273.15,
                "T_ci": TCI_C + 273.15,
                "T_ei": TEI_C + 273.15,
                "Q_h": QH, "Q_c": QC, "Q_e": QE,
            },
        )
        s_k = self.solver.solve(p_k)
        self.assertTrue(s_c.success, s_c.message)
        self.assertTrue(s_k.success, s_k.message)
        self.assertClose(s_c["T_hi"] + 273.15, s_k["T_hi"])
        self.assertClose(s_c.cop, s_k.cop)


if __name__ == "__main__":
    unittest.main()
