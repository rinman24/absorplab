from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


TEMPERATURES = ("T_h", "T_hi", "T_c", "T_ci", "T_e", "T_ei")
HEAT_RATES = ("Q_h", "Q_c", "Q_e")
UA_VALUES = ("UA_h", "UA_c", "UA_e")
UA_COMPONENTS = ("U_h", "A_h", "U_c", "A_c", "U_e", "A_e")
RESISTANCES = ("R_h", "R_c", "R_e")
SUPPORTED_VARIABLES = set(TEMPERATURES + HEAT_RATES + UA_VALUES + UA_COMPONENTS + RESISTANCES)


@dataclass(frozen=True)
class ResidualEvaluation:
    raw: np.ndarray
    scaled: np.ndarray


class ZeroOrderAbsorptionModel:
    """Six-equation zero-order absorption heat-pump model.

    Temperatures are absolute (K) internally. Each heat-exchanger conductance can
    independently be represented as UA, U*A, or 1/R.
    """

    residual_names = (
        "Eq1 hot-side heat transfer",
        "Eq2 condenser heat transfer",
        "Eq3 evaporator heat transfer",
        "Eq4 energy balance",
        "Eq5 reversible heat-transfer ratio",
        "Eq6 equal internal temperature spacing",
    )

    @staticmethod
    def _conductance(values: Mapping[str, float], side: str) -> float:
        ua_key = f"UA_{side}"
        r_key = f"R_{side}"
        u_key = f"U_{side}"
        a_key = f"A_{side}"

        has_ua = ua_key in values
        has_r = r_key in values
        has_u = u_key in values
        has_a = a_key in values

        representations = int(has_ua) + int(has_r) + int(has_u or has_a)
        if representations > 1:
            raise ValueError(
                f"For side {side!r}, choose exactly one conductance representation: "
                f"{ua_key}, {r_key}, or the pair ({u_key}, {a_key})."
            )

        if has_ua:
            return values[ua_key]
        if has_r:
            r = values[r_key]
            if r == 0:
                raise ValueError(f"{r_key} must be nonzero.")
            return 1.0 / r
        if has_u and has_a:
            return values[u_key] * values[a_key]
        if has_u != has_a:
            missing = a_key if has_u else u_key
            raise KeyError(f"{missing} is required with {u_key if has_u else a_key}.")

        raise KeyError(
            f"No {side}-side conductance supplied. Use {ua_key}, {r_key}, "
            f"or both {u_key} and {a_key}."
        )

    def residuals(self, values: Mapping[str, float]) -> ResidualEvaluation:
        gh = self._conductance(values, "h")
        gc = self._conductance(values, "c")
        ge = self._conductance(values, "e")

        qh, qc, qe = values["Q_h"], values["Q_c"], values["Q_e"]
        th, thi = values["T_h"], values["T_hi"]
        tc, tci = values["T_c"], values["T_ci"]
        te, tei = values["T_e"], values["T_ei"]

        r1 = qh - gh * (th - thi)
        r2 = qc - gc * (tci - tc)
        r3 = qe - ge * (te - tei)
        r4 = qh + qe - qc

        # Preserve the original Eq. 5 rather than cross-multiplying it. The
        # cross-multiplied form admits an extraneous root when T_ci == T_ei.
        denom_q = qh
        denom_t = tci - tei
        if abs(denom_q) < 1e-12 or abs(denom_t) < 1e-10:
            r5 = 1.0e6
        else:
            r5 = (qe / qh) - (tei / thi) * ((thi - tci) / denom_t)

        r6 = thi - 2.0 * tci + tei

        raw = np.array([r1, r2, r3, r4, r5, r6], dtype=float)

        q_scale = max(abs(qh), abs(qc), abs(qe), 1.0)
        t_scale = max(abs(th), abs(thi), abs(tc), abs(tci), abs(te), abs(tei), 1.0)
        scales = np.array(
            [q_scale, q_scale, q_scale, q_scale, 1.0, t_scale],
            dtype=float,
        )
        return ResidualEvaluation(raw=raw, scaled=raw / scales)
