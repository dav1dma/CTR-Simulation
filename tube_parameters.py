
"""
Supervisor-provided CTR parameters.

Tube order:
0 = innermost
1 = middle
2 = outermost

IMPORTANT:
The supervisor message labels pre-curvature as 1/mm. The inherited
superPosKin() model expects kappa_0 in 1/m, and the numerical values
19.12 and 14.04 match the earlier geometry only if interpreted as 1/m.
Therefore this file uses [0, 19.12, 14.04] 1/m in the model.
"""

import numpy as np

TUBE_DATA = {
    "inner": {
        "material": "hobby-grade Nitinol",
        "od_mm": 0.50,
        "id_mm": 0.00,
        "E_range_GPa": (40.0, 75.0),
        "total_length_mm": 350.0,
        "curved_length_mm": 0.0,
        "precurvature_per_m": 0.0,
    },
    "middle": {
        "material": "PIERTECH",
        "od_mm": 0.70,
        "id_mm": 0.62,
        "E_range_GPa": (60.0, 83.0),
        "total_length_mm": 170.0,
        "curved_length_mm": 90.0,
        "precurvature_per_m": 19.12,
    },
    "outer": {
        "material": "PIERTECH",
        "od_mm": 0.90,
        "id_mm": 0.80,
        "E_range_GPa": (60.0, 83.0),
        "total_length_mm": 80.0,
        "curved_length_mm": 65.0,
        "precurvature_per_m": 14.04,
    },
}


def mm_to_m(x):
    return x * 1e-3


def straight_length_mm(tube):
    return tube["total_length_mm"] - tube["curved_length_mm"]


def build_supervisor_ctr_parameters():
    inner = TUBE_DATA["inner"]
    middle = TUBE_DATA["middle"]
    outer = TUBE_DATA["outer"]

    # Nominal model value:
    # 75 GPa is within both supplied E ranges and matches the inherited model.
    E_nominal = 75e9
    G_nominal = E_nominal / 3.0

    return {
        "n_t": 3,

        # [straight length, curved length] in metres
        "l_t": [
            [mm_to_m(straight_length_mm(inner)),  mm_to_m(inner["curved_length_mm"])],
            [mm_to_m(straight_length_mm(middle)), mm_to_m(middle["curved_length_mm"])],
            [mm_to_m(straight_length_mm(outer)),  mm_to_m(outer["curved_length_mm"])],
        ],

        "E": [E_nominal, E_nominal, E_nominal],
        "G": [G_nominal, G_nominal, G_nominal],

        # inherited model expects 1/m
        "kappa_0": [
            inner["precurvature_per_m"],
            middle["precurvature_per_m"],
            outer["precurvature_per_m"],
        ],

        # [inner radius, outer radius] in metres
        "r": [
            [mm_to_m(inner["id_mm"] / 2.0),  mm_to_m(inner["od_mm"] / 2.0)],
            [mm_to_m(middle["id_mm"] / 2.0), mm_to_m(middle["od_mm"] / 2.0)],
            [mm_to_m(outer["id_mm"] / 2.0),  mm_to_m(outer["od_mm"] / 2.0)],
        ],
    }


def total_tube_lengths(CTR_par=None):
    if CTR_par is None:
        CTR_par = build_supervisor_ctr_parameters()
    return np.array([a + b for a, b in CTR_par["l_t"]], dtype=float)


def nominal_clearances_mm():
    inner = TUBE_DATA["inner"]
    middle = TUBE_DATA["middle"]
    outer = TUBE_DATA["outer"]

    d12 = middle["id_mm"] - inner["od_mm"]
    d23 = outer["id_mm"] - middle["od_mm"]

    return {
        "inner_to_middle_diametral_clearance_mm": d12,
        "inner_to_middle_radial_clearance_mm": d12 / 2.0,
        "middle_to_outer_diametral_clearance_mm": d23,
        "middle_to_outer_radial_clearance_mm": d23 / 2.0,
    }


if __name__ == "__main__":
    p = build_supervisor_ctr_parameters()
    print("l_t [m]:", p["l_t"])
    print("E [GPa]:", [e/1e9 for e in p["E"]])
    print("kappa_0 [1/m]:", p["kappa_0"])
    print("r [m]:", p["r"])
    print("total lengths [mm]:", total_tube_lengths(p) * 1000.0)
    print("clearances [mm]:", nominal_clearances_mm())
