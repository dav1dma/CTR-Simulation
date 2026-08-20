
"""
Static forward-model test for the supervisor-provided tube parameters.

Put this file in the same folder as:
    tube_parameters.py
    CTR_superPosKin_fun.py

Then run:
    python static_configuration_test.py
"""

import numpy as np
from tube_parameters import (
    build_supervisor_ctr_parameters,
    total_tube_lengths,
    nominal_clearances_mm,
)
from CTR_superPosKin_fun_compatible import superPosKin

def run_case(name, ul_mm, uphi_deg):
    CTR_par = build_supervisor_ctr_parameters()
    sim_par = {"n_p": 40, "isPlot": True}

    ul = (np.array(ul_mm, dtype=float) / 1000.0).tolist()
    uphi = np.deg2rad(uphi_deg).tolist()

    rhoQ_tip, R_tip, s, rhoQ, Kappa, Kappa_xy, phi, ul_sorted, t_tip = superPosKin(
        CTR_par,
        {"ul": ul, "uphi": uphi},
        sim_par,
    )

    tip_mm = np.array(rhoQ_tip[:3], dtype=float) * 1000.0

    print(f"\n{name}")
    print("ul [mm]:", ul_mm)
    print("uphi [deg]:", uphi_deg)
    print("tip XYZ [mm]:", np.round(tip_mm, 3))


if __name__ == "__main__":
    CTR_par = build_supervisor_ctr_parameters()

    print("Supervisor parameter set")
    print("------------------------")
    print("l_t [mm]:", [[v*1000 for v in pair] for pair in CTR_par["l_t"]])
    print("total lengths [mm]:", total_tube_lengths(CTR_par) * 1000.0)
    print("kappa_0 [1/m]:", CTR_par["kappa_0"])
    print("E [GPa]:", [e/1e9 for e in CTR_par["E"]])
    print("clearances [mm]:", nominal_clearances_mm())

    # Example software-check configurations within the new tube lengths.
    run_case("all aligned", [120, 70, 40], [0, 0, 0])
    run_case("middle rotated 90 deg", [120, 70, 40], [0, 90, 0])
    run_case("outer opposed", [140, 75, 50], [0, 0, 180])
