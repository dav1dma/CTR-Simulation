"""
Static checks for the section-aware CTR forward model.

These cases deliberately include tube deployments longer than the
pre-curved lengths so that straight portions are exposed and the new
section handling is actually exercised.

Tube parameters:
    inner  : total 350 mm, curved 0 mm
    middle : total 170 mm, curved 90 mm
    outer  : total 80 mm,  curved 65 mm
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tube_parameters import (
    build_supervisor_ctr_parameters,
)

from CTR_superPosKin_fun_sectioned import (
    superPosKin,
)


def run_case(name, ul_mm, uphi_deg):

    CTR_par = (
        build_supervisor_ctr_parameters()
    )

    inputs = {
        "ul": (
            np.asarray(ul_mm, dtype=float)
            / 1000.0
        ).tolist(),

        "uphi": np.deg2rad(
            uphi_deg
        ).tolist(),
    }

    sim_par = {
        "n_p": 50,
        "isPlot": False,
    }

    (
        rhoQ_tip,
        R_tip,
        s,
        rhoQ,
        Kappa,
        Kappa_xy,
        phi,
        ul_sorted,
        t_tip,

    ) = superPosKin(
        CTR_par,
        inputs,
        sim_par
    )

    tip_mm = (
        np.asarray(
            rhoQ_tip[0:3],
            dtype=float
        )
        * 1000.0
    )

    print("\n--------------------------------")
    print(name)
    print("--------------------------------")

    print(
        "ul [mm]:",
        ul_mm
    )

    print(
        "uphi [deg]:",
        uphi_deg
    )

    print(
        "Number of axial sections:",
        len(rhoQ)
    )

    print(
        "Section curvatures [1/m]:",
        np.round(Kappa, 4)
    )

    print(
        "Tip XYZ [mm]:",
        np.round(tip_mm, 3)
    )


if __name__ == "__main__":

    # --------------------------------------------------------
    # Case 1:
    # Middle and outer are deployed LESS than their curved
    # lengths. Therefore all exposed portions of those tubes
    # are curved. This behaves similarly to the inherited model.
    # --------------------------------------------------------
    run_case(
        "CASE 1 - all exposed curved portions",
        ul_mm=[120, 70, 40],
        uphi_deg=[0, 0, 0]
    )

    # --------------------------------------------------------
    # Case 2:
    # Middle deployment = 140 mm:
    #     50 mm straight + 90 mm curved exposed
    #
    # Outer deployment = 75 mm:
    #     10 mm straight + 65 mm curved exposed
    #
    # This directly tests the new straight/curved handling.
    # --------------------------------------------------------
    run_case(
        "CASE 2 - straight sections exposed",
        ul_mm=[160, 140, 75],
        uphi_deg=[0, 0, 0]
    )

    # --------------------------------------------------------
    # Case 3:
    # Same deployments as Case 2, but rotate the middle tube.
    # --------------------------------------------------------
    run_case(
        "CASE 3 - middle tube rotated 90 degrees",
        ul_mm=[160, 140, 75],
        uphi_deg=[0, 90, 0]
    )

    # --------------------------------------------------------
    # Case 4:
    # Same geometry but opposing outer-tube curvature.
    # --------------------------------------------------------
    run_case(
        "CASE 4 - outer tube opposed",
        ul_mm=[160, 140, 75],
        uphi_deg=[0, 0, 180]
    )
