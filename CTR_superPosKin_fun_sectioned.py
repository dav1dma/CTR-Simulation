"""
Geometry-aware extension of the inherited constant-curvature CTR forward model.

Main change relative to CTR_superPosKin_fun_compatible.py:
----------------------------------------------------------
The inherited model applies each tube's intrinsic curvature over its entire
deployed length ul[i].

This version uses CTR_par['l_t'][i] = [straight_length, curved_length] and
assumes the pre-curved portion is the DISTAL part of each tube.

For tube i:
    exposed length = ul[i]
    curved length  = Lc[i]

If ul[i] <= Lc[i]:
    the entire exposed portion is pre-curved.

If ul[i] > Lc[i]:
    [0, ul[i]-Lc[i]) is a straight intrinsic section
    [ul[i]-Lc[i], ul[i]] is the pre-curved distal section.

Straight tube portions still contribute bending stiffness EI to the
stiffness-weighted equilibrium, but contribute zero intrinsic curvature.

This remains an ideal constant-curvature stiffness-superposition model.
It does NOT add friction, distributed torsion, hysteresis, contact, or
experimental validation.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import transforms3d as tr
except ImportError:
    from scipy.spatial.transform import Rotation as _Rotation

    class _QuaternionFallback:
        @staticmethod
        def mat2quat(R):
            x, y, z, w = _Rotation.from_matrix(R).as_quat()
            return np.array([w, x, y, z])

    class _Transforms3DFallback:
        quaternions = _QuaternionFallback()

    tr = _Transforms3DFallback()


def superPosKin(CTR_par, inputs, sim_par):

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    n_t = CTR_par["n_t"]
    l_t = CTR_par["l_t"]          # [[straight, curved], ...] [m]
    E = CTR_par["E"]              # [Pa]
    kappa_0 = CTR_par["kappa_0"]  # [1/m]
    r = CTR_par["r"]              # [[inner radius, outer radius], ...] [m]

    ul = np.asarray(inputs["ul"], dtype=float)       # deployed length [m]
    uphi = np.asarray(inputs["uphi"], dtype=float)   # base rotation [rad]

    n_p = int(sim_par.get("n_p", 30))
    isPlot = bool(sim_par.get("isPlot", False))

    if len(ul) != n_t or len(uphi) != n_t:
        raise ValueError("ul and uphi must each contain one value per tube.")

    if np.any(ul < 0.0):
        raise ValueError("Tube deployment lengths ul must be non-negative.")

    total_lengths = np.asarray(
        [straight + curved for straight, curved in l_t],
        dtype=float
    )

    if np.any(ul > total_lengths + 1e-12):
        raise ValueError(
            f"A deployment exceeds the physical tube length.\n"
            f"ul = {ul}\n"
            f"tube totals = {total_lengths}"
        )

    # ------------------------------------------------------------------
    # Tube bending stiffness and rotated intrinsic curvature
    # ------------------------------------------------------------------
    EI = np.zeros(n_t, dtype=float)
    kappa_xy0 = np.zeros((n_t, 2), dtype=float)

    curved_lengths = np.asarray(
        [pair[1] for pair in l_t],
        dtype=float
    )

    for i_t in range(n_t):

        r_inner = r[i_t][0]
        r_outer = r[i_t][1]

        # Second moment used by the inherited model.
        I = np.pi * (r_outer**4 - r_inner**4) / 4.0
        EI[i_t] = E[i_t] * I

        # Preserve inherited curvature/rotation convention.
        kappa_xy0[i_t, 0] = kappa_0[i_t] * np.cos(uphi[i_t])
        kappa_xy0[i_t, 1] = kappa_0[i_t] * np.sin(uphi[i_t])

    # ------------------------------------------------------------------
    # Position, measured from insertion point, at which each tube's
    # distal pre-curved section starts.
    #
    # Example:
    # middle: ul=140 mm, curved length=90 mm
    # -> first 50 mm exposed is intrinsically straight
    # -> final 90 mm exposed is intrinsically curved
    # ------------------------------------------------------------------
    curved_start = np.maximum(0.0, ul - curved_lengths)

    # ------------------------------------------------------------------
    # Build all axial breakpoints at which the mechanics can change:
    # 1) a tube straight-to-curved transition
    # 2) a tube distal tip
    # ------------------------------------------------------------------
    breakpoints = {0.0}

    for i_t in range(n_t):

        if ul[i_t] > 0.0:
            breakpoints.add(float(ul[i_t]))

        if (
            curved_lengths[i_t] > 0.0
            and curved_start[i_t] > 0.0
            and curved_start[i_t] < ul[i_t]
        ):
            breakpoints.add(float(curved_start[i_t]))

    breakpoints = np.asarray(
        sorted(breakpoints),
        dtype=float
    )

    # ------------------------------------------------------------------
    # Calculate the stiffness-weighted equilibrium curvature separately
    # for every piecewise-constant axial section.
    # ------------------------------------------------------------------
    sections = []

    for i_s in range(len(breakpoints) - 1):

        s0 = breakpoints[i_s]
        s1 = breakpoints[i_s + 1]

        if s1 - s0 <= 1e-12:
            continue

        s_mid = 0.5 * (s0 + s1)

        # A tube is physically present if the section lies before its tip.
        active_tubes = [
            i_t
            for i_t in range(n_t)
            if s_mid < ul[i_t] - 1e-12
        ]

        if not active_tubes:
            continue

        weighted_intrinsic_curvature = np.zeros(2, dtype=float)
        stiffness_sum = 0.0

        for i_t in active_tubes:

            # Straight portions still resist bending.
            stiffness_sum += EI[i_t]

            # Intrinsic curvature is non-zero only in the distal
            # pre-curved part of this particular tube.
            is_in_curved_part = (
                curved_lengths[i_t] > 0.0
                and s_mid >= curved_start[i_t] - 1e-12
            )

            if is_in_curved_part:
                weighted_intrinsic_curvature += (
                    EI[i_t] * kappa_xy0[i_t]
                )

        Kappa_xy = (
            weighted_intrinsic_curvature / stiffness_sum
        )

        Kappa = float(
            np.linalg.norm(Kappa_xy)
        )

        if np.isclose(Kappa, 0.0):
            phi = 0.0
        else:
            # Preserve inherited bending-plane convention.
            phi = float(
                np.arctan2(
                    Kappa_xy[0],
                    Kappa_xy[1]
                )
            )

        sections.append(
            {
                "s0": s0,
                "s1": s1,
                "length": s1 - s0,
                "Kappa_xy": Kappa_xy,
                "Kappa": Kappa,
                "phi": phi,
                "active_tubes": active_tubes,
            }
        )

    # ------------------------------------------------------------------
    # Generate the piecewise constant-curvature backbone.
    #
    # rhoQ has one entry per axial section rather than exactly one entry
    # per tube. Existing code that only uses rhoQ_tip is unaffected.
    # Later visualisation code should iterate over len(rhoQ).
    # ------------------------------------------------------------------
    n_sections = len(sections)

    rhoQ = [
        [
            [0.0 for _ in range(n_p + 1)]
            for _ in range(7)
        ]
        for _ in range(n_sections)
    ]

    s = [
        [
            i_p * sections[i_s]["length"] / n_p
            for i_p in range(n_p + 1)
        ]
        for i_s in range(n_sections)
    ]

    T0 = np.identity(4)

    for i_s, section in enumerate(sections):

        Kappa = section["Kappa"]
        phi = section["phi"]

        for i_p in range(n_p + 1):

            s_local = s[i_s][i_p]
            theta = s_local * Kappa

            if np.isclose(Kappa, 0.0):

                rho = np.array(
                    [0.0, 0.0, s_local]
                )

            else:

                rho = np.array(
                    [
                        -(
                            np.cos(phi)
                            * (np.cos(Kappa * s_local) - 1.0)
                        ) / Kappa,

                        -(
                            np.sin(phi)
                            * (np.cos(Kappa * s_local) - 1.0)
                        ) / Kappa,

                        np.sin(Kappa * s_local) / Kappa,
                    ]
                )

            # Preserve the inherited model's section transform convention.
            R = np.array(
                [
                    [np.cos(phi), -np.sin(phi), 0.0],
                    [np.sin(phi),  np.cos(phi), 0.0],
                    [0.0,          0.0,         1.0],
                ]
            ) @ np.array(
                [
                    [ np.cos(theta), 0.0, np.sin(theta)],
                    [ 0.0,           1.0, 0.0],
                    [-np.sin(theta), 0.0, np.cos(theta)],
                ]
            )

            Tl = np.array(
                [
                    [R[0, 0], R[0, 1], R[0, 2], rho[0]],
                    [R[1, 0], R[1, 1], R[1, 2], rho[1]],
                    [R[2, 0], R[2, 1], R[2, 2], rho[2]],
                    [0.0,     0.0,     0.0,     1.0],
                ]
            )

            T = T0 @ Tl

            rhoQ[i_s][0][i_p] = T[0, 3]
            rhoQ[i_s][1][i_p] = T[1, 3]
            rhoQ[i_s][2][i_p] = T[2, 3]

            q = tr.quaternions.mat2quat(
                T[0:3, 0:3]
            )

            rhoQ[i_s][3][i_p] = q[0]
            rhoQ[i_s][4][i_p] = q[1]
            rhoQ[i_s][5][i_p] = q[2]
            rhoQ[i_s][6][i_p] = q[3]

        T0 = T

    # ------------------------------------------------------------------
    # Tip values
    # ------------------------------------------------------------------
    if n_sections == 0:

        R_tip = np.identity(3)
        t_tip = np.array([[0.0], [0.0], [1.0]])

        rhoQ_tip = [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0
        ]

    else:

        R_tip = T0[0:3, 0:3]

        t_tip = (
            R_tip
            @ np.array([[0.0], [0.0], [1.0]])
        )

        q_tip = tr.quaternions.mat2quat(
            R_tip
        )

        rhoQ_tip = [
            T0[0, 3],
            T0[1, 3],
            T0[2, 3],
            q_tip[0],
            q_tip[1],
            q_tip[2],
            q_tip[3],
        ]

    Kappa = [
        section["Kappa"]
        for section in sections
    ]

    Kappa_xy = [
        section["Kappa_xy"].tolist()
        for section in sections
    ]

    phi = [
        section["phi"]
        for section in sections
    ]

    ul_sorted = sorted(
        ul.tolist()
    )

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if isPlot:

        ax = plt.axes(
            projection="3d"
        )

        for i_s in range(n_sections):

            ax.plot3D(
                rhoQ[i_s][0],
                rhoQ[i_s][1],
                rhoQ[i_s][2]
            )

        tip_mm = (
            np.asarray(rhoQ_tip[0:3])
            * 1000.0
        )

        ax.scatter(
            [rhoQ_tip[0]],
            [rhoQ_tip[1]],
            [rhoQ_tip[2]]
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        ax.set_title(
            "Section-aware CTR forward model\n"
            f"Tip = {np.round(tip_mm, 1)} mm"
        )

        plt.show()

    return (
        rhoQ_tip,
        R_tip,
        s,
        rhoQ,
        Kappa,
        Kappa_xy,
        phi,
        ul_sorted,
        t_tip,
    )
