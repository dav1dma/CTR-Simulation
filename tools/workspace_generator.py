import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tube_parameters import (
    build_supervisor_ctr_parameters,
    total_tube_lengths
)

from CTR_superPosKin_fun_sectioned import superPosKin


# ============================================================
# SETTINGS
# ============================================================

N_SAMPLES = 10000
SEED = 42

# Number of points used to calculate each CTR backbone.
# We only need the tip for the workspace, so 30 is sufficient.
N_POINTS_PER_SEGMENT = 30

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "workspace_10000_sectioned.csv"
OUTPUT_PNG = RESULTS_DIR / "workspace_10000_sectioned_3D.png"


# ============================================================
# RANDOM VALID TRANSLATION SAMPLING
# ============================================================

def sample_valid_ul(rng, total_lengths):
    """
    Generate one valid nested tube deployment.

    Tube order:
        ul[0] = inner
        ul[1] = middle
        ul[2] = outer

    The inherited workspace convention assumes:

        0 <= outer <= middle <= inner

    while also respecting each tube's total physical length.
    """

    L_inner = total_lengths[0]
    L_middle = total_lengths[1]
    L_outer = total_lengths[2]

    # Start from the shortest/outer tube
    ul_outer = rng.uniform(0.0, L_outer)

    # Middle must extend at least as far as outer
    ul_middle = rng.uniform(
        ul_outer,
        L_middle
    )

    # Inner must extend at least as far as middle
    ul_inner = rng.uniform(
        ul_middle,
        L_inner
    )

    return [
        ul_inner,
        ul_middle,
        ul_outer
    ]


# ============================================================
# MAIN WORKSPACE SIMULATION
# ============================================================

def main():

    RESULTS_DIR.mkdir(exist_ok=True)

    print("======================================")
    print("CTR WORKSPACE GENERATOR")
    print("======================================")

    # --------------------------------------------------------
    # Load final supervisor parameters
    # --------------------------------------------------------

    CTR_par = build_supervisor_ctr_parameters()

    total_lengths = total_tube_lengths(CTR_par)

    print("\nTube total lengths:")

    print(
        f"Inner  = {total_lengths[0] * 1000:.1f} mm"
    )

    print(
        f"Middle = {total_lengths[1] * 1000:.1f} mm"
    )

    print(
        f"Outer  = {total_lengths[2] * 1000:.1f} mm"
    )

    print("\nPre-curvatures:")

    print(
        CTR_par["kappa_0"]
    )

    print(f"\nGenerating {N_SAMPLES} configurations...\n")


    # --------------------------------------------------------
    # Simulation settings
    # --------------------------------------------------------

    sim_par = {

        "n_p": N_POINTS_PER_SEGMENT,

        # IMPORTANT:
        # Do not let superPosKin create a new Matplotlib plot
        # for every single sample.
        "isPlot": False
    }


    # --------------------------------------------------------
    # Random number generator
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)


    # --------------------------------------------------------
    # Storage arrays
    # --------------------------------------------------------

    tips = np.zeros(
        (N_SAMPLES, 3)
    )

    ul_values = np.zeros(
        (N_SAMPLES, 3)
    )

    uphi_values = np.zeros(
        (N_SAMPLES, 3)
    )


    # --------------------------------------------------------
    # Run simulations
    # --------------------------------------------------------

    successful_samples = 0

    for i in range(N_SAMPLES):

        if i % 100 == 0:

            print(
                f"Running sample {i} / {N_SAMPLES}"
            )


        # ----------------------------------------------------
        # Random tube translations
        # ----------------------------------------------------

        ul = sample_valid_ul(
            rng,
            total_lengths
        )


        # ----------------------------------------------------
        # Random tube rotations
        #
        # Each tube is allowed one full revolution:
        #
        # -pi -> +pi
        # = -180 -> +180 degrees
        # ----------------------------------------------------

        uphi = rng.uniform(
            -np.pi,
            np.pi,
            size=3
        ).tolist()


        inputs = {

            "ul": ul,

            "uphi": uphi
        }


        # ----------------------------------------------------
        # Forward kinematics
        # ----------------------------------------------------

        try:

            (
                rhoQ_tip,
                R_tip,
                s,
                rhoQ,
                Kappa,
                Kappa_xy,
                phi,
                ul_sorted,
                t_tip

            ) = superPosKin(

                CTR_par,
                inputs,
                sim_par
            )


            tip = np.array(
                rhoQ_tip[0:3],
                dtype=float
            )


            # Check numerical validity
            if not np.all(
                np.isfinite(tip)
            ):

                print(
                    f"Skipping invalid sample {i}"
                )

                continue


            # Save values
            tips[successful_samples] = tip

            ul_values[successful_samples] = ul

            uphi_values[successful_samples] = uphi

            successful_samples += 1


        except Exception as error:

            print(
                f"Sample {i} failed:"
            )

            print(error)


    # --------------------------------------------------------
    # Remove unused rows if any samples failed
    # --------------------------------------------------------

    tips = tips[:successful_samples]

    ul_values = ul_values[:successful_samples]

    uphi_values = uphi_values[:successful_samples]


    print(
        f"\nSuccessful configurations: "
        f"{successful_samples}/{N_SAMPLES}"
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    output_path = Path(
        OUTPUT_CSV
    )


    with output_path.open(
        "w",
        newline=""
    ) as csvfile:

        writer = csv.writer(
            csvfile
        )


        writer.writerow([
            "ul_inner_mm",
            "ul_middle_mm",
            "ul_outer_mm",

            "uphi_inner_deg",
            "uphi_middle_deg",
            "uphi_outer_deg",

            "tip_x_mm",
            "tip_y_mm",
            "tip_z_mm"
        ])


        for i in range(successful_samples):

            writer.writerow([

                ul_values[i, 0] * 1000,
                ul_values[i, 1] * 1000,
                ul_values[i, 2] * 1000,

                np.rad2deg(
                    uphi_values[i, 0]
                ),

                np.rad2deg(
                    uphi_values[i, 1]
                ),

                np.rad2deg(
                    uphi_values[i, 2]
                ),

                tips[i, 0] * 1000,
                tips[i, 1] * 1000,
                tips[i, 2] * 1000
            ])


    print(
        f"\nSaved CSV: {OUTPUT_CSV}"
    )


    # ========================================================
    # WORKSPACE SIZE INFORMATION
    # ========================================================

    tips_mm = tips * 1000


    x_min = np.min(
        tips_mm[:, 0]
    )

    x_max = np.max(
        tips_mm[:, 0]
    )

    y_min = np.min(
        tips_mm[:, 1]
    )

    y_max = np.max(
        tips_mm[:, 1]
    )

    z_min = np.min(
        tips_mm[:, 2]
    )

    z_max = np.max(
        tips_mm[:, 2]
    )


    print("\nWorkspace limits:")

    print(
        f"X: {x_min:.1f} to "
        f"{x_max:.1f} mm"
    )

    print(
        f"Y: {y_min:.1f} to "
        f"{y_max:.1f} mm"
    )

    print(
        f"Z: {z_min:.1f} to "
        f"{z_max:.1f} mm"
    )


    # ========================================================
    # 3D WORKSPACE PLOT
    # ========================================================

    fig = plt.figure(
        figsize=(8, 7)
    )


    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    ax.scatter(

        tips_mm[:, 0],

        tips_mm[:, 1],

        tips_mm[:, 2],

        s=5,

        alpha=0.5
    )


    # Robot base
    ax.scatter(

        [0],

        [0],

        [0],

        s=60,

        marker="x",

        label="Robot base"
    )


    ax.set_xlabel(
        "X (mm)"
    )

    ax.set_ylabel(
        "Y (mm)"
    )

    ax.set_zlabel(
        "Z (mm)"
    )


    ax.set_title(
        f"Simulated CTR Tip Workspace\n"
        f"N = {successful_samples}"
    )


    ax.legend()


    # --------------------------------------------------------
    # Equal-ish axis scale
    # --------------------------------------------------------

    x_range = x_max - x_min

    y_range = y_max - y_min

    z_range = z_max - z_min


    max_range = max(
        x_range,
        y_range,
        z_range
    )


    x_mid = (
        x_max + x_min
    ) / 2

    y_mid = (
        y_max + y_min
    ) / 2

    z_mid = (
        z_max + z_min
    ) / 2


    ax.set_xlim(
        x_mid - max_range / 2,
        x_mid + max_range / 2
    )

    ax.set_ylim(
        y_mid - max_range / 2,
        y_mid + max_range / 2
    )

    ax.set_zlim(
        z_mid - max_range / 2,
        z_mid + max_range / 2
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_PNG,
        dpi=300
    )


    print(
        f"Saved figure: {OUTPUT_PNG}"
    )


    # Show graph on screen
    plt.show()


    print("\nWorkspace generation complete.")


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
