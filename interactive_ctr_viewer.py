import math
import time

import numpy as np
import pygame
import pyvista as pv

from tube_parameters import build_supervisor_ctr_parameters
from CTR_superPosKin_fun_sectioned import superPosKin


# ============================================================
# PS5 MAPPING
# ============================================================

BUTTON_CROSS = 0
BUTTON_L1 = 9
BUTTON_R1 = 10

RIGHT_STICK_X = 2

L2_AXIS = 4
R2_AXIS = 5


# ============================================================
# CONTROL / PERFORMANCE SETTINGS
# ============================================================

STICK_DEADZONE = 0.08

MAX_ROTATION_SPEED_DEG_S = 300.0
MAX_TRANSLATION_SPEED_MM_S = 80.0

# Keyboard precision steps
KEYBOARD_TRANSLATION_STEP_MM = 1.0
KEYBOARD_ROTATION_STEP_DEG = 5.0

# Reduced slightly to make interactive simulation faster
MODEL_POINTS_PER_SECTION = 15

# Controller input polling
TIMER_INTERVAL_MS = 16

# Robot/model refresh
MODEL_UPDATE_HZ = 30.0

# On-screen text does not need to refresh at 60 Hz
STATUS_UPDATE_HZ = 5.0

# Lightweight visual backbone
BACKBONE_LINE_WIDTH = 7.0

TIP_RADIUS_MM = 2.5


# ============================================================
# ROBOT SETUP
# ============================================================

TUBE_NAMES = [
    "INNER",
    "MIDDLE",
    "OUTER"
]


CTR_par = build_supervisor_ctr_parameters()


sim_par = {
    "n_p": MODEL_POINTS_PER_SECTION,
    "isPlot": False
}


# Initial deployment [m]
ul = np.array([
    120e-3,
    70e-3,
    40e-3
], dtype=float)


# Initial rotation [rad]
uphi = np.deg2rad([
    0.0,
    0.0,
    0.0
])


selected_tube = 0


TOTAL_LENGTHS = np.array([
    sum(CTR_par["l_t"][0]),
    sum(CTR_par["l_t"][1]),
    sum(CTR_par["l_t"][2])
])


robot_dirty = True


# ============================================================
# CONTROLLER SETUP
# ============================================================

pygame.init()

pygame.joystick.init()


if pygame.joystick.get_count() == 0:

    controller = None

    print("")
    print("WARNING: No controller detected.")
    print("Keyboard controls are still available.")
    print("")

else:

    controller = pygame.joystick.Joystick(0)

    controller.init()

    print("")
    print("Controller detected:")
    print(controller.get_name())
    print("")


# ============================================================
# INPUT FILTERING
# ============================================================

def apply_deadzone(value):

    if abs(value) < STICK_DEADZONE:

        return 0.0


    sign = (
        1.0
        if value > 0.0
        else -1.0
    )


    scaled = (

        abs(value) - STICK_DEADZONE

    ) / (

        1.0 - STICK_DEADZONE
    )


    return sign * scaled


def trigger_value(raw_value):

    """
    On your PS5 controller:

    -1 = released
    +1 = fully pressed

    Convert to:

     0 = released
     1 = fully pressed
    """

    return float(

        np.clip(

            (raw_value + 1.0) / 2.0,

            0.0,

            1.0
        )
    )


# ============================================================
# PYVISTA SETUP
# ============================================================

plotter = pv.Plotter(
    window_size=[
        1200,
        800
    ]
)


plotter.set_background(
    "white"
)


plotter.show_grid()

plotter.add_axes()


# These will be created once
backbone_poly = None

backbone_actor = None

tip_actor = None


current_tip_mm = np.zeros(3)


# Controller diagnostics
last_right_x = 0.0

last_l2 = 0.0

last_r2 = 0.0


# ============================================================
# FORWARD MODEL
# ============================================================

def calculate_backbone():

    inputs = {

        "ul": ul.tolist(),

        "uphi": uphi.tolist()
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
        t_tip

    ) = superPosKin(

        CTR_par,

        inputs,

        sim_par
    )


    all_points = []


    for section_number, section in enumerate(rhoQ):

        section_points = np.column_stack([

            np.asarray(
                section[0],
                dtype=float
            ),

            np.asarray(
                section[1],
                dtype=float
            ),

            np.asarray(
                section[2],
                dtype=float
            )
        ])


        # Remove duplicate point between sections
        if section_number > 0:

            section_points = (
                section_points[1:]
            )


        all_points.append(
            section_points
        )


    if not all_points:

        raise RuntimeError(
            "Forward model returned no backbone."
        )


    backbone_mm = (

        np.vstack(
            all_points
        )

        * 1000.0
    )


    tip_mm = (

        np.asarray(
            rhoQ_tip[0:3],
            dtype=float
        )

        * 1000.0
    )


    return (
        backbone_mm,
        tip_mm
    )


# ============================================================
# STATUS / CONTROL LEGEND
# ============================================================

def make_status_text():

    ul_mm = (
        ul * 1000.0
    )


    rotations_deg = np.rad2deg(
        uphi
    )


    return (

        f"ACTIVE TUBE: "
        f"{TUBE_NAMES[selected_tube]}\n\n"


        f"Inner   | "
        f"{ul_mm[0]:6.1f} mm   "
        f"{rotations_deg[0]:7.1f} deg\n"


        f"Middle  | "
        f"{ul_mm[1]:6.1f} mm   "
        f"{rotations_deg[1]:7.1f} deg\n"


        f"Outer   | "
        f"{ul_mm[2]:6.1f} mm   "
        f"{rotations_deg[2]:7.1f} deg\n\n"


        f"TIP XYZ\n"

        f"X = {current_tip_mm[0]:7.1f} mm\n"

        f"Y = {current_tip_mm[1]:7.1f} mm\n"

        f"Z = {current_tip_mm[2]:7.1f} mm\n\n"


        f"CONTROLLER INPUT\n"

        f"Right stick L/R = "
        f"{last_right_x:+.2f}\n"

        f"L2              = "
        f"{last_l2:.2f}\n"

        f"R2              = "
        f"{last_r2:.2f}\n\n"


        f"PS5 CONTROLS\n"

        f"L1 / R1         select tube\n"

        f"L2              retract\n"

        f"R2              insert\n"

        f"Right stick L/R rotate tube\n"

        f"Cross           reset\n"

        f"Left stick      disabled for now\n\n"


        f"KEYBOARD / MOUSE BACKUP\n"

        f"1 / 2 / 3       "
        f"select Inner / Middle / Outer\n"

        f"W               insert +1 mm\n"

        f"S               retract -1 mm\n"

        f"A               rotate -5 deg\n"

        f"D               rotate +5 deg\n"

        f"R               reset\n"

        f"Q               quit\n"

        f"Mouse drag      rotate view\n"

        f"Mouse wheel     zoom"
    )


def update_status(
    render=False
):

    plotter.add_text(

        make_status_text(),

        position="upper_left",

        font_size=9,

        name="status_text",

        render=False
    )


    if render:

        plotter.render()


# ============================================================
# FAST ROBOT DRAWING
# ============================================================

def update_robot_scene():

    global backbone_poly

    global backbone_actor

    global tip_actor

    global current_tip_mm

    global robot_dirty


    backbone_mm, tip_mm = (
        calculate_backbone()
    )


    current_tip_mm = (
        tip_mm.copy()
    )


    # Remember current mouse-selected view
    saved_camera = (
        plotter.camera_position
    )


    # ========================================================
    # UPDATE BACKBONE IN PLACE
    # ========================================================

    if (
        backbone_poly is not None

        and

        backbone_poly.n_points
        == len(backbone_mm)
    ):

        # Much faster than removing and rebuilding
        backbone_poly.points = (
            backbone_mm
        )

        backbone_poly.Modified()


    else:

        # Number of model points changed,
        # so topology has to be rebuilt.

        if backbone_actor is not None:

            try:

                plotter.remove_actor(

                    backbone_actor,

                    render=False
                )

            except Exception:

                pass


        backbone_poly = (
            pv.lines_from_points(
                backbone_mm
            )
        )


        backbone_actor = (
            plotter.add_mesh(

                backbone_poly,

                line_width=BACKBONE_LINE_WIDTH,

                # Gives a rounded tube-like line
                # without constructing an expensive
                # physical tube mesh every frame.
                render_lines_as_tubes=True,

                smooth_shading=False,

                reset_camera=False,

                name="ctr_backbone",

                render=False
            )
        )


    # ========================================================
    # TIP MARKER
    # ========================================================

    if tip_actor is None:

        tip_mesh = pv.Sphere(

            radius=TIP_RADIUS_MM,

            center=(
                0.0,
                0.0,
                0.0
            )
        )


        tip_actor = plotter.add_mesh(

            tip_mesh,

            name="ctr_tip",

            reset_camera=False,

            render=False
        )


    # Instead of rebuilding the sphere,
    # simply move its actor.
    tip_actor.SetPosition(

        float(
            tip_mm[0]
        ),

        float(
            tip_mm[1]
        ),

        float(
            tip_mm[2]
        )
    )


    # Keep the camera exactly where the
    # user put it with the mouse.
    plotter.camera_position = (
        saved_camera
    )


    robot_dirty = False


    update_status(
        render=False
    )


    plotter.render()


# ============================================================
# TRANSLATION LIMITS
# ============================================================

def translation_limits(
    tube_number
):

    # Maintain:
    #
    # outer <= middle <= inner


    if tube_number == 0:

        minimum = ul[1]

        maximum = (
            TOTAL_LENGTHS[0]
        )


    elif tube_number == 1:

        minimum = ul[2]

        maximum = min(

            ul[0],

            TOTAL_LENGTHS[1]
        )


    else:

        minimum = 0.0

        maximum = min(

            ul[1],

            TOTAL_LENGTHS[2]
        )


    return (
        minimum,
        maximum
    )


# ============================================================
# TRANSLATION
# ============================================================

def change_translation_mm(
    amount_mm
):

    global robot_dirty


    minimum, maximum = (

        translation_limits(
            selected_tube
        )
    )


    old_value = float(
        ul[selected_tube]
    )


    ul[selected_tube] = np.clip(

        old_value

        + amount_mm / 1000.0,

        minimum,

        maximum
    )


    changed = not np.isclose(

        old_value,

        ul[selected_tube]
    )


    if changed:

        robot_dirty = True


    return changed


# ============================================================
# ROTATION
# ============================================================

def change_rotation_deg(
    amount_deg
):

    global robot_dirty


    old_value = float(
        uphi[selected_tube]
    )


    uphi[selected_tube] += (
        np.deg2rad(
            amount_deg
        )
    )


    # Wrap angle to -180 -> +180
    uphi[selected_tube] = (

        (

            uphi[selected_tube]

            + np.pi

        )

        % (

            2.0 * np.pi

        )

        - np.pi
    )


    changed = not np.isclose(

        old_value,

        uphi[selected_tube]
    )


    if changed:

        robot_dirty = True


    return changed


# ============================================================
# TUBE SELECTION
# ============================================================

def select_tube(
    number
):

    global selected_tube


    selected_tube = (
        number % 3
    )


    print(

        "Selected:",

        TUBE_NAMES[
            selected_tube
        ]
    )


    update_status(
        render=True
    )


def previous_tube():

    select_tube(
        selected_tube - 1
    )


def next_tube():

    select_tube(
        selected_tube + 1
    )


# ============================================================
# RESET
# ============================================================

def reset_robot():

    global selected_tube

    global robot_dirty


    ul[:] = np.array([

        120e-3,

        70e-3,

        40e-3
    ])


    uphi[:] = np.deg2rad([

        0.0,

        0.0,

        0.0
    ])


    selected_tube = 0

    robot_dirty = True


    update_robot_scene()


    print(
        "Robot reset."
    )


# ============================================================
# KEYBOARD BACKUP
# ============================================================

def keyboard_insert():

    if change_translation_mm(
        +KEYBOARD_TRANSLATION_STEP_MM
    ):

        update_robot_scene()


def keyboard_retract():

    if change_translation_mm(
        -KEYBOARD_TRANSLATION_STEP_MM
    ):

        update_robot_scene()


def keyboard_rotate_negative():

    if change_rotation_deg(
        -KEYBOARD_ROTATION_STEP_DEG
    ):

        update_robot_scene()


def keyboard_rotate_positive():

    if change_rotation_deg(
        +KEYBOARD_ROTATION_STEP_DEG
    ):

        update_robot_scene()


def quit_viewer():

    plotter.close()


plotter.add_key_event(

    "1",

    lambda: select_tube(0)
)


plotter.add_key_event(

    "2",

    lambda: select_tube(1)
)


plotter.add_key_event(

    "3",

    lambda: select_tube(2)
)


plotter.add_key_event(

    "w",

    keyboard_insert
)


plotter.add_key_event(

    "s",

    keyboard_retract
)


plotter.add_key_event(

    "a",

    keyboard_rotate_negative
)


plotter.add_key_event(

    "d",

    keyboard_rotate_positive
)


plotter.add_key_event(

    "r",

    reset_robot
)


plotter.add_key_event(

    "q",

    quit_viewer
)


# ============================================================
# CONTROLLER BUTTON EDGE STATE
# ============================================================

previous_l1 = False

previous_r1 = False

previous_cross = False


# ============================================================
# TIMER STATE
# ============================================================

last_timer_time = (
    time.perf_counter()
)


last_model_update_time = (
    last_timer_time
)


last_status_update_time = (
    last_timer_time
)


MODEL_UPDATE_INTERVAL = (
    1.0 / MODEL_UPDATE_HZ
)


STATUS_UPDATE_INTERVAL = (
    1.0 / STATUS_UPDATE_HZ
)


# ============================================================
# PS5 TIMER CALLBACK
# ============================================================

def controller_tick(
    step
):

    global previous_l1

    global previous_r1

    global previous_cross

    global last_timer_time

    global last_model_update_time

    global last_status_update_time

    global last_right_x

    global last_l2

    global last_r2


    if controller is None:

        return


    now = (
        time.perf_counter()
    )


    dt = min(

        now - last_timer_time,

        0.05
    )


    last_timer_time = (
        now
    )


    pygame.event.pump()


    # ========================================================
    # BUTTONS
    # ========================================================

    l1_now = bool(

        controller.get_button(
            BUTTON_L1
        )
    )


    r1_now = bool(

        controller.get_button(
            BUTTON_R1
        )
    )


    cross_now = bool(

        controller.get_button(
            BUTTON_CROSS
        )
    )


    if (

        l1_now

        and not previous_l1
    ):

        previous_tube()


    if (

        r1_now

        and not previous_r1
    ):

        next_tube()


    if (

        cross_now

        and not previous_cross
    ):

        reset_robot()


    previous_l1 = (
        l1_now
    )


    previous_r1 = (
        r1_now
    )


    previous_cross = (
        cross_now
    )


    # ========================================================
    # RIGHT STICK LEFT / RIGHT
    #
    # This is what "Right X" meant previously.
    # ========================================================

    last_right_x = apply_deadzone(

        controller.get_axis(
            RIGHT_STICK_X
        )
    )


    # ========================================================
    # TRIGGERS
    # ========================================================

    last_l2 = trigger_value(

        controller.get_axis(
            L2_AXIS
        )
    )


    last_r2 = trigger_value(

        controller.get_axis(
            R2_AXIS
        )
    )


    # ========================================================
    # RIGHT STICK -> TUBE ROTATION
    # ========================================================

    if last_right_x != 0.0:

        change_rotation_deg(

            last_right_x

            * MAX_ROTATION_SPEED_DEG_S

            * dt
        )


    # ========================================================
    # TRIGGERS -> TUBE TRANSLATION
    # ========================================================

    translation_command = (

        last_r2

        - last_l2
    )


    if abs(
        translation_command
    ) > 0.02:

        change_translation_mm(

            translation_command

            * MAX_TRANSLATION_SPEED_MM_S

            * dt
        )


    # ========================================================
    # MODEL UPDATE
    # ========================================================

    if (

        robot_dirty

        and

        now - last_model_update_time
        >= MODEL_UPDATE_INTERVAL
    ):

        update_robot_scene()


        last_model_update_time = (
            now
        )


    # ========================================================
    # STATUS UPDATE
    #
    # Do NOT redraw the text at 60 Hz.
    # ========================================================

    if (

        now - last_status_update_time

        >= STATUS_UPDATE_INTERVAL
    ):

        update_status(
            render=True
        )


        last_status_update_time = (
            now
        )


# ============================================================
# INITIAL SCENE
# ============================================================

update_robot_scene()


plotter.camera_position = (
    "iso"
)


plotter.reset_camera()


update_status(
    render=False
)


print("")
print("======================================")
print("PS5 CTR SIMULATOR - FAST VIEW")
print("======================================")
print("")
print("PS5")
print("L1 / R1         select tube")
print("L2              retract")
print("R2              insert")
print("Right stick L/R rotate tube")
print("Cross           reset")
print("")
print("Left stick currently disabled.")
print("")
print("KEYBOARD")
print("1 / 2 / 3       select tubes")
print("W               insert +1 mm")
print("S               retract -1 mm")
print("A               rotate -5 deg")
print("D               rotate +5 deg")
print("R               reset")
print("Q               quit")
print("")
print("MOUSE")
print("Drag            rotate view")
print("Wheel           zoom")
print("")


# ============================================================
# TIMER
# ============================================================

plotter.add_timer_event(

    max_steps=10_000_000,

    duration=TIMER_INTERVAL_MS,

    callback=controller_tick
)


# ============================================================
# START VIEWER
# ============================================================

try:

    plotter.show(

        title="PS5 CTR Simulator"
    )


finally:

    pygame.quit()


    print("")
    print("CTR simulator closed.")