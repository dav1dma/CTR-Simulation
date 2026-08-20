import math
import time

import numpy as np
import pygame
import pyvista as pv

from tube_parameters import build_supervisor_ctr_parameters
from CTR_superPosKin_fun_sectioned import superPosKin


# ============================================================
# PS5 CONTROLLER MAPPING
# ============================================================

BUTTON_CROSS = 0

BUTTON_L1 = 9
BUTTON_R1 = 10

LEFT_STICK_X = 0
LEFT_STICK_Y = 1

RIGHT_STICK_X = 2
RIGHT_STICK_Y = 3      # intentionally unused

L2_AXIS = 4
R2_AXIS = 5


# ============================================================
# CONTROL SETTINGS
# ============================================================

STICK_DEADZONE = 0.08

MAX_ROTATION_SPEED_DEG_S = 300.0
MAX_TRANSLATION_SPEED_MM_S = 80.0

CAMERA_ORBIT_SPEED_DEG_S = 90.0


# Keyboard backup
KEYBOARD_TRANSLATION_STEP_MM = 1.0
KEYBOARD_ROTATION_STEP_DEG = 5.0

KEYBOARD_CAMERA_STEP_DEG = 5.0


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Controller / UI polling:
# ~60 Hz
TIMER_INTERVAL_MS = 16


# Mechanical model update rate
MODEL_UPDATE_HZ = 40.0

MODEL_UPDATE_INTERVAL = (
    1.0 / MODEL_UPDATE_HZ
)


# Text does not need to update at 60 Hz
STATUS_UPDATE_HZ = 5.0

STATUS_UPDATE_INTERVAL = (
    1.0 / STATUS_UPDATE_HZ
)


# Points calculated in each model section
MODEL_POINTS_PER_SECTION = 12


# Fixed number of points used to DRAW the CTR.
#
# Keeping this constant means PyVista can reuse
# the same actor instead of creating a new one.
DISPLAY_POINTS = 100


BACKBONE_LINE_WIDTH = 7.0

TIP_RADIUS_MM = 2.5


SHOW_GRID = True


# ============================================================
# ROBOT PARAMETERS
# ============================================================

TUBE_NAMES = [

    "INNER",

    "MIDDLE",

    "OUTER"
]


CTR_par = (
    build_supervisor_ctr_parameters()
)


sim_par = {

    "n_p": MODEL_POINTS_PER_SECTION,

    "isPlot": False
}


# ============================================================
# INITIAL ROBOT STATE
# ============================================================

# Deployment [m]
ul = np.array([

    120e-3,

    70e-3,

    40e-3

], dtype=float)


# Rotation [rad]
uphi = np.deg2rad([

    0.0,

    0.0,

    0.0
])


selected_tube = 0


TOTAL_LENGTHS = np.array([

    sum(
        CTR_par["l_t"][0]
    ),

    sum(
        CTR_par["l_t"][1]
    ),

    sum(
        CTR_par["l_t"][2]
    )

], dtype=float)


robot_dirty = True


# ============================================================
# CONTROLLER SETUP
# ============================================================

pygame.init()

pygame.joystick.init()


controller = None


last_controller_reconnect_attempt = (
    0.0
)


# Button edge detection
previous_l1 = False

previous_r1 = False

previous_cross = False


def connect_controller():

    global controller

    global previous_l1
    global previous_r1
    global previous_cross


    try:

        pygame.joystick.quit()

        pygame.joystick.init()


        if pygame.joystick.get_count() == 0:

            controller = None

            return False


        controller = (
            pygame.joystick.Joystick(0)
        )


        controller.init()


        previous_l1 = False

        previous_r1 = False

        previous_cross = False


        print("")
        print("Controller connected:")
        print(
            controller.get_name()
        )
        print("")


        return True


    except pygame.error as exc:

        controller = None

        print(
            f"Controller connection error: {exc}"
        )


        return False


connect_controller()


# ============================================================
# CONTROLLER FILTERING
# ============================================================

def apply_axis_deadzone(value):

    """
    Deadzone for one analogue axis.
    """

    if abs(value) < STICK_DEADZONE:

        return 0.0


    sign = (

        1.0

        if value > 0.0

        else -1.0
    )


    scaled = (

        abs(value)

        - STICK_DEADZONE

    ) / (

        1.0

        - STICK_DEADZONE
    )


    return (
        sign * scaled
    )


def apply_stick_deadzone(
    x,
    y
):

    """
    Radial deadzone for the left analogue stick.
    """

    magnitude = (
        math.hypot(
            x,
            y
        )
    )


    if magnitude < STICK_DEADZONE:

        return (
            0.0,
            0.0
        )


    magnitude = min(

        magnitude,

        1.0
    )


    scaled_magnitude = (

        magnitude

        - STICK_DEADZONE

    ) / (

        1.0

        - STICK_DEADZONE
    )


    scale = (

        scaled_magnitude

        / magnitude
    )


    return (

        x * scale,

        y * scale
    )


def trigger_value(
    raw_value
):

    """
    PS5 trigger:

    -1 = released
    +1 = fully pressed

    converted to:

     0 = released
     1 = fully pressed
    """

    return float(

        np.clip(

            (
                raw_value
                + 1.0
            )
            / 2.0,

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


if SHOW_GRID:

    plotter.show_grid()


plotter.add_axes()


# ============================================================
# FORWARD MODEL
# ============================================================

def calculate_backbone():

    inputs = {

        "ul":
            ul.tolist(),

        "uphi":
            uphi.tolist()
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


    for (
        section_number,
        section

    ) in enumerate(rhoQ):


        section_points = (
            np.column_stack([

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
        )


        # Remove duplicate connection
        # point between model sections
        if section_number > 0:

            section_points = (
                section_points[1:]
            )


        all_points.append(
            section_points
        )


    if not all_points:

        raise RuntimeError(

            "Forward model returned "
            "no backbone."
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
# FIXED DISPLAY RESAMPLING
# ============================================================

def resample_polyline(

    points,

    number_of_points=DISPLAY_POINTS

):

    """
    Resample the calculated robot centreline
    to a constant number of display points.

    This is only a VISUAL operation.

    It does not change the CTR mechanics.
    """

    points = np.asarray(

        points,

        dtype=float
    )


    if len(points) == 0:

        return np.zeros(

            (
                number_of_points,
                3
            ),

            dtype=float
        )


    if len(points) == 1:

        return np.repeat(

            points,

            number_of_points,

            axis=0
        )


    segment_lengths = (

        np.linalg.norm(

            np.diff(

                points,

                axis=0
            ),

            axis=1
        )
    )


    distance = np.concatenate([

        [0.0],

        np.cumsum(
            segment_lengths
        )
    ])


    total_distance = (
        distance[-1]
    )


    if total_distance <= 1e-12:

        return np.repeat(

            points[:1],

            number_of_points,

            axis=0
        )


    targets = np.linspace(

        0.0,

        total_distance,

        number_of_points
    )


    resampled = (
        np.column_stack([

            np.interp(

                targets,

                distance,

                points[:, 0]
            ),

            np.interp(

                targets,

                distance,

                points[:, 1]
            ),

            np.interp(

                targets,

                distance,

                points[:, 2]
            )
        ])
    )


    return resampled


# ============================================================
# CREATE ROBOT ACTORS ONCE
# ============================================================

initial_backbone_mm, current_tip_mm = (
    calculate_backbone()
)


initial_display_points = (
    resample_polyline(
        initial_backbone_mm
    )
)


backbone_poly = (
    pv.lines_from_points(
        initial_display_points
    )
)


backbone_actor = (
    plotter.add_mesh(

        backbone_poly,

        line_width=
            BACKBONE_LINE_WIDTH,

        render_lines_as_tubes=True,

        smooth_shading=False,

        reset_camera=False,

        name="ctr_backbone",

        render=False
    )
)


# Tip sphere is also created only once
tip_mesh = pv.Sphere(

    radius=
        TIP_RADIUS_MM,

    center=(
        0.0,
        0.0,
        0.0
    )
)


tip_actor = (
    plotter.add_mesh(

        tip_mesh,

        name="ctr_tip",

        reset_camera=False,

        render=False
    )
)


tip_actor.position = (
    tuple(
        current_tip_mm
    )
)


# Origin / insertion point
plotter.add_mesh(

    pv.Sphere(

        radius=2.5,

        center=(
            0.0,
            0.0,
            0.0
        )
    ),

    name="ctr_origin",

    reset_camera=False,

    render=False
)


# ============================================================
# DIAGNOSTICS
# ============================================================

last_left_x = 0.0

last_left_y = 0.0

last_right_x = 0.0

last_l2 = 0.0

last_r2 = 0.0


last_model_ms = 0.0


model_updates_in_window = 0

model_update_rate = 0.0


rate_window_start = (
    time.perf_counter()
)


last_error_text = ""


# ============================================================
# STATUS TEXT
# ============================================================

def make_status_text():

    ul_mm = (
        ul * 1000.0
    )


    rotations_deg = (
        np.rad2deg(
            uphi
        )
    )


    controller_text = (

        "CONNECTED"

        if controller is not None

        else "DISCONNECTED"
    )


    text = (

        f"ACTIVE TUBE: "
        f"{TUBE_NAMES[selected_tube]}\n"

        f"PS5: "
        f"{controller_text}\n\n"


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

        f"X = "
        f"{current_tip_mm[0]:7.1f} mm\n"

        f"Y = "
        f"{current_tip_mm[1]:7.1f} mm\n"

        f"Z = "
        f"{current_tip_mm[2]:7.1f} mm\n\n"


        f"LIVE INPUT\n"

        f"Left stick X/Y  = "
        f"{last_left_x:+.2f}, "
        f"{last_left_y:+.2f}\n"

        f"Right stick L/R = "
        f"{last_right_x:+.2f}\n"

        f"L2 / R2         = "
        f"{last_l2:.2f}, "
        f"{last_r2:.2f}\n\n"


        f"PERFORMANCE\n"

        f"Model calc      = "
        f"{last_model_ms:5.1f} ms\n"

        f"Model updates   = "
        f"{model_update_rate:4.1f} /s\n\n"


        f"PS5 CONTROLS\n"

        f"L1 / R1         "
        f"previous / next tube\n"

        f"L2              "
        f"retract selected tube\n"

        f"R2              "
        f"insert selected tube\n"

        f"Right stick L/R "
        f"rotate selected tube\n"

        f"Left stick      "
        f"orbit camera about origin\n"

        f"Cross           "
        f"reset robot\n\n"


        f"KEYBOARD BACKUP\n"

        f"1 / 2 / 3       "
        f"Inner / Middle / Outer\n"

        f"W / S           "
        f"insert / retract 1 mm\n"

        f"A / D           "
        f"rotate -/+ 5 deg\n"

        f"J / L           "
        f"camera left / right\n"

        f"I / K           "
        f"camera up / down\n"

        f"R               "
        f"reset robot\n"

        f"Q               "
        f"quit\n"

        f"Mouse           "
        f"free camera control"
    )


    if last_error_text:

        text += (

            "\n\nERROR: "

            + last_error_text
        )


    return text


# Create the text actor only ONCE.
status_actor = (
    plotter.add_text(

        make_status_text(),

        position="upper_left",

        font_size=9,

        name="status_text",

        render=False
    )
)


def update_status_text():

    # Because this is a CornerAnnotation,
    # use set_text rather than SetInput.
    status_actor.set_text(

        "upper_left",

        make_status_text()
    )


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

        minimum = (
            ul[1]
        )

        maximum = (
            TOTAL_LENGTHS[0]
        )


    elif tube_number == 1:

        minimum = (
            ul[2]
        )


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
# TRANSLATION COMMAND
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


    ul[selected_tube] = (
        np.clip(

            old_value

            + amount_mm
            / 1000.0,

            minimum,

            maximum
        )
    )


    changed = not np.isclose(

        old_value,

        ul[selected_tube]
    )


    if changed:

        robot_dirty = True


    return changed


# ============================================================
# ROTATION COMMAND
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


    # Wrap:
    #
    # -180 ... +180 deg
    uphi[selected_tube] = (

        (

            uphi[selected_tube]

            + np.pi

        )

        % (
            2.0
            * np.pi
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


    update_status_text()


def previous_tube():

    select_tube(

        selected_tube - 1
    )


def next_tube():

    select_tube(

        selected_tube + 1
    )


# ============================================================
# RESET ROBOT
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


    print(
        "Robot reset."
    )


# ============================================================
# CAMERA CONTROL
# ============================================================

# Back / insertion point of the tubes.
CAMERA_ORIGIN = np.array([

    0.0,

    0.0,

    0.0

], dtype=float)


def orbit_camera(

    delta_yaw_deg,

    delta_pitch_deg

):

    """
    Orbit around the CTR insertion point.

    Important:
    This does NOT lock the camera every frame.

    It only modifies it while the left stick
    or camera keyboard controls are used.
    """

    camera = (
        plotter.camera
    )


    position = np.asarray(

        camera.position,

        dtype=float
    )


    relative = (

        position

        - CAMERA_ORIGIN
    )


    radius = float(

        np.linalg.norm(
            relative
        )
    )


    if radius <= 1e-9:

        radius = 450.0


        relative = np.array([

            radius,

            0.0,

            0.0
        ])


    yaw = math.atan2(

        relative[1],

        relative[0]
    )


    pitch = math.asin(

        float(

            np.clip(

                relative[2]
                / radius,

                -1.0,

                1.0
            )
        )
    )


    yaw += math.radians(

        delta_yaw_deg
    )


    pitch += math.radians(

        delta_pitch_deg
    )


    # Prevent camera flipping upside down.
    pitch = float(

        np.clip(

            pitch,

            math.radians(
                -80.0
            ),

            math.radians(
                80.0
            )
        )
    )


    new_position = (

        CAMERA_ORIGIN

        + radius

        * np.array([

            math.cos(pitch)
            * math.cos(yaw),

            math.cos(pitch)
            * math.sin(yaw),

            math.sin(pitch)
        ])
    )


    camera.position = (
        tuple(
            new_position
        )
    )


    # Always look at the back of the tubes.
    camera.focal_point = (

        0.0,

        0.0,

        0.0
    )


    camera.up = (

        0.0,

        0.0,

        1.0
    )


# ============================================================
# FAST ROBOT VISUAL UPDATE
# ============================================================

def update_robot_visual():

    global current_tip_mm

    global robot_dirty

    global last_model_ms

    global model_updates_in_window

    global model_update_rate

    global rate_window_start


    start = (
        time.perf_counter()
    )


    backbone_mm, tip_mm = (
        calculate_backbone()
    )


    display_points = (
        resample_polyline(
            backbone_mm
        )
    )


    last_model_ms = (

        time.perf_counter()

        - start

    ) * 1000.0


    # ========================================================
    # UPDATE EXISTING BACKBONE
    #
    # No removing actor.
    # No adding actor.
    # No new tube mesh.
    # ========================================================

    backbone_poly.points = (
        display_points
    )


    backbone_poly.Modified()


    # ========================================================
    # UPDATE EXISTING TIP
    # ========================================================

    current_tip_mm = (
        tip_mm.copy()
    )


    tip_actor.position = (

        tuple(
            current_tip_mm
        )
    )


    robot_dirty = False


    # ========================================================
    # PERFORMANCE COUNTER
    # ========================================================

    model_updates_in_window += 1


    now = (
        time.perf_counter()
    )


    elapsed = (

        now

        - rate_window_start
    )


    if elapsed >= 1.0:

        model_update_rate = (

            model_updates_in_window

            / elapsed
        )


        model_updates_in_window = 0


        rate_window_start = (
            now
        )


# ============================================================
# KEYBOARD BACKUP
# ============================================================

# PyVista already assigns some keys such as
# W, S, R and 3 to its own rendering commands.
#
# Remove those defaults before assigning
# our robot controls.

for key in (

    "1",

    "2",

    "3",

    "w",

    "s",

    "a",

    "d",

    "r",

    "i",

    "j",

    "k",

    "l"

):

    try:

        plotter.iren.clear_events_for_key(
            key
        )

    except Exception:

        pass


# Tube selection
plotter.add_key_event(

    "1",

    lambda:
        select_tube(0)
)


plotter.add_key_event(

    "2",

    lambda:
        select_tube(1)
)


plotter.add_key_event(

    "3",

    lambda:
        select_tube(2)
)


# Linear movement
plotter.add_key_event(

    "w",

    lambda:
        change_translation_mm(
            +KEYBOARD_TRANSLATION_STEP_MM
        )
)


plotter.add_key_event(

    "s",

    lambda:
        change_translation_mm(
            -KEYBOARD_TRANSLATION_STEP_MM
        )
)


# Rotation
plotter.add_key_event(

    "a",

    lambda:
        change_rotation_deg(
            -KEYBOARD_ROTATION_STEP_DEG
        )
)


plotter.add_key_event(

    "d",

    lambda:
        change_rotation_deg(
            +KEYBOARD_ROTATION_STEP_DEG
        )
)


# Camera
plotter.add_key_event(

    "j",

    lambda:
        orbit_camera(
            -KEYBOARD_CAMERA_STEP_DEG,
            0.0
        )
)


plotter.add_key_event(

    "l",

    lambda:
        orbit_camera(
            +KEYBOARD_CAMERA_STEP_DEG,
            0.0
        )
)


plotter.add_key_event(

    "i",

    lambda:
        orbit_camera(
            0.0,
            +KEYBOARD_CAMERA_STEP_DEG
        )
)


plotter.add_key_event(

    "k",

    lambda:
        orbit_camera(
            0.0,
            -KEYBOARD_CAMERA_STEP_DEG
        )
)


plotter.add_key_event(

    "r",

    reset_robot
)


# Q is already PyVista's normal quit key.


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


# ============================================================
# CONTROLLER TIMER CALLBACK
# ============================================================

def controller_tick(

    step

):

    global controller

    global previous_l1
    global previous_r1
    global previous_cross

    global last_timer_time

    global last_model_update_time

    global last_status_update_time

    global last_controller_reconnect_attempt

    global last_left_x
    global last_left_y

    global last_right_x

    global last_l2
    global last_r2

    global last_error_text


    try:

        now = (
            time.perf_counter()
        )


        dt = min(

            now
            - last_timer_time,

            0.05
        )


        last_timer_time = (
            now
        )


        # ====================================================
        # VERY IMPORTANT FIX
        #
        # event.get() pumps AND removes events from the queue.
        #
        # The previous code only pumped the queue.
        # ====================================================

        pygame.event.get()


        # ====================================================
        # RECONNECT CONTROLLER IF NECESSARY
        # ====================================================

        if controller is None:

            if (

                now
                - last_controller_reconnect_attempt

                >= 1.0
            ):

                last_controller_reconnect_attempt = (
                    now
                )


                connect_controller()


        # ====================================================
        # CONTROLLER INPUT
        # ====================================================

        if controller is not None:

            try:

                # --------------------------------------------
                # L1 / R1 / CROSS
                # --------------------------------------------

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


                # ============================================
                # LEFT STICK = CAMERA
                # ============================================

                raw_left_x = (
                    controller.get_axis(
                        LEFT_STICK_X
                    )
                )


                raw_left_y = (
                    controller.get_axis(
                        LEFT_STICK_Y
                    )
                )


                (

                    last_left_x,

                    last_left_y

                ) = apply_stick_deadzone(

                    raw_left_x,

                    raw_left_y
                )


                if (

                    last_left_x != 0.0

                    or

                    last_left_y != 0.0

                ):

                    orbit_camera(

                        last_left_x

                        * CAMERA_ORBIT_SPEED_DEG_S

                        * dt,


                        -last_left_y

                        * CAMERA_ORBIT_SPEED_DEG_S

                        * dt
                    )


                # ============================================
                # RIGHT STICK L/R = TUBE ROTATION
                # ============================================

                last_right_x = (
                    apply_axis_deadzone(

                        controller.get_axis(
                            RIGHT_STICK_X
                        )
                    )
                )


                if last_right_x != 0.0:

                    change_rotation_deg(

                        last_right_x

                        * MAX_ROTATION_SPEED_DEG_S

                        * dt
                    )


                # Right stick up/down intentionally
                # does nothing.


                # ============================================
                # L2 / R2 = TRANSLATION
                # ============================================

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


                translation_command = (

                    last_r2

                    - last_l2
                )


                if (

                    abs(
                        translation_command
                    )

                    > 0.02

                ):

                    change_translation_mm(

                        translation_command

                        * MAX_TRANSLATION_SPEED_MM_S

                        * dt
                    )


                last_error_text = ""


            except pygame.error as exc:

                # If the controller is disconnected,
                # don't let the whole timer die.

                last_error_text = (

                    f"Controller lost: {exc}"
                )


                controller = None


        # ====================================================
        # UPDATE ROBOT
        # ====================================================

        if (

            robot_dirty

            and

            now
            - last_model_update_time

            >= MODEL_UPDATE_INTERVAL

        ):

            update_robot_visual()


            last_model_update_time = (
                now
            )


        # ====================================================
        # UPDATE STATUS
        # ====================================================

        if (

            now
            - last_status_update_time

            >= STATUS_UPDATE_INTERVAL

        ):

            update_status_text()


            last_status_update_time = (
                now
            )


        # ====================================================
        # IMPORTANT:
        #
        # NO plotter.render() HERE.
        #
        # Let PyVista/VTK's interactive event loop
        # perform the rendering.
        # ====================================================


    except Exception as exc:

        # If anything unexpected goes wrong in the
        # callback, print it rather than silently
        # killing the controller loop.

        last_error_text = (

            f"{type(exc).__name__}: {exc}"
        )


        print(

            "Controller timer error:",

            last_error_text
        )


# ============================================================
# INITIAL CAMERA
# ============================================================

plotter.camera_position = (
    "iso"
)


plotter.reset_camera()


plotter.camera.focal_point = (

    0.0,

    0.0,

    0.0
)


update_status_text()


# ============================================================
# TERMINAL KEY
# ============================================================

print("")
print("========================================")
print("PS5 CTR SIMULATOR - RESPONSIVE VERSION")
print("========================================")
print("")
print("PS5")
print("L1 / R1         previous / next tube")
print("L2              retract")
print("R2              insert")
print("Right stick L/R rotate selected tube")
print("Left stick      orbit camera")
print("Cross           reset robot")
print("")
print("KEYBOARD")
print("1 / 2 / 3       select tubes")
print("W / S           insert / retract 1 mm")
print("A / D           rotate -/+ 5 deg")
print("J / L           camera left / right")
print("I / K           camera up / down")
print("R               reset")
print("Q               quit")
print("")
print(
    "D-pad precision controls "
    "are intentionally not added yet."
)
print("")


# ============================================================
# START TIMER
# ============================================================

plotter.add_timer_event(

    max_steps=
        10_000_000,

    duration=
        TIMER_INTERVAL_MS,

    callback=
        controller_tick
)


# ============================================================
# START WINDOW
# ============================================================

try:

    plotter.show(

        title=
            "PS5 CTR Simulator"
    )


finally:

    pygame.quit()


    print("")
    print(
        "CTR simulator closed."
    )
