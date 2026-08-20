import pygame
import time
import math


# ============================================================
# SETTINGS
# ============================================================

# Your resting stick drift was roughly 0.01-0.04.
# 0.08 should comfortably remove it.
STICK_DEADZONE = 0.08

# Based on the values your Mac is currently reporting:
#
# Axis 0 = likely left stick X
# Axis 1 = likely left stick Y
# Axis 2 = likely right stick X
# Axis 3 = likely right stick Y
# Axis 4 = likely L2
# Axis 5 = likely R2
#
# We will confirm these by moving each control individually.

LEFT_STICK_X = 0
LEFT_STICK_Y = 1

RIGHT_STICK_X = 2
RIGHT_STICK_Y = 3

L2_AXIS = 4
R2_AXIS = 5


# ============================================================
# DEADZONE FUNCTION
# ============================================================

def apply_stick_deadzone(x, y, deadzone=STICK_DEADZONE):
    """
    Apply a radial deadzone to a joystick.

    Small stick movements around the centre are treated as zero.

    Outside the deadzone, the remaining range is rescaled
    so that full stick movement can still reach approximately 1.0.
    """

    magnitude = math.sqrt(
        x * x + y * y
    )

    # Inside deadzone -> completely zero
    if magnitude < deadzone:
        return 0.0, 0.0

    # Prevent magnitude exceeding 1 because of diagonal movement
    magnitude = min(
        magnitude,
        1.0
    )

    # Rescale range:
    #
    # deadzone ... 1
    #
    # becomes:
    #
    # 0 ... 1
    scaled_magnitude = (
        magnitude - deadzone
    ) / (
        1.0 - deadzone
    )

    scale = (
        scaled_magnitude
        / magnitude
    )

    filtered_x = (
        x * scale
    )

    filtered_y = (
        y * scale
    )

    return (
        filtered_x,
        filtered_y
    )


# ============================================================
# TRIGGER CONVERSION
# ============================================================

def convert_trigger(raw_value):
    """
    Convert trigger input:

        -1.0 = untouched
         0.0 = approximately half pressed
        +1.0 = fully pressed

    into:

         0.0 = untouched
         0.5 = half pressed
         1.0 = fully pressed
    """

    value = (
        raw_value + 1.0
    ) / 2.0

    # Prevent tiny numerical values outside 0-1
    value = max(
        0.0,
        min(
            1.0,
            value
        )
    )

    return value


# ============================================================
# INITIALISE PYGAME
# ============================================================

pygame.init()

pygame.joystick.init()


print("")
print("========================================")
print("PS5 CONTROLLER TEST WITH DEADZONE")
print("========================================")
print("")

print(
    f"Stick deadzone = {STICK_DEADZONE:.2f}"
)

print("")


# ============================================================
# FIND CONTROLLER
# ============================================================

controller_count = (
    pygame.joystick.get_count()
)


print(
    f"Controllers detected: {controller_count}"
)


if controller_count == 0:

    print("")
    print("No controller detected.")
    print("")
    print(
        "Connect your PS5 controller and "
        "run this script again."
    )

    pygame.quit()

    raise SystemExit


# ============================================================
# OPEN FIRST CONTROLLER
# ============================================================

controller = (
    pygame.joystick.Joystick(0)
)

controller.init()


print("")

print(
    "Controller name:",
    controller.get_name()
)

print(
    "Number of axes:",
    controller.get_numaxes()
)

print(
    "Number of buttons:",
    controller.get_numbuttons()
)

print(
    "Number of hats:",
    controller.get_numhats()
)


print("")
print("----------------------------------------")
print("Expected mapping based on your Mac:")
print("----------------------------------------")

print(
    f"Axis {LEFT_STICK_X} = Left stick X"
)

print(
    f"Axis {LEFT_STICK_Y} = Left stick Y"
)

print(
    f"Axis {RIGHT_STICK_X} = Right stick X"
)

print(
    f"Axis {RIGHT_STICK_Y} = Right stick Y"
)

print(
    f"Axis {L2_AXIS} = L2 trigger"
)

print(
    f"Axis {R2_AXIS} = R2 trigger"
)


print("")
print("----------------------------------------")
print("Move ONE control at a time.")
print("")
print("Press CTRL+C in Terminal to stop.")
print("----------------------------------------")
print("")


# ============================================================
# MAIN CONTROLLER LOOP
# ============================================================

try:

    while True:

        # Required for pygame to receive controller updates
        pygame.event.pump()


        # ====================================================
        # RAW AXIS VALUES
        # ====================================================

        raw_axes = []

        for i in range(
            controller.get_numaxes()
        ):

            raw_value = (
                controller.get_axis(i)
            )

            raw_axes.append(
                raw_value
            )


        # ====================================================
        # LEFT STICK
        # ====================================================

        raw_left_x = (
            raw_axes[LEFT_STICK_X]
        )

        raw_left_y = (
            raw_axes[LEFT_STICK_Y]
        )


        (
            left_x,
            left_y

        ) = apply_stick_deadzone(

            raw_left_x,
            raw_left_y
        )


        # ====================================================
        # RIGHT STICK
        # ====================================================

        raw_right_x = (
            raw_axes[RIGHT_STICK_X]
        )

        raw_right_y = (
            raw_axes[RIGHT_STICK_Y]
        )


        (
            right_x,
            right_y

        ) = apply_stick_deadzone(

            raw_right_x,
            raw_right_y
        )


        # ====================================================
        # TRIGGERS
        # ====================================================

        raw_l2 = (
            raw_axes[L2_AXIS]
        )

        raw_r2 = (
            raw_axes[R2_AXIS]
        )


        l2 = convert_trigger(
            raw_l2
        )

        r2 = convert_trigger(
            raw_r2
        )


        # ====================================================
        # BUTTONS
        # ====================================================

        buttons_pressed = []

        for i in range(
            controller.get_numbuttons()
        ):

            if controller.get_button(i):

                buttons_pressed.append(i)


        # ====================================================
        # D-PAD / HAT
        # ====================================================

        hats = []

        for i in range(
            controller.get_numhats()
        ):

            hats.append(
                controller.get_hat(i)
            )


        # ====================================================
        # DISPLAY
        # ====================================================

        raw_text = (

            f"RAW "
            f"L({raw_left_x:+.3f},{raw_left_y:+.3f}) "
            f"R({raw_right_x:+.3f},{raw_right_y:+.3f}) "
            f"L2={raw_l2:+.3f} "
            f"R2={raw_r2:+.3f}"
        )


        filtered_text = (

            f" | FILTERED "
            f"L({left_x:+.3f},{left_y:+.3f}) "
            f"R({right_x:+.3f},{right_y:+.3f}) "
            f"L2={l2:.3f} "
            f"R2={r2:.3f}"
        )


        button_text = (

            f" | Buttons={buttons_pressed}"
            f" Hats={hats}"
        )


        print(

            "\r"
            + raw_text
            + filtered_text
            + button_text
            + " " * 20,

            end="",

            flush=True
        )


        # 20 updates per second
        time.sleep(
            0.05
        )


# ============================================================
# CLEAN EXIT
# ============================================================

except KeyboardInterrupt:

    print("")
    print("")
    print("Controller test stopped.")


finally:

    pygame.quit()