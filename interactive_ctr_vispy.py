"""Real-time VisPy viewer for the concentric-tube robot.

Controls
--------
PS5: L1/R1 select tube, D-pad Up/Down rotate, D-pad Left/Right move,
Cross resets.  Mouse drag orbits the view and the wheel zooms.
Keyboard: 1/2/3 select, W/S move, A/D rotate, R reset, Q quit.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pygame
from vispy import app, scene
from vispy.visuals import TextVisual
from vispy.visuals.transforms import STTransform

from CTR_superPosKin_fun_sectioned import superPosKin
from tube_parameters import build_supervisor_ctr_parameters


app.use_app(os.environ.get("VISPY_APP", "glfw"))

BUTTON_CROSS, BUTTON_L1, BUTTON_R1 = 0, 9, 10
DPAD_UP_BUTTON, DPAD_DOWN_BUTTON = 11, 12
DPAD_LEFT_BUTTON, DPAD_RIGHT_BUTTON = 13, 14
RIGHT_STICK_X, RIGHT_STICK_Y = 2, 3

DPAD_ROTATION_SPEED_DEG_S = 90.0
DPAD_TRANSLATION_SPEED_MM_S = 30.0
CAMERA_ORBIT_SPEED_DEG_S = 90.0
STICK_DEADZONE = 0.10
KEYBOARD_TRANSLATION_STEP_MM = 1.0
KEYBOARD_ROTATION_STEP_DEG = 5.0

TIMER_INTERVAL_S = 1.0 / 60.0
MODEL_UPDATE_HZ = 40.0
STATUS_UPDATE_HZ = 2.0
MODEL_POINTS_PER_SECTION = 8
DISPLAY_POINTS = 48
CONTROLLER_RETRY_S = 1.0
ERROR_PRINT_INTERVAL_S = 2.0

TUBE_NAMES = ("INNER", "MIDDLE", "OUTER")
TUBE_COLOUR_NAMES = ("BLUE", "GREEN", "ORANGE")
TUBE_COLOURS = (
    (0.05, 0.32, 0.9, 1.0),
    (0.05, 0.65, 0.28, 1.0),
    (0.95, 0.42, 0.05, 1.0),
)
TUBE_LINE_WIDTHS = (4.0, 6.0, 8.0)
INITIAL_DEPLOYMENT_M = np.array([120e-3, 70e-3, 40e-3])
INITIAL_ROTATION_RAD = np.zeros(3)


def resample_polyline(points: np.ndarray, count: int = DISPLAY_POINTS) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if count < 2:
        raise ValueError("count must be at least 2")
    if len(points) == 0:
        return np.zeros((count, 3), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, count, axis=0).astype(np.float32)
    distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    if distance[-1] <= 1e-12:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    targets = np.linspace(0.0, distance[-1], count)
    result = np.column_stack(
        [np.interp(targets, distance, points[:, axis]) for axis in range(3)]
    )
    return result.astype(np.float32)


def make_floor_grid(extent: float = 200.0, spacing: float = 50.0) -> np.ndarray:
    """Return XY floor-grid vertices connected as independent segments."""
    values = np.arange(-extent, extent + spacing * 0.5, spacing, dtype=np.float32)
    segments = []
    for value in values:
        segments.extend(([-extent, value, 0.0], [extent, value, 0.0]))
        segments.extend(([value, -extent, 0.0], [value, extent, 0.0]))
    return np.asarray(segments, dtype=np.float32)


def apply_stick_deadzone(
    x: float, y: float, deadzone: float = STICK_DEADZONE
) -> tuple[float, float]:
    """Apply a radial deadzone and rescale the remaining stick travel."""
    magnitude = math.hypot(x, y)
    if magnitude <= deadzone:
        return 0.0, 0.0
    scale = ((min(magnitude, 1.0) - deadzone) / (1.0 - deadzone)) / magnitude
    return x * scale, y * scale


def polyline_interval(
    points: np.ndarray, start_mm: float, end_mm: float
) -> np.ndarray:
    """Extract an arc-length interval from a centreline."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.repeat(points[:1], 2, axis=0).astype(np.float32)
    distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    total = float(distance[-1])
    start = float(np.clip(start_mm, 0.0, total))
    end = float(np.clip(end_mm, start, total))
    count = max(2, int(DISPLAY_POINTS * (end - start) / max(total, 1e-12)) + 2)
    targets = np.linspace(start, end, count)
    result = np.column_stack(
        [np.interp(targets, distance, points[:, axis]) for axis in range(3)]
    )
    return result.astype(np.float32)


def visible_tube_segments(
    backbone: np.ndarray, deployment_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the physically visible inner, middle, and outer tube sections."""
    inner_tip, middle_tip, outer_tip = np.asarray(deployment_m) * 1000.0
    inner = polyline_interval(backbone, middle_tip, inner_tip)
    middle = polyline_interval(backbone, outer_tip, middle_tip)
    outer = polyline_interval(backbone, 0.0, outer_tip)
    return inner, middle, outer


@dataclass
class InputSnapshot:
    dpad_x: int = 0
    dpad_y: int = 0
    right_x: float = 0.0
    right_y: float = 0.0


class ControllerManager:
    def __init__(self) -> None:
        self.joystick = None
        self.previous = {BUTTON_L1: False, BUTTON_R1: False, BUTTON_CROSS: False}
        self.last_retry = -CONTROLLER_RETRY_S
        self.connect()

    @property
    def connected(self) -> bool:
        return self.joystick is not None and self.joystick.get_init()

    def connect(self, now: float | None = None) -> bool:
        now = time.perf_counter() if now is None else now
        self.last_retry = now
        if self.connected:
            return True
        if pygame.joystick.get_count() == 0:
            self.joystick = None
            return False

        joystick = pygame.joystick.Joystick(0)
        buttons = joystick.get_numbuttons()
        hats = joystick.get_numhats()
        has_dpad_buttons = buttons > DPAD_RIGHT_BUTTON
        if (
            buttons <= BUTTON_R1
            or joystick.get_numaxes() <= RIGHT_STICK_Y
            or not (hats > 0 or has_dpad_buttons)
        ):
            joystick.quit()
            raise RuntimeError(
                f"Controller has {buttons} buttons and {hats} hats; "
                "the expected PS5 D-pad mapping was not found."
            )
        self.joystick = joystick
        self.previous = dict.fromkeys(self.previous, False)
        print(f"Controller connected: {joystick.get_name()}")
        return True

    def disconnect(self) -> None:
        if self.joystick is not None:
            try:
                self.joystick.quit()
            except pygame.error:
                pass
        self.joystick = None
        self.previous = dict.fromkeys(self.previous, False)

    def process_events(self, now: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEREMOVED and self.connected:
                instance_id = self.joystick.get_instance_id()
                if getattr(event, "instance_id", instance_id) == instance_id:
                    print("Controller disconnected; keyboard control remains available.")
                    self.disconnect()
            elif event.type == pygame.JOYDEVICEADDED and not self.connected:
                self.connect(now)
        if not self.connected and now - self.last_retry >= CONTROLLER_RETRY_S:
            self.connect(now)

    def pressed(self, button: int) -> bool:
        if not self.connected:
            return False
        current = bool(self.joystick.get_button(button))
        rising_edge = current and not self.previous[button]
        self.previous[button] = current
        return rising_edge

    def read_inputs(self) -> InputSnapshot:
        if not self.connected:
            return InputSnapshot()
        dpad_x = 0
        dpad_y = 0
        if self.joystick.get_numhats() > 0:
            dpad_x, dpad_y = self.joystick.get_hat(0)
        if not (dpad_x or dpad_y) and self.joystick.get_numbuttons() > DPAD_RIGHT_BUTTON:
            dpad_x = int(self.joystick.get_button(DPAD_RIGHT_BUTTON)) - int(
                self.joystick.get_button(DPAD_LEFT_BUTTON)
            )
            dpad_y = int(self.joystick.get_button(DPAD_UP_BUTTON)) - int(
                self.joystick.get_button(DPAD_DOWN_BUTTON)
            )
        right_x, right_y = apply_stick_deadzone(
            self.joystick.get_axis(RIGHT_STICK_X),
            self.joystick.get_axis(RIGHT_STICK_Y),
        )
        return InputSnapshot(int(dpad_x), int(dpad_y), right_x, right_y)


class SidebarPanel(scene.Widget):
    """A fixed-width screen-space panel whose text cannot drift off-canvas."""

    def __init__(self, text: str) -> None:
        self.label = TextVisual(
            text=text,
            color=(0.05, 0.05, 0.05, 1.0),
            font_size=8,
            anchor_x="left",
            anchor_y="bottom",
        )
        super().__init__(
            bgcolor=(0.965, 0.97, 0.98, 1.0),
            border_color=(0.72, 0.75, 0.8, 1.0),
            border_width=1,
        )
        self.add_subvisual(self.label)
        self.events.resize.connect(self._position_text)
        self._position_text()

    def _position_text(self, _event=None) -> None:
        self.label.pos = (16, 16)


class VisPyCTRViewer:
    def __init__(self) -> None:
        pygame.init()
        pygame.joystick.init()
        self.controller = ControllerManager()

        self.parameters = build_supervisor_ctr_parameters()
        self.sim_parameters = {"n_p": MODEL_POINTS_PER_SECTION, "isPlot": False}
        self.total_lengths = np.array(
            [sum(lengths) for lengths in self.parameters["l_t"]], dtype=float
        )
        self.deployment = INITIAL_DEPLOYMENT_M.copy()
        self.rotation = INITIAL_ROTATION_RAD.copy()
        self.selected_tube = 0
        self.inputs = InputSnapshot()
        self.robot_dirty = True
        self.error_text = ""
        self.last_error_print = -ERROR_PRINT_INTERVAL_S

        self.last_model_ms = 0.0
        self.ui_rate = 0.0
        self.robot_rate = 0.0
        self.ui_count = 0
        self.robot_count = 0
        self.rate_start = time.perf_counter()

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            title="PS5 CTR Simulator — VisPy",
            size=(1200, 800),
            bgcolor="white",
            dpi=96.0,
            show=False,
        )
        self.layout = self.canvas.central_widget.add_grid(spacing=0)
        self.view = self.layout.add_view(row=0, col=0, bgcolor="white")
        self.view.width_min = 600
        self.view.camera = scene.TurntableCamera(
            fov=45.0,
            azimuth=45.0,
            elevation=25.0,
            distance=500.0,
            center=(0.0, 0.0, 100.0),
            up="+z",
        )

        backbone, self.tip_mm = self.calculate_backbone()
        segments = visible_tube_segments(backbone, self.deployment)
        self.tube_lines = []
        # Outer is created first so the smaller distal tubes remain visible.
        for tube in (2, 1, 0):
            line = scene.visuals.Line(
                pos=segments[tube],
                color=TUBE_COLOURS[tube],
                width=TUBE_LINE_WIDTHS[tube],
                connect="strip",
                method="gl",
                antialias=True,
                parent=self.view.scene,
            )
            self.tube_lines.append((tube, line))
        self.tip = scene.visuals.Markers(parent=self.view.scene)
        self.tip.set_data(
            np.asarray([self.tip_mm], dtype=np.float32),
            face_color=(0.9, 0.2, 0.15, 1.0),
            edge_color=(0.45, 0.05, 0.03, 1.0),
            size=14.0,
        )
        self.origin = scene.visuals.Markers(parent=self.view.scene)
        self.origin.set_data(
            np.zeros((1, 3), dtype=np.float32),
            face_color=(0.15, 0.15, 0.15, 1.0),
            size=10.0,
        )
        scene.visuals.Line(
            pos=make_floor_grid(),
            color=(0.78, 0.8, 0.84, 0.75),
            width=1.0,
            connect="segments",
            method="gl",
            parent=self.view.scene,
        )
        axes = scene.visuals.XYZAxis(parent=self.view.scene)
        axes.transform = STTransform(scale=(45.0, 45.0, 45.0))

        self.sidebar = SidebarPanel(self.sidebar_text())
        self.sidebar.width_min = 360
        self.sidebar.width_max = 360
        self.layout.add_widget(self.sidebar, row=0, col=1)

        now = time.perf_counter()
        self.last_tick = now
        self.last_model_update = now
        self.last_status_update = now
        self.timer = app.Timer(
            interval=TIMER_INTERVAL_S,
            connect=self.tick,
            start=False,
        )
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.close.connect(self.on_close)
        self.closed = False

    def calculate_backbone(self) -> tuple[np.ndarray, np.ndarray]:
        result = superPosKin(
            self.parameters,
            {"ul": self.deployment.tolist(), "uphi": self.rotation.tolist()},
            self.sim_parameters,
        )
        tip, sections = result[0], result[3]
        points = []
        for index, section in enumerate(sections):
            part = np.column_stack([np.asarray(section[axis]) for axis in range(3)])
            if index:
                part = part[1:]
            if len(part):
                points.append(part)
        if not points:
            raise RuntimeError("Forward model returned no backbone.")
        return np.vstack(points) * 1000.0, np.asarray(tip[:3]) * 1000.0

    def translation_limits(self, tube: int) -> tuple[float, float]:
        if tube == 0:
            return self.deployment[1], self.total_lengths[0]
        if tube == 1:
            return self.deployment[2], min(self.deployment[0], self.total_lengths[1])
        return 0.0, min(self.deployment[1], self.total_lengths[2])

    def move(self, amount_mm: float) -> None:
        low, high = self.translation_limits(self.selected_tube)
        old = self.deployment[self.selected_tube]
        new = float(np.clip(old + amount_mm / 1000.0, low, high))
        if not np.isclose(old, new, atol=1e-12):
            self.deployment[self.selected_tube] = new
            self.robot_dirty = True

    def rotate(self, amount_deg: float) -> None:
        index = self.selected_tube
        old = self.rotation[index]
        new = (old + math.radians(amount_deg) + math.pi) % (2 * math.pi) - math.pi
        if not np.isclose(old, new, atol=1e-12):
            self.rotation[index] = new
            self.robot_dirty = True

    def select(self, index: int) -> None:
        self.selected_tube = index % len(TUBE_NAMES)
        self.update_status()

    def reset(self) -> None:
        self.deployment[:] = INITIAL_DEPLOYMENT_M
        self.rotation[:] = INITIAL_ROTATION_RAD
        self.selected_tube = 0
        self.robot_dirty = True
        print("Robot reset.")

    def update_robot(self) -> None:
        started = time.perf_counter()
        backbone, tip = self.calculate_backbone()
        segments = visible_tube_segments(backbone, self.deployment)
        self.last_model_ms = (time.perf_counter() - started) * 1000.0
        for tube, line in self.tube_lines:
            line.set_data(pos=segments[tube])
        self.tip_mm = tip.copy()
        self.tip.set_data(
            np.asarray([tip], dtype=np.float32),
            face_color=(0.9, 0.2, 0.15, 1.0),
            edge_color=(0.45, 0.05, 0.03, 1.0),
            size=14.0,
        )
        self.robot_dirty = False
        self.robot_count += 1
        self.canvas.update()

    def update_rates(self, now: float) -> None:
        self.ui_count += 1
        elapsed = now - self.rate_start
        if elapsed >= 1.0:
            self.ui_rate = self.ui_count / elapsed
            self.robot_rate = self.robot_count / elapsed
            self.ui_count = 0
            self.robot_count = 0
            self.rate_start = now

    def sidebar_text(self) -> str:
        mm = self.deployment * 1000.0
        deg = np.rad2deg(self.rotation)
        state = "CONNECTED" if self.controller.connected else "DISCONNECTED"
        markers = [">" if index == self.selected_tube else " " for index in range(3)]
        text = f"""ROBOT STATE

ACTIVE TUBE: {TUBE_NAMES[self.selected_tube]}
PS5: {state}

              EXTENSION     ROTATION
{markers[0]} INNER      {mm[0]:6.1f} mm     {deg[0]:7.1f} deg
{markers[1]} MIDDLE     {mm[1]:6.1f} mm     {deg[1]:7.1f} deg
{markers[2]} OUTER      {mm[2]:6.1f} mm     {deg[2]:7.1f} deg

TUBE COLOURS
INNER   Blue
MIDDLE  Green
OUTER   Orange

TIP POSITION
X {self.tip_mm[0]:7.1f} mm
Y {self.tip_mm[1]:7.1f} mm
Z {self.tip_mm[2]:7.1f} mm

PERFORMANCE
Model calculation   {self.last_model_ms:5.1f} ms
UI refresh          {self.ui_rate:5.1f} FPS
Robot updates       {self.robot_rate:5.1f} /s

{self.control_key_text()}"""
        return text + (f"\n\nERROR\n{self.error_text}" if self.error_text else "")

    @staticmethod
    def control_key_text() -> str:
        return """CONTROLLER
L1 / R1       Previous / next tube
D-pad Up      Increase angle
D-pad Down    Decrease angle
D-pad Left    Decrease extension
D-pad Right   Increase extension
Right stick   Change view orientation
Left stick    Unused
Cross         Reset robot

MOUSE
Drag          Change view
Wheel         Zoom

KEYBOARD
1 / 2 / 3     Inner / Middle / Outer
W / S         Increase / decrease extension
A / D         Decrease / increase angle
R             Reset robot
Q             Quit"""

    def update_status(self) -> None:
        self.sidebar.label.text = self.sidebar_text()
        self.canvas.update()

    def record_error(self, exc: Exception, now: float) -> None:
        self.error_text = f"{type(exc).__name__}: {exc}"
        if now - self.last_error_print >= ERROR_PRINT_INTERVAL_S:
            print(f"Controller update error: {self.error_text}")
            self.last_error_print = now

    def tick(self, _event=None) -> None:
        now = time.perf_counter()
        dt = float(np.clip(now - self.last_tick, 0.0, 0.05))
        self.last_tick = now
        try:
            self.controller.process_events(now)
            if self.controller.connected:
                if self.controller.pressed(BUTTON_L1):
                    self.select(self.selected_tube - 1)
                if self.controller.pressed(BUTTON_R1):
                    self.select(self.selected_tube + 1)
                if self.controller.pressed(BUTTON_CROSS):
                    self.reset()
                self.inputs = self.controller.read_inputs()
                if self.inputs.dpad_y:
                    self.rotate(self.inputs.dpad_y * DPAD_ROTATION_SPEED_DEG_S * dt)
                if self.inputs.dpad_x:
                    self.move(self.inputs.dpad_x * DPAD_TRANSLATION_SPEED_MM_S * dt)
                if self.inputs.right_x or self.inputs.right_y:
                    camera = self.view.camera
                    camera.azimuth += (
                        self.inputs.right_x * CAMERA_ORBIT_SPEED_DEG_S * dt
                    )
                    camera.elevation = float(
                        np.clip(
                            camera.elevation
                            - self.inputs.right_y * CAMERA_ORBIT_SPEED_DEG_S * dt,
                            -85.0,
                            85.0,
                        )
                    )
                    self.canvas.update()
            else:
                self.inputs = InputSnapshot()

            if self.robot_dirty and now - self.last_model_update >= 1 / MODEL_UPDATE_HZ:
                self.update_robot()
                self.last_model_update = now
            if now - self.last_status_update >= 1 / STATUS_UPDATE_HZ:
                self.update_status()
                self.last_status_update = now
            self.error_text = ""
        except pygame.error as exc:
            self.controller.disconnect()
            self.inputs = InputSnapshot()
            self.record_error(exc, now)
        except Exception as exc:
            self.record_error(exc, now)
        finally:
            self.update_rates(now)

    def on_key_press(self, event) -> None:
        key = event.key.name.lower() if event.key is not None else ""
        actions = {
            "1": lambda: self.select(0),
            "2": lambda: self.select(1),
            "3": lambda: self.select(2),
            "w": lambda: self.move(KEYBOARD_TRANSLATION_STEP_MM),
            "s": lambda: self.move(-KEYBOARD_TRANSLATION_STEP_MM),
            "a": lambda: self.rotate(-KEYBOARD_ROTATION_STEP_DEG),
            "d": lambda: self.rotate(KEYBOARD_ROTATION_STEP_DEG),
            "r": self.reset,
            "q": self.canvas.close,
        }
        action = actions.get(key)
        if action is not None:
            action()

    def on_close(self, _event=None) -> None:
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        self.controller.disconnect()
        pygame.quit()

    def run(self) -> None:
        print(
            "\nVISPY PS5 CTR SIMULATOR\n"
            "L1/R1 select | D-pad Up/Down rotate | D-pad Left/Right move | "
            "mouse orbit/zoom | Cross reset\n"
        )
        self.canvas.show()
        self.timer.start()
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.on_close()
            print("VisPy CTR simulator closed.")


def main() -> None:
    VisPyCTRViewer().run()


if __name__ == "__main__":
    main()
