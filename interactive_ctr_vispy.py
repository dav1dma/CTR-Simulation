"""Real-time VisPy viewer for the concentric-tube robot.

Controls
--------
PS5: L1/R1 select tube, D-pad Up/Down rotate, D-pad Left/Right move,
right stick orbits and L2/R2 zoom.  Mouse drag also orbits the view.
Keyboard: 1/2/3 select, W/S move, A/D rotate, R reset, Q quit.
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
from vispy import app, io, scene
from vispy.visuals import TextVisual
from vispy.visuals.transforms import STTransform

from CTR_superPosKin_fun_sectioned import superPosKin
from tube_parameters import build_supervisor_ctr_parameters


app.use_app(os.environ.get("VISPY_APP", "glfw"))

BUTTON_CROSS, BUTTON_CIRCLE, BUTTON_SQUARE, BUTTON_TRIANGLE = 0, 1, 2, 3
BUTTON_OPTIONS, BUTTON_R3, BUTTON_L1, BUTTON_R1 = 6, 8, 9, 10
DPAD_UP_BUTTON, DPAD_DOWN_BUTTON = 11, 12
DPAD_LEFT_BUTTON, DPAD_RIGHT_BUTTON = 13, 14
BUTTON_TOUCHPAD = 15
RIGHT_STICK_X, RIGHT_STICK_Y = 2, 3
L2_AXIS, R2_AXIS = 4, 5

DPAD_ROTATION_SPEED_DEG_S = 90.0
DPAD_TRANSLATION_SPEED_MM_S = 30.0
PRECISION_ROTATION_SPEED_DEG_S = 10.0
PRECISION_TRANSLATION_SPEED_MM_S = 5.0
CAMERA_ORBIT_SPEED_DEG_S = 90.0
CAMERA_ZOOM_RATE_S = 1.5
STICK_DEADZONE = 0.10
TRIGGER_DEADZONE = 0.02
TOUCHPAD_RESET_HOLD_S = 1.0
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
TUBE_STRIPE_OFFSETS_MM = (1.0, 1.5, 2.0)
TUBE_STRIPE_COLOUR = (0.08, 0.08, 0.08, 1.0)
TUBE_STRIPE_WIDTH = 1.5
TIP_MARKER_SIZE = 8.0
TIP_ARROW_LENGTH_MM = 30.0
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


def trigger_value(raw_value: float) -> float:
    """Convert the observed PS5 trigger range to 0 released .. 1 pressed."""
    return float(np.clip((raw_value + 1.0) * 0.5, 0.0, 1.0))


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


def tube_surface_stripe(
    centreline: np.ndarray, rotation_rad: float, offset_mm: float
) -> np.ndarray:
    """Offset a stripe from a tube centreline using a rotation-minimising frame."""
    points = np.asarray(centreline, dtype=float)
    if len(points) < 2:
        return points.astype(np.float32)

    tangents = np.gradient(points, axis=0)
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    for index in range(len(tangents)):
        if tangent_lengths[index] > 1e-9:
            tangents[index] /= tangent_lengths[index]
        elif index:
            tangents[index] = tangents[index - 1]
        else:
            tangents[index] = (0.0, 0.0, 1.0)

    first_tangent = tangents[0]
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(first_tangent, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first_normal = reference - np.dot(reference, first_tangent) * first_tangent
    first_normal /= np.linalg.norm(first_normal)

    normals = np.empty_like(tangents)
    normals[0] = first_normal
    for index in range(1, len(points)):
        previous_tangent = tangents[index - 1]
        tangent = tangents[index]
        axis = np.cross(previous_tangent, tangent)
        sine = float(np.linalg.norm(axis))
        cosine = float(np.clip(np.dot(previous_tangent, tangent), -1.0, 1.0))
        normal = normals[index - 1]
        if sine > 1e-9:
            axis /= sine
            normal = (
                normal * cosine
                + np.cross(axis, normal) * sine
                + axis * np.dot(axis, normal) * (1.0 - cosine)
            )
        normal -= np.dot(normal, tangent) * tangent
        normal_length = float(np.linalg.norm(normal))
        normals[index] = (
            normal / normal_length if normal_length > 1e-9 else normals[index - 1]
        )

    binormals = np.cross(tangents, normals)
    radial = math.cos(rotation_rad) * normals + math.sin(rotation_rad) * binormals
    return (points + float(offset_mm) * radial).astype(np.float32)


def make_vertical_reference(
    length_mm: float, dash_mm: float = 7.0, gap_mm: float = 5.0
) -> np.ndarray:
    """Create a dashed global +Z reference beginning at the tube base."""
    length = max(0.0, float(length_mm))
    vertices = []
    start = 0.0
    while start < length:
        end = min(start + dash_mm, length)
        vertices.extend(([0.0, 0.0, start], [0.0, 0.0, end]))
        start += dash_mm + gap_mm
    if not vertices:
        vertices = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return np.asarray(vertices, dtype=np.float32)


def tip_arrow_geometry(
    tip_mm: np.ndarray,
    direction: np.ndarray,
    length_mm: float = TIP_ARROW_LENGTH_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the arrow body and arrow-head direction vertices."""
    tip = np.asarray(tip_mm, dtype=float).reshape(3)
    unit = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(unit))
    unit = np.array([0.0, 0.0, 1.0]) if norm <= 1e-12 else unit / norm
    end = tip + float(length_mm) * unit
    body = np.asarray([tip, end], dtype=np.float32)
    arrows = np.asarray([[*tip, *end]], dtype=np.float32)
    return body, arrows


def orientation_from_vertical(direction: np.ndarray) -> tuple[float, float]:
    """Return tip tilt from global +Z and XY azimuth, both in degrees."""
    unit = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-12:
        return 0.0, 0.0
    unit /= norm
    tilt = math.degrees(math.acos(float(np.clip(unit[2], -1.0, 1.0))))
    horizontal = math.hypot(unit[0], unit[1])
    azimuth = 0.0 if horizontal <= 1e-12 else math.degrees(math.atan2(unit[1], unit[0]))
    return tilt, azimuth


@dataclass
class InputSnapshot:
    dpad_x: int = 0
    dpad_y: int = 0
    right_x: float = 0.0
    right_y: float = 0.0
    l2: float = 0.0
    r2: float = 0.0
    precision: bool = False


class ControllerManager:
    def __init__(self) -> None:
        self.joystick = None
        self.previous = dict.fromkeys(
            (
                BUTTON_CROSS,
                BUTTON_CIRCLE,
                BUTTON_TRIANGLE,
                BUTTON_OPTIONS,
                BUTTON_R3,
                BUTTON_L1,
                BUTTON_R1,
            ),
            False,
        )
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
            buttons <= BUTTON_TOUCHPAD
            or joystick.get_numaxes() <= R2_AXIS
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

    def down(self, button: int) -> bool:
        return self.connected and bool(self.joystick.get_button(button))

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
        return InputSnapshot(
            dpad_x=int(dpad_x),
            dpad_y=int(dpad_y),
            right_x=right_x,
            right_y=right_y,
            l2=trigger_value(self.joystick.get_axis(L2_AXIS)),
            r2=trigger_value(self.joystick.get_axis(R2_AXIS)),
            precision=self.down(BUTTON_SQUARE),
        )


class SidebarPanel(scene.Widget):
    """A fixed-width screen-space panel whose text cannot drift off-canvas."""

    def __init__(self, text: str) -> None:
        self.label = TextVisual(
            text=text,
            color=(0.05, 0.05, 0.05, 1.0),
            font_size=7,
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
        self.undo_stack = []
        self.waypoints = []
        self.previous_dpad = (0, 0)
        self.touchpad_hold_start = None
        self.touchpad_reset_fired = False
        self.ui_visible = True
        self.guides_visible = True
        self.last_action = "Ready"

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

        backbone, self.tip_mm, self.tip_direction = self.calculate_backbone()
        self.tip_tilt_deg, self.tip_azimuth_deg = orientation_from_vertical(
            self.tip_direction
        )
        segments = visible_tube_segments(backbone, self.deployment)

        self.vertical_reference = scene.visuals.Line(
            pos=make_vertical_reference(self.deployment[0] * 1000.0),
            color=(0.42, 0.45, 0.5, 0.75),
            width=2.0,
            connect="segments",
            method="gl",
            antialias=True,
            parent=self.view.scene,
        )
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
        self.rotation_stripes = []
        for tube in (2, 1, 0):
            stripe = scene.visuals.Line(
                pos=tube_surface_stripe(
                    segments[tube],
                    self.rotation[tube],
                    TUBE_STRIPE_OFFSETS_MM[tube],
                ),
                color=TUBE_STRIPE_COLOUR,
                width=TUBE_STRIPE_WIDTH,
                connect="strip",
                method="gl",
                antialias=True,
                parent=self.view.scene,
            )
            self.rotation_stripes.append((tube, stripe))
        self.tip = scene.visuals.Markers(parent=self.view.scene)
        self.tip.set_data(
            np.asarray([self.tip_mm], dtype=np.float32),
            face_color=(0.9, 0.2, 0.15, 1.0),
            edge_color=(0.45, 0.05, 0.03, 1.0),
            size=TIP_MARKER_SIZE,
        )
        arrow_body, arrow_heads = tip_arrow_geometry(
            self.tip_mm, self.tip_direction
        )
        self.tip_arrow = scene.visuals.Arrow(
            pos=arrow_body,
            arrows=arrow_heads,
            color=(0.55, 0.05, 0.75, 1.0),
            arrow_color=(0.55, 0.05, 0.75, 1.0),
            width=3.0,
            connect="strip",
            method="gl",
            antialias=True,
            arrow_type="triangle_60",
            arrow_size=10.0,
            parent=self.view.scene,
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

    def calculate_backbone(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = superPosKin(
            self.parameters,
            {"ul": self.deployment.tolist(), "uphi": self.rotation.tolist()},
            self.sim_parameters,
        )
        tip, sections, tip_direction = result[0], result[3], result[8]
        points = []
        for index, section in enumerate(sections):
            part = np.column_stack([np.asarray(section[axis]) for axis in range(3)])
            if index:
                part = part[1:]
            if len(part):
                points.append(part)
        if not points:
            raise RuntimeError("Forward model returned no backbone.")
        return (
            np.vstack(points) * 1000.0,
            np.asarray(tip[:3]) * 1000.0,
            np.asarray(tip_direction, dtype=float).reshape(3),
        )

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
        self.last_action = "Robot reset"
        print("Robot reset.")

    def remember_undo_state(self) -> None:
        self.undo_stack.append(
            (
                self.deployment.copy(),
                self.rotation.copy(),
                self.selected_tube,
            )
        )
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if not self.undo_stack:
            self.last_action = "Nothing to undo"
            return
        deployment, rotation, selected = self.undo_stack.pop()
        self.deployment[:] = deployment
        self.rotation[:] = rotation
        self.selected_tube = selected
        self.robot_dirty = True
        self.last_action = "Previous adjustment restored"

    def keyboard_move(self, amount_mm: float) -> None:
        self.remember_undo_state()
        self.move(amount_mm)

    def keyboard_rotate(self, amount_deg: float) -> None:
        self.remember_undo_state()
        self.rotate(amount_deg)

    def reset_with_undo(self) -> None:
        self.remember_undo_state()
        self.reset()

    def pose_record(self, label: str) -> dict[str, float | str]:
        if self.robot_dirty:
            self.update_robot()
        mm = self.deployment * 1000.0
        deg = np.rad2deg(self.rotation)
        return {
            "label": label,
            "inner_extension_mm": float(mm[0]),
            "middle_extension_mm": float(mm[1]),
            "outer_extension_mm": float(mm[2]),
            "inner_rotation_deg": float(deg[0]),
            "middle_rotation_deg": float(deg[1]),
            "outer_rotation_deg": float(deg[2]),
            "tip_x_mm": float(self.tip_mm[0]),
            "tip_y_mm": float(self.tip_mm[1]),
            "tip_z_mm": float(self.tip_mm[2]),
            "tip_tilt_deg": float(self.tip_tilt_deg),
            "tip_azimuth_deg": float(self.tip_azimuth_deg),
        }

    def save_waypoint(self) -> None:
        number = len(self.waypoints) + 1
        self.waypoints.append(self.pose_record(f"waypoint_{number}"))
        self.last_action = f"Waypoint {number} saved"
        print(self.last_action)

    def export_snapshot(self) -> None:
        export_dir = Path(__file__).resolve().parent / "exports"
        export_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = export_dir / f"ctr-{stamp}.png"
        csv_path = export_dir / f"ctr-{stamp}.csv"
        current = self.pose_record("current")
        io.write_png(str(image_path), self.canvas.render())
        records = [*self.waypoints, current]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        self.last_action = f"Exported {image_path.name}"
        print(f"Screenshot: {image_path}")
        print(f"Configuration: {csv_path}")

    def toggle_guides(self) -> None:
        self.guides_visible = not self.guides_visible
        self.vertical_reference.visible = self.guides_visible
        self.tip_arrow.visible = self.guides_visible
        self.last_action = "Guides shown" if self.guides_visible else "Guides hidden"
        self.canvas.update()

    def toggle_ui(self) -> None:
        self.ui_visible = not self.ui_visible
        if self.ui_visible:
            self.sidebar.width_max = 360
            self.sidebar.width_min = 360
            self.sidebar.visible = True
            self.last_action = "UI shown"
        else:
            self.sidebar.width_min = 0
            self.sidebar.width_max = 0
            self.sidebar.visible = False
            self.last_action = "UI hidden"
        self.canvas.update()

    def update_touchpad_reset(self, now: float) -> None:
        if self.controller.down(BUTTON_TOUCHPAD):
            if self.touchpad_hold_start is None:
                self.touchpad_hold_start = now
                self.touchpad_reset_fired = False
            elif (
                not self.touchpad_reset_fired
                and now - self.touchpad_hold_start >= TOUCHPAD_RESET_HOLD_S
            ):
                self.reset_with_undo()
                self.touchpad_reset_fired = True
        else:
            self.touchpad_hold_start = None
            self.touchpad_reset_fired = False

    def update_robot(self) -> None:
        started = time.perf_counter()
        backbone, tip, tip_direction = self.calculate_backbone()
        segments = visible_tube_segments(backbone, self.deployment)
        self.last_model_ms = (time.perf_counter() - started) * 1000.0
        for tube, line in self.tube_lines:
            line.set_data(pos=segments[tube])
        for tube, stripe in self.rotation_stripes:
            stripe.set_data(
                pos=tube_surface_stripe(
                    segments[tube],
                    self.rotation[tube],
                    TUBE_STRIPE_OFFSETS_MM[tube],
                )
            )
        self.tip_mm = tip.copy()
        self.tip_direction = tip_direction.copy()
        self.tip_tilt_deg, self.tip_azimuth_deg = orientation_from_vertical(
            tip_direction
        )
        self.tip.set_data(
            np.asarray([tip], dtype=np.float32),
            face_color=(0.9, 0.2, 0.15, 1.0),
            edge_color=(0.45, 0.05, 0.03, 1.0),
            size=TIP_MARKER_SIZE,
        )
        arrow_body, arrow_heads = tip_arrow_geometry(tip, tip_direction)
        self.tip_arrow.set_data(pos=arrow_body, arrows=arrow_heads)
        self.vertical_reference.set_data(
            pos=make_vertical_reference(self.deployment[0] * 1000.0)
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
MODE: {"PRECISION" if self.inputs.precision else "NORMAL"}
WAYPOINTS: {len(self.waypoints)}
LAST: {self.last_action}

              EXTENSION     ROTATION
{markers[0]} INNER      {mm[0]:6.1f} mm     {deg[0]:7.1f} deg
{markers[1]} MIDDLE     {mm[1]:6.1f} mm     {deg[1]:7.1f} deg
{markers[2]} OUTER      {mm[2]:6.1f} mm     {deg[2]:7.1f} deg

TUBE COLOURS
INNER   Blue
MIDDLE  Green
OUTER   Orange
Dark line = rotation stripe

TIP POSITION
X {self.tip_mm[0]:7.1f} mm
Y {self.tip_mm[1]:7.1f} mm
Z {self.tip_mm[2]:7.1f} mm

TIP ORIENTATION
Tilt from +Z  {self.tip_tilt_deg:7.1f} deg
Azimuth       {self.tip_azimuth_deg:7.1f} deg
Direction X   {self.tip_direction[0]:+7.3f}
Direction Y   {self.tip_direction[1]:+7.3f}
Direction Z   {self.tip_direction[2]:+7.3f}

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
D-pad Up/Down Angle increase / decrease
D-pad Left/Right Extension decrease / increase
Right stick   Change view orientation
L2 / R2       Zoom out / in
R3            Hide / show UI
Square (hold) Precision mode
Triangle      Hide / show guides
Circle        Undo adjustment
Cross         Save waypoint
Touchpad hold Reset robot
Options       Export screenshot and data
Left stick    Unused

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
                    self.save_waypoint()
                if self.controller.pressed(BUTTON_CIRCLE):
                    self.undo()
                if self.controller.pressed(BUTTON_TRIANGLE):
                    self.toggle_guides()
                if self.controller.pressed(BUTTON_R3):
                    self.toggle_ui()
                if self.controller.pressed(BUTTON_OPTIONS):
                    self.export_snapshot()
                self.update_touchpad_reset(now)
                self.inputs = self.controller.read_inputs()
                dpad = (self.inputs.dpad_x, self.inputs.dpad_y)
                if dpad != (0, 0) and dpad != self.previous_dpad:
                    self.remember_undo_state()
                self.previous_dpad = dpad
                rotation_speed = (
                    PRECISION_ROTATION_SPEED_DEG_S
                    if self.inputs.precision
                    else DPAD_ROTATION_SPEED_DEG_S
                )
                translation_speed = (
                    PRECISION_TRANSLATION_SPEED_MM_S
                    if self.inputs.precision
                    else DPAD_TRANSLATION_SPEED_MM_S
                )
                if self.inputs.dpad_y:
                    self.rotate(self.inputs.dpad_y * rotation_speed * dt)
                if self.inputs.dpad_x:
                    self.move(self.inputs.dpad_x * translation_speed * dt)
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
                zoom_command = self.inputs.r2 - self.inputs.l2
                if abs(zoom_command) > TRIGGER_DEADZONE:
                    camera = self.view.camera
                    camera.distance = float(
                        np.clip(
                            camera.distance
                            * math.exp(-zoom_command * CAMERA_ZOOM_RATE_S * dt),
                            60.0,
                            2000.0,
                        )
                    )
                    self.canvas.update()
            else:
                self.inputs = InputSnapshot()
                self.previous_dpad = (0, 0)

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
            "w": lambda: self.keyboard_move(KEYBOARD_TRANSLATION_STEP_MM),
            "s": lambda: self.keyboard_move(-KEYBOARD_TRANSLATION_STEP_MM),
            "a": lambda: self.keyboard_rotate(-KEYBOARD_ROTATION_STEP_DEG),
            "d": lambda: self.keyboard_rotate(KEYBOARD_ROTATION_STEP_DEG),
            "r": self.reset_with_undo,
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
            "right stick orbit | L2/R2 zoom | R3 UI | Touchpad hold reset\n"
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
