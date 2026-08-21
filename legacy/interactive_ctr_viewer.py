"""Responsive CTR viewer with PS5 and keyboard control."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pygame
import pyvista as pv

from CTR_superPosKin_fun_sectioned import superPosKin
from tube_parameters import build_supervisor_ctr_parameters

# Mapping confirmed by the controller-identification scripts on this Mac.
BUTTON_CROSS, BUTTON_L1, BUTTON_R1 = 0, 9, 10
DPAD_UP_BUTTON, DPAD_DOWN_BUTTON = 11, 12
DPAD_LEFT_BUTTON, DPAD_RIGHT_BUTTON = 13, 14

DPAD_ROTATION_SPEED_DEG_S = 90.0
DPAD_TRANSLATION_SPEED_MM_S = 30.0
KEYBOARD_TRANSLATION_STEP_MM = 1.0
KEYBOARD_ROTATION_STEP_DEG = 5.0

TIMER_INTERVAL_MS = 16
MODEL_UPDATE_HZ = 40.0
STATUS_UPDATE_HZ = 2.0
MODEL_POINTS_PER_SECTION = 8
DISPLAY_POINTS = 48
CONTROLLER_RETRY_S = 1.0
ERROR_PRINT_INTERVAL_S = 2.0

TUBE_NAMES = ("INNER", "MIDDLE", "OUTER")
INITIAL_DEPLOYMENT_M = np.array([120e-3, 70e-3, 40e-3])
INITIAL_ROTATION_RAD = np.zeros(3)


def resample_polyline(points: np.ndarray, count: int = DISPLAY_POINTS) -> np.ndarray:
    """Give the centreline fixed topology so its actor can be reused."""
    points = np.asarray(points, dtype=float)
    if count < 2:
        raise ValueError("count must be at least 2")
    if len(points) == 0:
        return np.zeros((count, 3))
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    if distance[-1] <= 1e-12:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, distance[-1], count)
    return np.column_stack(
        [np.interp(targets, distance, points[:, axis]) for axis in range(3)]
    )


@dataclass
class InputSnapshot:
    dpad_x: int = 0
    dpad_y: int = 0


class ControllerManager:
    """Own one joystick without repeatedly restarting Pygame."""

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
        joystick.init()
        required_buttons = max(BUTTON_CROSS, BUTTON_L1, BUTTON_R1) + 1
        buttons = joystick.get_numbuttons()
        hats = joystick.get_numhats()
        has_hat = hats > 0
        has_dpad_buttons = buttons > DPAD_RIGHT_BUTTON
        if buttons < required_buttons or not (has_hat or has_dpad_buttons):
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
        """Drain Pygame's bounded queue and handle hot-plugging."""
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEREMOVED and self.connected:
                current_id = self.joystick.get_instance_id()
                if getattr(event, "instance_id", current_id) == current_id:
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

    def read_dpad(self) -> InputSnapshot:
        """Read either SDL hat input or the macOS PS5 D-pad button mapping."""
        if not self.connected:
            return InputSnapshot()
        if self.joystick.get_numhats() > 0:
            x, y = self.joystick.get_hat(0)
            if x or y:
                return InputSnapshot(int(x), int(y))
        if self.joystick.get_numbuttons() > DPAD_RIGHT_BUTTON:
            x = int(self.joystick.get_button(DPAD_RIGHT_BUTTON)) - int(
                self.joystick.get_button(DPAD_LEFT_BUTTON)
            )
            y = int(self.joystick.get_button(DPAD_UP_BUTTON)) - int(
                self.joystick.get_button(DPAD_DOWN_BUTTON)
            )
            return InputSnapshot(x, y)
        return InputSnapshot()


class CTRViewer:
    def __init__(self) -> None:
        pygame.init()
        pygame.joystick.init()
        self.parameters = build_supervisor_ctr_parameters()
        self.sim_parameters = {"n_p": MODEL_POINTS_PER_SECTION, "isPlot": False}
        self.total_lengths = np.array(
            [sum(lengths) for lengths in self.parameters["l_t"]]
        )
        self.deployment = INITIAL_DEPLOYMENT_M.copy()
        self.rotation = INITIAL_ROTATION_RAD.copy()
        self.selected_tube = 0
        self.robot_dirty = True
        self.inputs = InputSnapshot()
        self.last_model_ms = 0.0
        self.model_rate = 0.0
        self.ui_rate = 0.0
        self.model_count = 0
        self.ui_count = 0
        self.rate_start = time.perf_counter()
        self.error_text = ""
        self.last_error_print = -ERROR_PRINT_INTERVAL_S
        self.controller = ControllerManager()

        # Multisampling and tube-style line rendering are unnecessary for this
        # centreline and can be disproportionately costly on macOS/Metal.
        self.plotter = pv.Plotter(window_size=(1200, 800))
        self.plotter.render_window.SetMultiSamples(0)
        self.plotter.set_background("white")
        self.plotter.show_grid()
        self.plotter.add_axes()
        backbone, self.tip_mm = self.calculate_backbone()
        self.backbone_poly = pv.lines_from_points(resample_polyline(backbone))
        self.plotter.add_mesh(
            self.backbone_poly,
            line_width=6.0,
            render_lines_as_tubes=False,
            smooth_shading=False,
            reset_camera=False,
            name="ctr_backbone",
            render=False,
        )
        self.tip_actor = self.plotter.add_mesh(
            pv.Sphere(radius=2.5, theta_resolution=12, phi_resolution=12),
            name="ctr_tip",
            reset_camera=False,
            render=False,
        )
        self.tip_actor.position = tuple(self.tip_mm)
        self.plotter.add_mesh(
            pv.Sphere(radius=2.5, theta_resolution=12, phi_resolution=12),
            name="ctr_origin",
            reset_camera=False,
            render=False,
        )
        self.status_actor = self.plotter.add_text(
            self.status_text(),
            position="upper_left",
            font_size=9,
            name="status",
            render=False,
        )
        now = time.perf_counter()
        self.last_tick = self.last_model_update = self.last_status_update = now
        self.configure_keyboard()
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()
        self.plotter.camera.focal_point = (0.0, 0.0, 0.0)

    def calculate_backbone(self) -> tuple[np.ndarray, np.ndarray]:
        result = superPosKin(
            self.parameters,
            {"ul": self.deployment.tolist(), "uphi": self.rotation.tolist()},
            self.sim_parameters,
        )
        tip, sections = result[0], result[3]
        points = []
        for index, section in enumerate(sections):
            part = np.column_stack([np.asarray(section[a]) for a in range(3)])
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
        self.selected_tube = index % 3
        self.update_status()

    def reset(self) -> None:
        self.deployment[:] = INITIAL_DEPLOYMENT_M
        self.rotation[:] = INITIAL_ROTATION_RAD
        self.selected_tube = 0
        self.robot_dirty = True
        print("Robot reset.")

    def update_robot(self) -> None:
        start = time.perf_counter()
        backbone, tip = self.calculate_backbone()
        self.last_model_ms = (time.perf_counter() - start) * 1000.0
        self.backbone_poly.points[:] = resample_polyline(backbone)
        self.backbone_poly.Modified()
        self.tip_mm = tip.copy()
        self.tip_actor.position = tuple(tip)
        self.robot_dirty = False
        self.model_count += 1

    def update_rates(self, now: float) -> None:
        """Update independent UI and changed-configuration rates once a second."""
        self.ui_count += 1
        elapsed = now - self.rate_start
        if elapsed >= 1.0:
            self.ui_rate = self.ui_count / elapsed
            self.model_rate = self.model_count / elapsed
            self.ui_count = 0
            self.model_count = 0
            self.rate_start = now

    def status_text(self) -> str:
        mm, deg = self.deployment * 1000, np.rad2deg(self.rotation)
        state = "CONNECTED" if self.controller.connected else "DISCONNECTED"
        text = f"""ACTIVE TUBE: {TUBE_NAMES[self.selected_tube]}
PS5: {state}

Inner   | {mm[0]:6.1f} mm   {deg[0]:7.1f} deg
Middle  | {mm[1]:6.1f} mm   {deg[1]:7.1f} deg
Outer   | {mm[2]:6.1f} mm   {deg[2]:7.1f} deg

TIP XYZ
X = {self.tip_mm[0]:7.1f} mm
Y = {self.tip_mm[1]:7.1f} mm
Z = {self.tip_mm[2]:7.1f} mm

LIVE INPUT
D-pad X/Y       = {self.inputs.dpad_x:+d}, {self.inputs.dpad_y:+d}

PERFORMANCE
Model calc      = {self.last_model_ms:5.1f} ms
UI loop         = {self.ui_rate:4.1f} FPS
Robot updates   = {self.model_rate:4.1f} /s

PS5 CONTROLS
L1 / R1         previous / next tube
D-pad Up / Down increase / decrease angle
D-pad Left/Right retract / insert selected tube
Left stick      unused (use mouse for view)
Right stick     unused
L2 / R2         unused
Cross           reset robot

KEYBOARD BACKUP
1 / 2 / 3       Inner / Middle / Outer
W / S           insert / retract 1 mm
A / D           rotate -/+ 5 deg
R / Q           reset / quit
Mouse drag      change view smoothly
Mouse wheel     zoom"""
        return text + (f"\n\nERROR: {self.error_text}" if self.error_text else "")

    def update_status(self) -> None:
        self.status_actor.set_text("upper_left", self.status_text())

    def configure_keyboard(self) -> None:
        callbacks = {
            "1": lambda: self.select(0), "2": lambda: self.select(1),
            "3": lambda: self.select(2),
            "w": lambda: self.move(KEYBOARD_TRANSLATION_STEP_MM),
            "s": lambda: self.move(-KEYBOARD_TRANSLATION_STEP_MM),
            "a": lambda: self.rotate(-KEYBOARD_ROTATION_STEP_DEG),
            "d": lambda: self.rotate(KEYBOARD_ROTATION_STEP_DEG),
            "r": self.reset,
        }
        for key, callback in callbacks.items():
            try:
                self.plotter.iren.clear_events_for_key(key)
            except Exception:
                pass
            self.plotter.add_key_event(key, callback)

    def record_error(self, exc: Exception, now: float) -> None:
        self.error_text = f"{type(exc).__name__}: {exc}"
        if now - self.last_error_print >= ERROR_PRINT_INTERVAL_S:
            print(f"Controller update error: {self.error_text}")
            self.last_error_print = now

    def tick(self, _step: int = 0) -> None:
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
                self.inputs = self.controller.read_dpad()
                if self.inputs.dpad_y:
                    self.rotate(
                        self.inputs.dpad_y * DPAD_ROTATION_SPEED_DEG_S * dt
                    )
                if self.inputs.dpad_x:
                    self.move(
                        self.inputs.dpad_x * DPAD_TRANSLATION_SPEED_MM_S * dt
                    )
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

    def run(self) -> None:
        print("\nPS5 CTR SIMULATOR\nL1/R1 select | D-pad Up/Down rotate | "
              "D-pad Left/Right move | mouse changes view | Cross reset\n"
              "Keyboard backup: 1/2/3, W/S, A/D, R, Q\n")
        try:
            # PyVista/VTK timer events can be heavily coalesced on macOS.  A
            # continuously serviced interactive loop gives controller input
            # the same event/render cadence as native mouse interaction.
            self.plotter.show(
                title="PS5 CTR Simulator",
                interactive_update=True,
                auto_close=False,
            )
            # PyVista 0.48 exposes only a getter on its wrapper.  Set the
            # desired rate through the underlying VTK interactor instead.
            self.plotter.iren.interactor.SetDesiredUpdateRate(
                1000.0 / TIMER_INTERVAL_MS
            )
            frame_interval = TIMER_INTERVAL_MS / 1000.0
            while self.plotter.iren is not None:
                frame_start = time.perf_counter()
                self.tick()
                self.plotter.update(stime=1, force_redraw=True)
                remaining = frame_interval - (time.perf_counter() - frame_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        finally:
            self.controller.disconnect()
            pygame.quit()
            self.plotter.close()
            print("CTR simulator closed.")


def main() -> None:
    CTRViewer().run()


if __name__ == "__main__":
    main()
