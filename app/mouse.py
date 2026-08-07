"""Reusable mouse-coordinate conversion and movement utilities.

The conversion functions deliberately do not depend on a mouse-control library.  A caller can
therefore replace the backend without changing the FOV or sensitivity mathematics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol


class MouseBackend(Protocol):
    """Minimal interface required by :func:`move_mouse`."""

    def move_relative(self, dx: float, dy: float) -> None:
        """Move the system pointer by a relative delta in the backend's coordinate system."""


@dataclass(frozen=True, slots=True)
class MouseStep:
    """One relative movement and its following delay."""

    dx: float
    dy: float
    delay: float


class PynputMouseBackend:
    """System mouse backend implemented with ``pynput``.

    Importing this class is safe without pynput installed; constructing it requires the package.
    """

    def __init__(self) -> None:
        try:
            from pynput import mouse
        except ImportError as exc:
            raise RuntimeError("pynput is required for the default mouse backend.") from exc
        self._controller: Any = mouse.Controller()

    def move_relative(self, dx: float, dy: float) -> None:
        """Move the operating-system pointer by ``dx`` and ``dy``."""
        self._controller.move(dx, dy)


def calculate_mouse_delta(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    fov: float,
    sensitivity: float,
    reference_fov: float = 90.0,
) -> tuple[float, float]:
    """Convert two screen coordinates into a sensitivity- and FOV-adjusted mouse delta.

    Parameters:
        point_a: Source ``(x, y)`` coordinate in screen pixels.
        point_b: Destination ``(x, y)`` coordinate in screen pixels.
        fov: Current horizontal field of view, in degrees. The name ``fov`` is used because
            "point of view" is not a conventional measurement for this value.
        sensitivity: Positive game or mouse sensitivity. A higher sensitivity needs less
            physical mouse movement for the same intended angular movement.
        reference_fov: Calibration FOV for the input coordinate system, in degrees.

    Returns:
        The adjusted relative ``(dx, dy)``. With ``fov == reference_fov`` and sensitivity 1,
        it equals ``point_b - point_a``.

    Raises:
        TypeError: If a coordinate or numeric argument has an invalid type.
        ValueError: If a value is non-finite, FOV is outside ``(0, 180)``, or sensitivity is
            not positive.

    Calculation:
        ``raw_delta * tan(reference_fov / 2) / tan(fov / 2) / sensitivity``.
        This is a normalized perspective conversion layer, not a game-specific sensitivity
        formula. Replace the ``fov_scale`` calculation below when calibrating for a game.
    """
    ax, ay = _validate_point(point_a, "point_a")
    bx, by = _validate_point(point_b, "point_b")
    validated_fov = _validate_fov(fov, "fov")
    validated_reference_fov = _validate_fov(reference_fov, "reference_fov")
    validated_sensitivity = _validate_positive(sensitivity, "sensitivity")

    # Replace this one conversion layer with a game-specific yaw/pitch or cm/360 formula.
    fov_scale = math.tan(math.radians(validated_reference_fov / 2)) / math.tan(
        math.radians(validated_fov / 2)
    )
    return (
        (bx - ax) * fov_scale / validated_sensitivity,
        (by - ay) * fov_scale / validated_sensitivity,
    )


def generate_mouse_path(
    dx: float,
    dy: float,
    duration: float = 0.2,
    steps: int = 20,
    smooth: bool = True,
) -> list[MouseStep]:
    """Generate a relative mouse path whose deltas sum to the requested movement.

    Parameters:
        dx: Total horizontal mouse delta.
        dy: Total vertical mouse delta.
        duration: Approximate total duration in seconds; it must be non-negative.
        steps: Number of relative movements; it must be a positive integer.
        smooth: Use cubic ease-in-out when true, otherwise use linear interpolation.

    Returns:
        ``steps`` relative :class:`MouseStep` values. Their ``dx`` and ``dy`` values sum to the
        supplied totals (using :func:`math.fsum`) without accumulated interpolation drift.

    Raises:
        TypeError: If ``steps`` or ``smooth`` has an invalid type.
        ValueError: If a numeric value is non-finite, duration is negative, or steps is invalid.

    Calculation:
        The path is derived from cumulative positions then converted to relative deltas. The
        final delta is calculated from the remaining total, preserving the requested endpoint.
    """
    total_dx = _validate_finite(dx, "dx")
    total_dy = _validate_finite(dy, "dy")
    validated_duration = _validate_finite(duration, "duration")
    if validated_duration < 0:
        raise ValueError("duration must be greater than or equal to 0.")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer.")
    if steps <= 0:
        raise ValueError("steps must be greater than 0.")
    if not isinstance(smooth, bool):
        raise TypeError("smooth must be a bool.")

    path: list[MouseStep] = []
    dx_values: list[float] = []
    dy_values: list[float] = []
    delay = validated_duration / steps
    for index in range(1, steps + 1):
        progress = index / steps
        eased = _ease_in_out(progress) if smooth else progress
        next_x = total_dx if index == steps else total_dx * eased
        next_y = total_dy if index == steps else total_dy * eased
        step_dx = next_x - math.fsum(dx_values)
        step_dy = next_y - math.fsum(dy_values)
        dx_values.append(step_dx)
        dy_values.append(step_dy)
        path.append(MouseStep(step_dx, step_dy, delay))
    return path


def move_mouse(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    fov: float,
    sensitivity: float,
    duration: float = 0.2,
    steps: int = 20,
    smooth: bool = True,
    reference_fov: float = 90.0,
    backend: MouseBackend | None = None,
) -> tuple[float, float]:
    """Calculate and execute a relative mouse movement from ``point_a`` towards ``point_b``.

    Parameters:
        point_a: Source ``(x, y)`` coordinate.
        point_b: Destination ``(x, y)`` coordinate.
        fov: Current horizontal FOV in degrees.
        sensitivity: Positive sensitivity used by :func:`calculate_mouse_delta`.
        duration: Approximate movement duration in seconds.
        steps: Number of relative mouse movements.
        smooth: Choose cubic ease-in-out or linear interpolation.
        reference_fov: Calibration FOV used by the normalized conversion layer.
        backend: Optional relative-movement backend. Defaults to :class:`PynputMouseBackend`.

    Returns:
        The final calculated ``(dx, dy)`` passed cumulatively to the backend.

    Raises:
        TypeError: If an argument type is invalid.
        ValueError: If a coordinate or movement setting is invalid.
        RuntimeError: If the default pynput backend is unavailable.

    Calculation:
        This function delegates coordinate conversion to :func:`calculate_mouse_delta` and path
        generation to :func:`generate_mouse_path`; it contains no FOV or sensitivity formula.
    """
    dx, dy = calculate_mouse_delta(point_a, point_b, fov, sensitivity, reference_fov)
    path = generate_mouse_path(dx, dy, duration, steps, smooth)
    if dx == 0 and dy == 0:
        return dx, dy
    selected_backend = backend or PynputMouseBackend()
    for step in path:
        selected_backend.move_relative(step.dx, step.dy)
        if step.delay:
            time.sleep(step.delay)
    return dx, dy


def _ease_in_out(progress: float) -> float:
    """Return a cubic ease-in-out value for a normalized progress value."""
    return progress * progress * (3 - 2 * progress)


def _validate_point(point: object, name: str) -> tuple[float, float]:
    if not isinstance(point, tuple) or len(point) != 2:
        raise TypeError(f"{name} must be a tuple containing exactly two finite numbers.")
    return _validate_finite(point[0], f"{name}[0]"), _validate_finite(point[1], f"{name}[1]")


def _validate_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _validate_positive(value: object, name: str) -> float:
    result = _validate_finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return result


def _validate_fov(value: object, name: str) -> float:
    result = _validate_positive(value, name)
    if result >= 180:
        raise ValueError(f"{name} must be less than 180 degrees.")
    return result
