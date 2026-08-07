from __future__ import annotations

import math

import pytest

from app.mouse import calculate_mouse_delta, generate_mouse_path, move_mouse


class FakeMouseBackend:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []

    def move_relative(self, dx: float, dy: float) -> None:
        self.moves.append((dx, dy))


def test_equal_points_return_zero_without_backend_movement() -> None:
    backend = FakeMouseBackend()
    assert move_mouse((1, 2), (1, 2), 90, 1, backend=backend) == (0.0, 0.0)
    assert backend.moves == []


def test_equal_points_still_validate_movement_settings() -> None:
    with pytest.raises(ValueError):
        move_mouse((1, 2), (1, 2), 90, 1, steps=0)


@pytest.mark.parametrize(
    ("point_a", "point_b", "expected"),
    [
        ((0, 0), (10, 0), (10.0, 0.0)),
        ((10, 0), (0, 0), (-10.0, 0.0)),
        ((0, 10), (0, 0), (0.0, -10.0)),
        ((0, 0), (0, 10), (0.0, 10.0)),
        ((0, 0), (3, 4), (3.0, 4.0)),
    ],
)
def test_directional_deltas(
    point_a: tuple[float, float], point_b: tuple[float, float], expected: tuple[float, float]
) -> None:
    assert calculate_mouse_delta(point_a, point_b, 90, 1) == expected


def test_higher_sensitivity_reduces_required_mouse_delta() -> None:
    assert calculate_mouse_delta((0, 0), (20, 0), 90, 2) == (10.0, 0.0)


def test_fov_changes_normalized_delta() -> None:
    narrow_fov = calculate_mouse_delta((0, 0), (10, 0), 60, 1)[0]
    wide_fov = calculate_mouse_delta((0, 0), (10, 0), 120, 1)[0]
    assert narrow_fov > 10
    assert wide_fov < 10


@pytest.mark.parametrize("sensitivity", [0, -1, float("nan"), float("inf")])
def test_invalid_sensitivity_raises_value_error(sensitivity: float) -> None:
    with pytest.raises(ValueError):
        calculate_mouse_delta((0, 0), (1, 1), 90, sensitivity)


@pytest.mark.parametrize("steps", [0, -1])
def test_invalid_steps_raise_value_error(steps: int) -> None:
    with pytest.raises(ValueError):
        generate_mouse_path(1, 1, steps=steps)


def test_smooth_path_differs_from_linear_path() -> None:
    smooth = generate_mouse_path(10, 0, duration=0, steps=4, smooth=True)
    linear = generate_mouse_path(10, 0, duration=0, steps=4, smooth=False)
    assert smooth[0].dx != linear[0].dx
    assert linear[0].dx == 2.5


def test_path_preserves_total_without_rounding_drift() -> None:
    path = generate_mouse_path(1 / 3, -2 / 3, duration=0, steps=17, smooth=True)
    assert math.fsum(step.dx for step in path) == 1 / 3
    assert math.fsum(step.dy for step in path) == -2 / 3


@pytest.mark.parametrize(
    ("point_a", "point_b", "fov", "duration"),
    [((float("nan"), 0), (1, 1), 90, 0), ((0, 0), (1, 1), 0, 0), ((0, 0), (1, 1), 90, -1)],
)
def test_invalid_coordinates_fov_and_duration_raise_value_error(
    point_a: tuple[float, float], point_b: tuple[float, float], fov: float, duration: float
) -> None:
    if duration < 0:
        with pytest.raises(ValueError):
            generate_mouse_path(1, 1, duration=duration)
    else:
        with pytest.raises(ValueError):
            calculate_mouse_delta(point_a, point_b, fov, 1)


def test_move_mouse_uses_injected_backend() -> None:
    backend = FakeMouseBackend()
    result = move_mouse((0, 0), (8, 4), 90, 1, duration=0, steps=4, backend=backend)
    assert result == (8.0, 4.0)
    assert len(backend.moves) == 4
    assert math.fsum(dx for dx, _ in backend.moves) == 8.0
    assert math.fsum(dy for _, dy in backend.moves) == 4.0
