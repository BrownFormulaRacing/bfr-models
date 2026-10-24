from dataclasses import dataclass


@dataclass
class Engine:
    torque_curve: list[tuple[float, float]] | None = None
    primary_gear: float = 19.0 / 9.0
    final_gear: float = 40.0 / 11.0
    gears: tuple[float, ...] = (2.66666667, 1.9375, 1.61111111, 1.40909091, 1.26086957, 1.16666667)
    redline: float = 13500.0
    max_vel: float = 0.0
