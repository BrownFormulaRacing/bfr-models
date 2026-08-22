from dataclasses import dataclass, field

from structs.car import Car
from structs.direction import Direction

MAX_ITERATIONS = 50
CONVERGENCE_TOL = 0.01
GUESS_SENSITIVITY = 1.0

@dataclass
class GGV:
    lat_accels: dict[float, float] = field(default_factory=dict)
    long_accels: dict[float, float] = field(default_factory=dict)
    brake_accels: dict[float, float] = field(default_factory=dict)

    def populate_lat_accels(
        self,
        car: Car,
        corner_radii: list[float] = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0],
    ) -> None:
        """Ported from MATLAB `makeGGVc.m`'s lateral-grip loop + `findAy.m`.

        For each corner radius, finds the max lateral acceleration achievable
        with zero net yaw moment (front/rear slip angles balanced), by:
          1. outer loop: adjust front/rear slip angles (af/ar) until Mz ~ 0
          2. inner loop: converge the lateral-accel guess against
             car.cornering_force's output (fixed-point iteration)
        """
        results: dict[float, float] = {}
        for corner_radius in corner_radii:
            lat_accel, _ = self._find_balanced_lateral_accel(car, corner_radius)
            results[corner_radius] = lat_accel
        self.lat_accels = results

    def populate_long_accels(
        self,
        car: Car,
        velocities: list[float] = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0],
    ) -> None:
        """Ported from MATLAB `makeGGVc.m`'s accel loop.

        NOTE: uses `longitudinal_force_traction_limited` (no torque curve),
        matching the suspension-only testing approach -- so unlike MATLAB's
        `longForce`, there's no P_eng to record here.
        """
        results: dict[float, float] = {}
        for velocity in velocities:
            results[velocity] = self._converge_long_accel(
                car, velocity, Direction.ACCEL, initial_guess=1.0
            )
        self.long_accels = results

    def _find_balanced_lateral_accel(car, corner_radius):
        # TODO: Finish this

    def populate_brake_accels(
        self,
        car: Car,
        velocities: list[float] = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0],
    ) -> None:
        """Ported from MATLAB `makeGGVc.m`'s decel loop."""
        results: dict[float, float] = {}
        for velocity in velocities:
            results[velocity] = self._converge_long_accel(
                car, velocity, Direction.BRAKING, initial_guess=-1.0
            )
        self.brake_accels = results

    def _converge_long_accel(
        self, car: Car, velocity: float, direction: Direction, initial_guess: float
    ) -> float:
        """Shared fixed-point convergence for accel/brake, matching MATLAB's
        identical accel-loop and decel-loop bodies (just different `dir`
        and starting guess sign).
        """
        long_accel_guess = initial_guess
        long_accel = long_accel_guess
        for _ in range(MAX_ITERATIONS):
            long_accel = car.longitudinal_force_traction_limited(
                long_accel_guess, velocity, direction
            )
            diff = long_accel - long_accel_guess
            if abs(diff) < CONVERGENCE_TOL:
                return long_accel
            long_accel_guess += GUESS_SENSITIVITY * diff

        return long_accel