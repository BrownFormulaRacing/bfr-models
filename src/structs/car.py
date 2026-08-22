from dataclasses import dataclass, field

from .aero import Aero
from .engine import Engine
from .suspension import Suspension
from .direction import Direction

from ..utils.constants import AIR_DENSITY, G, TIRE_RADIUS
from collections import defaultdict

import math

@dataclass
class Car:
    sus_params: Suspension = field(default_factory=Suspension)
    aero_params: Aero = field(default_factory=Aero)
    engine_params: Engine = field(default_factory=Engine)

    # tire_params: Tire = field(default_factory=Tire)
    # tire_load_params: dict[str, float] = field(init=False)

    # def __post_init__(self) -> None:
    #     front_axle_load = self.sus_params.mass * self.sus_params.weight_dist_front
    #     rear_axle_load = self.sus_params.mass * (1 - self.sus_params.weight_dist_front)

    #     self.tire_load_params = {
    #         "FO": front_axle_load / 2,
    #         "FI": front_axle_load / 2,
    #         "RO": rear_axle_load / 2,
    #         "RI": rear_axle_load / 2 
    #     }

    #     tire_coeffs_path = input("Enter the path to the tire coefficients")
    #     self.tire_params.load_coefficients(tire_coeffs_path)

    # def load_transfer(self, lat_accel: float, long_accel: float, vel: float) -> dict[str, float]:
    #     """Placeholder for wheel-load calculation."""
    #     # Step 1: Calculate downforce
    #     total_downforce = (0.5) * AIR_DENSITY * math.pow(vel, 2)
    #     single_tire_downforce = total_downforce / 4

    #     # Step 2: Calculate loads after cornering/accelerating
    #     lat_load_transfer = (self.sus_params.mass * lat_accel * G) * (self.sus_params.cg_height) / self.sus_params.trackwidth
    #     long_load_transfer = (self.sus_params.mass * long_accel * G) * (self.sus_params.cg_height) / self.sus_params.wheelbase

    #     return {
    #         "FO" : self.tire_load_params["FO"] + lat_load_transfer + single_tire_downforce,
    #         "FI" : self.tire_load_params["FI"] - lat_load_transfer + single_tire_downforce,
    #         "RO" : self.tire_load_params["RO"] + long_load_transfer + single_tire_downforce,
    #         "RI" : self.tire_load_params["RI"] - long_load_transfer + single_tire_downforce
    #     }
    

    # def cornering_force(self, lat_accel_guess: float, corner_radius: float, front_slip_angle: float, rear_slip_angle: float) -> tuple[float, float]:
    #     """Ported from MATLAB `corneringForce.m`.

    #     Args:
    #         lat_accel_guess: lateral acceleration guess, g's (Ayg)
    #         corner_radius: corner radius, m (R)
    #         front_slip_angle: front axle slip/steer angle, rad (af)
    #         rear_slip_angle: rear axle slip/steer angle, rad (ar)

    #     Returns:
    #         (lat_accel, yaw_moment): lateral acceleration [m/s^2], yaw moment [N*m]
    #     """
    #     # Determined from UCM
    #     velocity = math.sqrt(abs(lat_accel_guess) * corner_radius)  # m/s
    #     # Pulls tire loads under steady-state cornering
    #     tire_loads = self.load_transfer(lat_accel_guess, 0.0, velocity)

    #     # NOTE: Use these values (for slip anlges) to compare against telemetry data
    #     front_roll_grad = self.sus_params.front_roll_stiffness
    #     rear_roll_grad = (1 - self.sus_params.lateral_load_transfer_dist) / (self.sus_params.lateral_load_transfer_dist * front_roll_grad)

    #     # Delta, beta (computed for parity with MATLAB; unused in the return,
    #     # same as the original -- looks like they were meant for an outer solver)
    #     cg_rear_distance = self.sus_params.wheelbase * self.sus_params.weight_dist_front       # cg -> rear axle distance
    #     cg_front_distance = self.sus_params.wheelbase - cg_rear_distance         # cg -> front axle distance

    #     # Equal to a = V^2 / R, but substituting V/R with yaw velocity
    #     yaw_velocity = abs(lat_accel_guess) / velocity
    #     vehicle_slip_angle = rear_slip_angle + (cg_rear_distance * yaw_velocity / velocity)
    #     steering_angle = vehicle_slip_angle + (cg_front_distance * yaw_velocity / velocity) - front_slip_angle  # noqa: F841

    #     # Roll angle from lateral load transfer (rigid chassis in roll)
    #     roll_angle = lat_accel_guess * self.sus_params.mass * self.sus_params.cg_height / (front_roll_grad + rear_roll_grad)

    #     # Camber from roll -- outer tires only, inner stays near static
    #     # TODO: Integrate static camber and camber gain curves here
    #     camber_angles = {
    #         "FO": -self.sus_params.camber_gain * roll_angle,
    #         "FI": 0.0,
    #         "RO": -self.sus_params.camber_gain * roll_angle,
    #         "RI": 0.0,
    #     }

    #     # Slip angles: track-width offset from the yaw center
    #     slip_angle_offset = self.sus_params.trackwidth / (2 * corner_radius)
    #     slip_angles = {
    #         "FO": front_slip_angle - slip_angle_offset,
    #         "FI": front_slip_angle + slip_angle_offset,
    #         "RO": rear_slip_angle - slip_angle_offset,
    #         "RI": rear_slip_angle + slip_angle_offset,
    #     }
    #     # NOTE: Can use this (above) for telemetry

    #     generated_grip = {
    #         "FO" : 0,
    #         "FI" : 0,
    #         "RO" : 0,
    #         "RI" : 0
    #     }
    #     for tire in generated_grip:
    #         # TODO: Change this to reference some function that consults the tire model, stored in tire.py
    #         generated_grip[tire], _ = self.tire_model(tire_loads[tire], camber_angles[tire], slip_angles[tire])

    #     lat_accel_updated = sum(generated_grip.values()) / self.sus_params.mass
    #     yaw_moment = (cg_front_distance * (generated_grip["FO"] + generated_grip["FI"])) - (cg_rear_distance * (generated_grip["RO"] + generated_grip["RI"]))  # OS > 0, US < 0

    #     return lat_accel_updated, yaw_moment


    # def longitudinal_force(self, *args, **kwargs) -> tuple[float, float]:
    #     return 0.0, 0.0

    # def longitudinal_force_traction_limited(self, long_accel_guess: float, velocity: float, direction: Direction) -> float:
    #     """Same as longitudinal_force, but assumes unlimited engine power --
    #     purely traction-limited. Useful for isolating suspension behavior
    #     without needing a torque curve / gear table.
    #     """
    #     tire_loads = self.load_transfer(0.0, long_accel_guess, velocity)

    #     squat_degree = self.sus_params.pitch_stiffness * long_accel_guess

    #     # TODO: Use the camber angle gain plot to identify camber gain
    #     front_camber_angle = squat_degree * 0
    #     rear_camber_angle = squat_degree * 0
    #     universal_slip_angle = 0.0

    #     generated_grip = {
    #         "FO" : 0,
    #         "FI" : 0,
    #         "RO" : 0,
    #         "RI" : 0
    #     }
    #     for tire in generated_grip:
    #         if tire.startswith("F"):
    #         # TODO: Change this to reference some function that consults the tire model, stored in tire.py
    #             current_camber_angle = front_camber_angle
    #         else:
    #             current_camber_angle = rear_camber_angle
    #         _, generated_grip[tire] = self.tire_model(tire_loads[tire], current_camber_angle, universal_slip_angle)

        
    #     if direction == Direction.BRAKING:  # braking -- both axles can contribute
    #         brake_bias = self.engine_params.brake_bias
    #         front_grip_limit = generated_grip["FO"] + generated_grip["FI"]
    #         rear_grip_limit = generated_grip["RO"] + generated_grip["RI"]
    #         total_grip = min(front_grip_limit / brake_bias, rear_grip_limit / (1 - brake_bias))
    #         long_accel_updated = -total_grip / self.sus_params.mass
    #     else:  # accel -- RWD traction limit only, no engine cap
    #         total_grip = generated_grip["RO"] + generated_grip["RI"]
    #         long_accel_updated = total_grip / self.sus_params.mass

    #     return long_accel_updated
