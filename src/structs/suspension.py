from dataclasses import dataclass


@dataclass
class Suspension:
    mass: float = 288.0
    cg_height: float = 0.300
    trackwidth: float = 1.245
    wheelbase: float = 1.53
    weight_dist_front: float = 0.5
    lateral_load_transfer_dist: float = 0.5
    front_roll_rate: float = 40000.0
    rear_roll_rate: float = 40000.0
    pitch_rate: float = 0.0
