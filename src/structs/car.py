from dataclasses import dataclass, field

from .aero import Aero
from .engine import Engine
from .suspension import Suspension
from .direction import Direction
from ..utils.constants import AIR_DENSITY, G
from collections import defaultdict
import json

class Car:

    def __init__(self):
        self.sus_params = Suspension()
        self.aero_params = Aero()
        self.engine_params = Engine()

    def setup(self, sus_params=None, aero_params=None, engine_params=None):
        if sus_params:
            self.sus_params.set_params(sus_params)

        if aero_params:
            self.aero_params.set_params(aero_params)


    def calculate_corner_loads(self, Ay):
        """Static load + simplistic lateral load transfer split via LLTD."""

        Fz_static_front = self.sus_params.mass * G * self.sus_params.weight_dist_front / 2
        Fz_static_rear = self.sus_params.mass * G * (1 - self.sus_params.weight_dist_front) / 2

        dFz_total = self.sus_params.mass * Ay * self.sus_params.cg_height / self.sus_params.trackwidth
        dFz_front = dFz_total * self.sus_params.lateral_load_transfer_dist
        dFz_rear = dFz_total * (1 - self.sus_params.lateral_load_transfer_dist)

        FzFL = max(Fz_static_front - dFz_front / 2, 0.0)
        FzFR = max(Fz_static_front + dFz_front / 2, 0.0)
        FzRL = max(Fz_static_rear - dFz_rear / 2, 0.0)
        FzRR = max(Fz_static_rear + dFz_rear / 2, 0.0)
        return FzFL, FzFR, FzRL, FzRR


    def calculate_downforce(self, V):
        total_downforce = 0.5 * AIR_DENSITY * (V ** 2) * self.aero_params.cla

        DfFL = (total_downforce * self.aero_params.cop) / 2
        DfFR = (total_downforce * self.aero_params.cop) / 2
        DfRL = (total_downforce * (1 - self.aero_params.cop)) / 2
        DfRR = (total_downforce * (1 - self.aero_params.cop)) / 2

        return DfFL, DfFR, DfRL, DfRR