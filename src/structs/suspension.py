import json
from ..utils.files import SUSPENSION_DEFAULT_CONFIG_PATH


class Suspension:

    def __init__(self):
        try:
            with open(SUSPENSION_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.set_params(data["settings"][0])
        except Exception as e:
            print("Error when attempting to read file: ", str(e))

    def set_params(self, params):
        self.mass = params["mass"]
        self.cg_height = params["cg_height"]
        self.trackwidth = params["trackwidth"]
        self.wheelbase = params["wheelbase"]
        self.weight_dist_front = params["weight_dist_front"]

        self.roll_rate = params["roll_rate"]
        self.pitch_rate = params["pitch_rate"]
        self.camber_gain_roll = params["camber_gain_roll"]
        self.camber_gain_heave = params["camber_gain_heave"]
        self.yaw_moi = params["yaw_moi"]

        self.lateral_load_transfer_dist = self.roll_rate["F"] / (self.roll_rate["F"] + self.roll_rate["R"]) 
        print("LLTD: ", self.lateral_load_transfer_dist)
