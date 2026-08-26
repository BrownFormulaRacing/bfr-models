import json
from ..utils.files import AERO_DEFAULT_CONFIG_PATH


class Aero:

    def __init__(self):
        try:
            with open(AERO_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.set_params(data["settings"][0])
        except Exception as e:
            print("Error when attempting to read file: ", str(e))


    def set_params(self, params):
        self.cla = params["cla"]
        self.cda = params["cda"]
        self.cop = params["cop"]
        