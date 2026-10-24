from pacejka_model_helper import load_data
from pathlib import Path

TYRE_DATA_PATH = Path("../../data/tyres/RunData_DriveBrake_Matlab_SI_Round9").expanduser().resolve()

if __name__ == "__main__":
    data = load_data(TYRE_DATA_PATH, [])