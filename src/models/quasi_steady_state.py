from structs.car import Car
from structs.ggv import GGV
from structs.track import EnduranceTrack, AutocrossTrack, SkidpadTrack

class QuasiSteadyStateModel:
    def __init__(self) -> None:
        self.car = Car()
        self.ggv = GGV()
        self.track = Track()

    def build_car(self) -> Car:
        return self.car

    def build_ggv(self) -> GGV:
        self.ggv.populate_lat_accels()
        self.ggv.populate_long_accels()
        self.ggv.populate_brake_accels()
        return self.ggv

    def load_track(self, path: str) -> Track:
        self.track.load(path)
        return self.track

    def run(self, path: str) -> tuple[Car, GGV, Track]:
        self.build_car()
        self.build_ggv()
        self.load_track(path)
        return self.car, self.ggv, self.track

if __name__ == "__main__":
