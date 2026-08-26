import math
import numpy as np

from scipy.integrate import solve_ivp
from .tire import Tire
from ..structs.car import Car
from ..utils.constants import G

class StepSteer():


    def __init__(self):
        self.car = Car()
        self.tire = Tire()

    def setup(self, sus_params=None, aero_params=None, engine_params=None):
        self.car.setup(sus_params=sus_params, aero_params=aero_params)
        self.tire.setup()
        print("Finished Setup")


    def _steering_input(self, t, delta_step=5):
        if t > 1:
            return delta_step
        return 0

    def step(self, t, state, V):

        beta, r = state

        sus_params = self.car.sus_params

        delta = self._steering_input(t)

        a = sus_params.wheelbase * sus_params.weight_dist_front
        b = sus_params.wheelbase * (1 - sus_params.weight_dist_front)

        # Tire slip angles
        alpha_f = math.degrees(math.radians(beta) + a*r/V - math.radians(delta))
        alpha_r = math.degrees(math.radians(beta) - b*r/V)

        # Evenly distributed vertical load --> assumes tires are not yet near saturated point
        Ay = 0.0
        FyFL, FyFR, FyRL, FyRR = 0, 0, 0, 0  

        for _ in range(50):

            FzFL, FzFR, FzRL, FzRR = self.car.calculate_corner_loads(Ay)
            DfFL, DfFR, DfRL, DfRR = self.car.calculate_downforce(V)

            # Fz_f = sus_params.mass * G * b / (a + b)
            # Fz_r = sus_params.mass * G * a / (a + b)

            # Tire forces from your nonlinear tire model
            FyFL = self.tire.calculate_lat_force_surface(alpha_f, 0.4, FzFL + DfFL, 83)
            FyFR = self.tire.calculate_lat_force_surface(alpha_f, 0.4, FzFR + DfFR, 83)
            FyRL = self.tire.calculate_lat_force_surface(alpha_r, 0.4, FzRL + DfRL, 83)
            FyRR = self.tire.calculate_lat_force_surface(alpha_r, 0.4, FzRR + DfRR, 83)

            Fy_total = FyFL + FyFR + FyRL + FyRR
            Ay_new = Fy_total / sus_params.mass

            if abs(Ay_new - Ay) < 0.001:
                Ay = Ay_new
                break

            Ay = Ay_new

        # if 1.0 < t < 1.01:

        #     print("\nV =", V)
        #     print("beta =", beta)
        #     print("r =", r)

        #     print("alpha_f =", alpha_f)
        #     print("alpha_r =", alpha_r)

        #     print("FzFL =", FzFL + DfFL)
        #     print("FzFR =", FzFR + DfFR)
        #     print("FzRL =", FzRL + DfRL)
        #     print("FzRR =", FzRR + DfRR)

        #     print("FyF =", FyFL + FyFR)
        #     print("FyR =", FyRL + FyRR)

        #     print("Ay =", Ay)

        # Vehicle equations
        # beta_dot = (Fy_f + Fy_r)/(sus_params.mass*V) - r
        beta_dot = (Ay / V) - r

        Mz = a*(FyFL + FyFR) - b*(FyRL + FyRR)

        r_dot = Mz / sus_params.yaw_moi

        return [beta_dot, r_dot]

    def simulate(self, time_interval, initial_state, V):
        sol = solve_ivp(
            self.step,
            time_interval,
            initial_state,
            args=(V,),
            t_eval=np.linspace(time_interval[0], time_interval[1], 20)
        )

        return sol

if __name__ == "__main__":

    for v in [5, 10, 15, 20]:
        step_steer = StepSteer()
        step_steer.setup()
        print("Velocity: ", v)
        print("WD: ", step_steer.car.sus_params.weight_dist_front)

        output = step_steer.simulate([0, 5], [0, 0], v)
        print(output)

        step_steer.car.sus_params.weight_dist_front = 0.4
        output = step_steer.simulate([0, 5], [0, 0], v)
        print(output)

        print("==============================================")