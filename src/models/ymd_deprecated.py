import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import brentq, root
from pathlib import Path

from ..structs.car import Car
from .tire import Tire
from ..utils.files import SUSPENSION_SKIDPAD_CONFIG_PATH

class YMD:

    def __init__(self):

        self.car = Car()
        self.tire = Tire()

        # Variables store angles in degrees; computations on these angles involve changing them to radians first
        self.n_beta = 81
        self.n_delta = 81
        self.beta_limit = 12
        self.delta_limit = 12

        self.Mz_grid = None
        self.Ay_grid = None
        self.beta_range = np.linspace(-self.beta_limit, self.beta_limit, self.n_beta)
        self.delta_range = np.linspace(-self.delta_limit, self.delta_limit, self.n_delta)


    # ---------- Setup ----------
    def setup(self, sus_params=None, aero_params=None, engine_params=None):
        self.car.setup(sus_params=sus_params, aero_params=aero_params)
        self.tire.setup()


    # ---------- YMD Solve Functions ----------
    def solve_point_constant_radius(self, beta, delta, R, max_iter=50, tol=1e-3):

        r = 0.0
        Ay = 0.0
        Fy_front = Fy_rear = 0.0

        sus_params = self.car.sus_params

        a = (1 - sus_params.weight_dist_front) * sus_params.wheelbase
        b = (sus_params.weight_dist_front) * sus_params.wheelbase

        RELAX = 0.3
        converged = False

        for _ in range(max_iter):

            # Step 1: Calculate Velocity
            V = Ay / r if r != 0 else math.sqrt(1.0 * R)
            
            # Step 2: Calculate SA given: Beta, Delta, Previous Yaw Velocity
            alpha_f = math.degrees(math.radians(beta) + (a * r / V) - math.radians(delta))
            alpha_r = math.degrees(math.radians(beta) - (b * r / V))

            # Step 3: Calculate Vertical Loads
            FzFL, FzFR, FzRL, FzRR = self.car.calculate_corner_loads(Ay)
            DfFL, DfFR, DfRL, DfRR = self.car.calculate_downforce(V)

            # Step 4: Calculate Lat. Force
            # TODO: Calculate roll amount based on roll gradient OR roll rate
            FyFL = self.tire.calculate_lat_force_mf52(alpha_f, 0.5, FzFL + DfFL, 68.9476)
            FyFR = self.tire.calculate_lat_force_mf52(alpha_f, 0.5, FzFR + DfFR, 68.9476)
            FyRL = self.tire.calculate_lat_force_mf52(alpha_r, 1.4, FzRL + DfRL, 68.9476)
            FyRR = self.tire.calculate_lat_force_mf52(alpha_r, 1.4, FzRR + DfRR, 68.9476)

            Fy_front = FyFL + FyFR
            Fy_rear = FyRL + FyRR

            # Step 5: Calculate New Ay (Lat. Accel) and r (Yaw Velocity)
            Ay_new = (Fy_front + Fy_rear) / sus_params.mass
            v_new = math.sqrt(abs(Ay_new) * R)
            r_new = Ay_new / v_new

            if abs(Ay_new - Ay) < tol and abs(r_new - r) < tol:
                Ay, r = Ay_new, r_new
                converged = True
                break

            Ay = Ay + RELAX * (Ay_new - Ay)
            r = r + RELAX * (r_new - r)

        Mz = a * Fy_front - b * Fy_rear
        return Ay, Mz, r, alpha_f, alpha_r


    def solve_point_constant_velocity(self, beta, delta, V, max_iter=50, tol=1e-4):
        """
        Given body slip angle beta, steer angle delta, and speed V, iterate
        on yaw rate r (via r = Ay / V) until self-consistent, then return
        the resulting (Ay, Mz).

        If you'd rather skip the feedback loop (faster, less accurate near
        the limit), just set r = 0 and remove the loop — see note below.
        """
        r = 0.0
        Ay = 0.0
        Fy_front = Fy_rear = 0.0

        sus_params = self.car.sus_params

        a = (1 - sus_params.weight_dist_front) * sus_params.wheelbase
        b = (sus_params.weight_dist_front) * sus_params.wheelbase

        RELAX = 0.3

        for _ in range(max_iter):

            alpha_f = math.degrees(math.radians(beta) + (a * r / V) - math.radians(delta))
            alpha_r = math.degrees(math.radians(beta) - (b * r / V))

            FzFL, FzFR, FzRL, FzRR = self.car.calculate_corner_loads(Ay)
            DfFL, DfFR, DfRL, DfRR = self.car.calculate_downforce(V)
            # print("Generated Corner Loads: ", FzFL, FzFR, FzRL, FzRR)

            # TODO: Calculate roll amount based on roll gradient OR roll rate
            FyFL = self.tire.calculate_lat_force_mf52(alpha_f, 0.5, FzFL + DfFL, 68.9476)
            FyFR = self.tire.calculate_lat_force_mf52(alpha_f, 0.5, FzFR + DfFR, 68.9476)
            FyRL = self.tire.calculate_lat_force_mf52(alpha_r, 1.4, FzRL + DfRL, 68.9476)
            FyRR = self.tire.calculate_lat_force_mf52(alpha_r, 1.4, FzRR + DfRR, 68.9476)

            Fy_front = FyFL + FyFR
            Fy_rear = FyRL + FyRR

            Ay_new = (Fy_front + Fy_rear) / sus_params.mass
            r_new = Ay_new / V

            if abs(Ay_new - Ay) < tol and abs(r_new - r) < tol:
                Ay, r = Ay_new, r_new
                break

            #  Ay, r = Ay_new, r_new
            Ay = Ay + (RELAX * (Ay_new - Ay))
            r = Ay / V

        Mz = a * Fy_front - b * Fy_rear
        return Ay, Mz, r, alpha_f, alpha_r


    def build_ymd_constant_radius(self, R):
        Ay_grid = np.zeros((len(self.beta_range), len(self.delta_range)))
        Mz_grid = np.zeros_like(Ay_grid)

        for i, beta in enumerate(self.beta_range):
            for j, delta in enumerate(self.delta_range):
                Ay, Mz, r, alpha_f, alpha_r = self.solve_point_constant_radius(beta, delta, R)

                Ay_grid[i, j] = Ay
                Mz_grid[i, j] = Mz

        print("Max Lateral Accel: ", np.nanmax(Ay_grid))

        self.Ay_grid = Ay_grid
        self.Mz_grid = Mz_grid

        return Ay_grid, Mz_grid



    def build_ymd_constant_velocity(self, V):
        """Sweep beta x delta and return grids of Ay, Mz (shape [beta, delta])."""
        Ay_grid = np.zeros((len(self.beta_range), len(self.delta_range)))
        Mz_grid = np.zeros_like(Ay_grid)

        for i, beta in enumerate(self.beta_range):
            for j, delta in enumerate(self.delta_range):
                Ay, Mz, r, alpha_f, alpha_r = self.solve_point_constant_velocity(beta, delta, V)

                if beta == 0 and delta == 0:
                    print("Alpha_F and Alpha_R: ", alpha_f, alpha_r)

                Ay_grid[i, j] = Ay
                Mz_grid[i, j] = Mz

        print("Max Lateral Accel: ", np.nanmax(Ay_grid))

        self.Ay_grid = Ay_grid
        self.Mz_grid = Mz_grid

        return Ay_grid, Mz_grid


    def plot_ymd(self, title, Ay_grid=None, Mz_grid=None):

        fig, ax = plt.subplots(figsize=(16, 16))

        # constant-beta lines: fix beta (row), sweep delta across columns
        for i in range(self.Ay_grid.shape[0]):
            ax.plot(self.Ay_grid[i, :], self.Mz_grid[i, :], color="tab:blue", lw=0.75)

        # constant-delta lines: fix delta (column), sweep beta across rows
        for j in range(self.Ay_grid.shape[1]):
            ax.plot(self.Ay_grid[:, j], self.Mz_grid[:, j], color="tab:red", lw=0.75)

        ax.axhline(0, color="k", lw=0.75)
        ax.axvline(0, color="k", lw=0.75)
        ax.set_xlabel("Lateral Acceleration Ay [m/s^2]", fontsize=20)
        ax.set_ylabel("Yaw Moment Mz [Nm]", fontsize=20)
        ax.set_title(title, fontsize=20)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(Path(f"figures/ymd/{title}.png").expanduser().resolve())


    # Occurs at Mz = 0
    def analyze_mid_corner(self):

        trim_points = []

        for i, beta in enumerate(self.beta_range):
            row = self.Mz_grid[i, :]
            for j in range(len(self.delta_range) - 1):
                Mz1, Mz2 = row[j], row[j + 1]

                if Mz1 == 0.0:
                    t, delta = 0.0, self.delta_range[j]
                elif np.sign(Mz1) == np.sign(Mz2):
                    continue
                else:
                    t = -Mz1 / (Mz2 - Mz1)
                    delta = self.delta_range[j] + t * (self.delta_range[j+1] - self.delta_range[j])

                Ay = self.Ay_grid[i, j] + t * (self.Ay_grid[i, j+1] - self.Ay_grid[i, j])
                dMz_ddelta = (Mz2 - Mz1) / (self.delta_range[j+1] - self.delta_range[j])

                trim_points.append({"beta": beta,
                                    "delta": delta, "Ay": Ay, "dMz_ddelta": dMz_ddelta })

        return trim_points
    

if __name__ == "__main__":
    skidpad_time = 5.155
    skidpad_radius = 8.3975
    V = (2 * math.pi * skidpad_radius) / skidpad_time

    ymd = YMD()
    ymd.setup()


    # for swd in [0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]:
    #     print("Current SWD: ", swd)

    # ymd.car.sus_params.weight_dist_front = swd

    ymd.build_ymd_constant_radius(skidpad_radius)
    ymd.plot_ymd(f"FSAE Michigan 2026 Skidpad YMD @ R = {skidpad_radius}, SWD = {ymd.car.sus_params.weight_dist_front}")


    mid_corner_results = ymd.analyze_mid_corner()
    print("Mid Corner Results: ", min(mid_corner_results, key = lambda entry : abs(entry["Ay"] - 10.57)))


    # ymd.build_ymd_constant_radius(skidpad_radius)
    # ymd.plot_ymd(f"FSAE Michigan 2026 Skidpad YMD @ R = {skidpad_radius}")

    # corner_entry_results = ymd.analyze_corner_entry(skidpad_radius)
    # print("Corner Entry Results: ", corner_entry_results)

    # print()

    # mid_corner_results = ymd.analyze_mid_corner()
    # # print("Mid Corner Results: ", mid_corner_results)
    # print("Mid Corner Results: ", max(mid_corner_results["dMz_ddelta"], key = lambda entry : entry["Ay"]))

    # print()