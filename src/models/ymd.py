import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import brentq, root
from pathlib import Path

from ..structs.car import Car
from .tire import Tire
from ..utils.constants import G
from ..utils.loads import calculate_corner_loads, calculate_downforce

class YMD:

    def __init__(self):
        self.car = Car()
        self.tire = Tire()

        # Variables store angles in degrees; computations on these angles involve changing them to radians first
        self.n_beta = 241
        self.n_delta = 241
        self.beta_limit = 20
        self.delta_limit = 20

        self.beta_range = np.linspace(-self.beta_limit, self.beta_limit, self.n_beta)
        self.delta_range = np.linspace(-self.delta_limit, self.delta_limit, self.n_delta)


    def solve_point(self, beta, delta, V, max_iter=50, tol=1e-6):
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

        for _ in range(max_iter):
            alpha_f = math.degrees(math.radians(beta) + (a * r / V) - math.radians(delta))
            alpha_r = math.degrees(math.radians(beta) - (b * r / V))

            FzFL, FzFR, FzRL, FzRR = calculate_corner_loads(sus_params, Ay)
            FzDF = calculate_downforce(V) / 4
            # print("Generated Corner Loads: ", FzFL, FzFR, FzRL, FzRR)

            # TODO: Calculate roll amount based on roll gradient OR roll rate
            FyFL = self.tire.calculate_lat_force(FzFL + FzDF, alpha_f, 0)
            FyFR = self.tire.calculate_lat_force(FzFR + FzDF, alpha_f, 0)
            FyRL = self.tire.calculate_lat_force(FzRL + FzDF, alpha_r, 0)
            FyRR = self.tire.calculate_lat_force(FzRR + FzDF, alpha_r, 0)

            Fy_front = FyFL + FyFR
            Fy_rear = FyRL + FyRR

            Ay_new = (Fy_front + Fy_rear) / sus_params.mass
            r_new = Ay_new / V

            if abs(Ay_new - Ay) < tol and abs(r_new - r) < tol:
                Ay, r = Ay_new, r_new
                break
            Ay, r = Ay_new, r_new

        Mz = a * Fy_front - b * Fy_rear
        return Ay, Mz, r, alpha_f, alpha_r


    def build_ymd(self, V):
        """Sweep beta x delta and return grids of Ay, Mz (shape [beta, delta])."""
        Ay_grid = np.zeros((len(self.beta_range), len(self.delta_range)))
        Mz_grid = np.zeros_like(Ay_grid)

        for i, beta in enumerate(self.beta_range):
            for j, delta in enumerate(self.delta_range):
                Ay, Mz, r, alpha_f, alpha_r = self.solve_point(beta, delta, V)

                Ay_grid[i, j] = Ay
                Mz_grid[i, j] = Mz

        print("Max Lateral Accel: ", np.max(Ay_grid))

        self.Ay_grid = Ay_grid
        self.Mz_grid = Mz_grid

        return Ay_grid, Mz_grid


    def plot_ymd(self, title, Ay_grid=None, Mz_grid=None):
        if not hasattr(self, "Mz_grid") or not hasattr(self, "Ay_grid"): 
            print("No YMD built yet.")
            return

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


    def analyze_corner_entry(self, beta=0.0, delta=0.0):
        beta_idx = np.argmin(np.abs(self.beta_range - beta))
        delta_idx = np.argmin(np.abs(self.delta_range - delta))

        if beta_idx == 0 or beta_idx == len(self.beta_range) - 1:
            raise ValueError("Beta is too close to grid boundary.")

        if delta_idx == 0 or delta_idx == len(self.delta_range) - 1:
            raise ValueError("Delta is too close to grid boundary.")

        h_beta = self.beta_range[beta_idx + 1] - self.beta_range[beta_idx]
        h_delta = self.delta_range[delta_idx + 1] - self.delta_range[delta_idx]

        dMz_dbeta = (
            self.Mz_grid[beta_idx + 1, delta_idx]
            - self.Mz_grid[beta_idx - 1, delta_idx]
        ) / (2 * h_beta)

        dMz_ddelta = (
            self.Mz_grid[beta_idx, delta_idx + 1]
            - self.Mz_grid[beta_idx, delta_idx - 1]
        ) / (2 * h_delta)

        return {
            "beta": self.beta_range[beta_idx],
            "delta": self.delta_range[delta_idx],
            "Ay": self.Ay_grid[beta_idx, delta_idx],
            "Mz": self.Mz_grid[beta_idx, delta_idx],
            "dMz_dbeta": dMz_dbeta,
            "dMz_ddelta": dMz_ddelta,
        }


    # Occurs at Mz = 0
    def analyze_mid_corner(self):
        beta_idx, delta_idx = np.unravel_index(np.argmin(np.abs(self.Mz_grid)), self.Mz_grid.shape)

        mid_corner_entries = {
            "dMz_dbeta": [],
            "dMz_ddelta": []
        }

        for beta_idx in range(1, len(self.beta_range)-1):
            for delta_idx in range(1, len(self.delta_range)-1):

                Mz_curr = self.Mz_grid[beta_idx][delta_idx]

                Mz_prev_beta = self.Mz_grid[beta_idx-1][delta_idx]
                Mz_next_beta = self.Mz_grid[beta_idx+1][delta_idx]

                Mz_prev_delta = self.Mz_grid[beta_idx][delta_idx-1]
                Mz_next_delta = self.Mz_grid[beta_idx][delta_idx+1]

                h_beta = self.beta_range[beta_idx + 1] - self.beta_range[beta_idx]
                h_delta = self.delta_range[delta_idx + 1] - self.delta_range[delta_idx]

                # 0.05 = Convergence Tolerance
                if abs(Mz_curr) < 0.05 or (Mz_prev_beta > 0 and Mz_next_beta < 0) or (Mz_prev_beta < 0 and Mz_next_beta > 0):
                    mid_corner_entries["dMz_dbeta"].append({
                        "beta": self.beta_range[beta_idx],
                        "delta": self.delta_range[delta_idx],
                        "Ay": self.Ay_grid[beta_idx][delta_idx],
                        "value": (Mz_next_beta - Mz_prev_beta) / (2 * h_beta) 
                    })

                if abs(Mz_curr) < 0.05 or (Mz_prev_delta > 0 and Mz_next_delta < 0) or (Mz_prev_delta < 0 and Mz_next_delta > 0):
                    mid_corner_entries["dMz_ddelta"].append({
                        "beta": self.beta_range[beta_idx],
                        "delta": self.delta_range[delta_idx],
                        "Ay": self.Ay_grid[beta_idx][delta_idx],
                        "value": (Mz_next_delta - Mz_prev_delta) / (2 * h_delta) 
                    })

        return mid_corner_entries
    

if __name__ == "__main__":
    skidpad_time = 5.6
    skidpad_radius = 8.3975
    V = (2 * math.pi * skidpad_radius) / skidpad_time

    # TODO: Set this with actual value
    V = 13.9

    ymd = YMD()
    ymd.build_ymd(V)
    ymd.plot_ymd("FSAE Michigan 2026 Skidpad YMD")

    corner_entry_results = ymd.analyze_corner_entry()
    print(corner_entry_results)

    mid_corner_results = ymd.analyze_mid_corner()
    # print(mid_corner_results)
