#!/usr/bin/env python3
"""
Rocker / bell-crank motion ratio estimator (pushrod -> rocker -> damper)

Implements the 2D cosine-rule iteration from:
  https://waveydynamics.com/blog/rising-rate-suspension/

Geometry (see Figure 1 in the article)
--------------------------------------
  Lp     : pushrod length (pivot-arm end to wheel/upright mount)
  L1     : rocker pushrod-arm radius (pivot -> pushrod pickup)
  Ld     : damper length (rocker damper pickup -> chassis mount)
  L2     : rocker damper-arm radius (pivot -> damper pickup)
  alpha  : angle between L1 and Lp (at the pushrod pickup)
  beta   : angle between L2 and Ld (at the damper pickup)

  d1 = fixed distance, rocker pivot -> pushrod wheel-end mount
  d2 = fixed distance, rocker pivot -> damper chassis mount

Method per wheel-travel step
----------------------------
  1. Lp' = Lp + dz                       (pushrod "extends" with bump)
  2. alpha' from triangle (Lp', L1, d1)
  3. gamma' from the same triangle -> rocker rotation = gamma' - gamma0
  4. theta' = theta0 - rotation          (damper side rotates with the rocker)
  5. Ld' from triangle (d2, L2, theta')
  6. beta' from triangle (Ld', L2, d2)
  7. Leverage  = L2 sin(beta') / (L1 sin(alpha'))
     Motion ratio (finite)  = (Ld0 - Ld') / (Lp' - Lp0)
     Motion ratio (instant) = -dLd/dLp   (equals the leverage ratio)

Same simplification as the article: the pushrod is treated as simply changing
length with wheel travel (the upright pickup path is not modelled), and wheel
travel is taken as equal to pushrod length change unless you supply
--wheel-to-pushrod (wheel->pushrod motion ratio).

Usage
-----
Edit the PARAMS dictionary below, then run:
  python rocker_motion_ratio.py
"""

import argparse
import math

import numpy as np

# --------------------------------------------------------------------------
# EDIT THESE VALUES
# --------------------------------------------------------------------------
PARAMS = {
    # Rocker geometry (mm / degrees)
    "Lp": 304.8,      # Pushrod length
    "L1": 126,       # Pushrod-arm radius (pivot -> pushrod pickup)
    "Ld": 156,      # Damper length
    "L2": 141.33,       # Damper-arm radius (pivot -> damper pickup)
    "alpha": 80.879022,    # Initial angle between L1 and Lp, deg
    "beta": 47.5,     # Initial angle between L2 and Ld, deg

    # Wheel -> pushrod motion ratio (pushrod mm per wheel mm; 1.0 = article)
    "wheel_to_pushrod": 1.0,

    # Travel range to sweep (mm)
    "bump": 30,     # Max bump travel
    "droop": 0,     # Max droop travel
    "step": 1.0,    # Step size

    # Optional outputs
    "spring_rate": None,   # N/mm, e.g. 100 -> adds wheel-rate column
    "csv_path": None,      # e.g. "results.csv"
    "plot": True,          # Show matplotlib plots
}


# --------------------------------------------------------------------------
# Core geometry
# --------------------------------------------------------------------------
def _clip_acos(x):
    """arccos with clipping so tiny float overshoots don't produce NaNs."""
    return np.arccos(np.clip(x, -1.0, 1.0))


def _side_from_angle(a, b, ang):
    """Cosine rule: third side given two sides and the included angle."""
    return np.sqrt(a**2 + b**2 - 2.0 * a * b * np.cos(ang))


def _angle_from_sides(opp, s1, s2):
    """Cosine rule: angle opposite side `opp`, between sides s1 and s2."""
    return _clip_acos((s1**2 + s2**2 - opp**2) / (2.0 * s1 * s2))


class Rocker:
    def __init__(self, Lp, L1, Ld, L2, alpha_deg, beta_deg,
                 wheel_to_pushrod=1.0):
        self.Lp0, self.L1, self.Ld0, self.L2 = Lp, L1, Ld, L2
        self.alpha0 = math.radians(alpha_deg)
        self.beta0 = math.radians(beta_deg)
        self.wp = wheel_to_pushrod  # pushrod travel per mm of wheel travel

        # Fixed construction lines (don't change during travel)
        self.d1 = _side_from_angle(Lp, L1, self.alpha0)
        self.d2 = _side_from_angle(Ld, L2, self.beta0)

        # Rocker angles at design (ride-height) condition
        self.gamma0 = _angle_from_sides(Lp, L1, self.d1)    # opposite Lp
        self.theta0 = _angle_from_sides(Ld, L2, self.d2)    # opposite Ld

    def solve(self, wheel_travel):
        """
        wheel_travel : array of wheel displacements in mm (+bump, -droop).
        Returns a dict of numpy arrays.
        """
        wheel_travel = np.asarray(wheel_travel, dtype=float)
        dLp = wheel_travel * self.wp
        Lp = self.Lp0 + dLp

        alpha = _angle_from_sides(self.d1, Lp, self.L1)     # opposite d1
        gamma = _angle_from_sides(Lp, self.L1, self.d1)     # opposite Lp
        rot = gamma - self.gamma0                           # rocker rotation
        theta = self.theta0 - rot
        Ld = _side_from_angle(self.d2, self.L2, theta)
        beta = _angle_from_sides(self.d2, self.L2, Ld)      # opposite d2

        with np.errstate(divide="ignore", invalid="ignore"):
            leverage = self.L2 * np.sin(beta) / (self.L1 * np.sin(alpha))
        # Pushrod collinear with its rocker arm (alpha ~ 0/180): geometry limit
        leverage = np.where(np.sin(alpha) < 1e-3, np.nan, leverage)
        dLd = self.Ld0 - Ld          # +ve = damper compression

        # Finite motion ratio relative to ride height (article definition)
        with np.errstate(divide="ignore", invalid="ignore"):
            mr_finite = np.where(np.abs(dLp) > 1e-12, dLd / dLp, np.nan)
        # Instantaneous (local slope) motion ratio, pushrod -> damper
        mr_inst = leverage.copy()   # equals dLd/dLp (verified vs. finite diff)
        # Reference to wheel travel
        mr_wheel_inst = mr_inst * self.wp

        return dict(
            wheel_travel=wheel_travel, pushrod_length=Lp,
            alpha_deg=np.degrees(alpha), beta_deg=np.degrees(beta),
            gamma_deg=np.degrees(gamma), theta_deg=np.degrees(theta),
            rocker_rotation_deg=np.degrees(rot),
            damper_length=Ld, damper_travel=dLd,
            leverage=leverage,
            mr_finite=mr_finite * self.wp,      # damper mm per wheel mm
            mr_instant=mr_wheel_inst,           # damper mm per wheel mm
        )


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main(P=PARAMS):
    rk = Rocker(P["Lp"], P["L1"], P["Ld"], P["L2"],
                P["alpha"], P["beta"], P["wheel_to_pushrod"])

    n = int(round((P["bump"] + P["droop"]) / P["step"])) + 1
    travel = np.linspace(-P["droop"], P["bump"], n)
    r = rk.solve(travel)

    if P["spring_rate"] is not None:
        r["wheel_rate_N_per_mm"] = P["spring_rate"] * r["mr_instant"] ** 2

    ab = P["alpha"] + P["beta"]
    print(f"d1 = {rk.d1:.2f} mm, d2 = {rk.d2:.2f} mm")
    print(f"gamma0 = {math.degrees(rk.gamma0):.2f} deg, "
          f"theta0 = {math.degrees(rk.theta0):.2f} deg")
    print(f"alpha+beta at ride height = {ab:.1f} deg "
          f"({'progressive' if ab < 180 else 'regressive/linear'}"
          " per article rule of thumb)\n")

    cols = ["wheel_travel", "damper_length", "damper_travel", "alpha_deg",
            "beta_deg", "rocker_rotation_deg", "leverage",
            "mr_finite", "mr_instant"]
    if "wheel_rate_N_per_mm" in r:
        cols.append("wheel_rate_N_per_mm")
    print("  ".join(f"{c[:14]:>14}" for c in cols))
    stride = max(1, n // 25)
    for i in range(0, n, stride):
        print("  ".join(f"{r[c][i]:14.3f}" for c in cols))

    if np.any(np.isnan(r["leverage"])):
        print("\nWARNING: pushrod becomes collinear with the rocker arm "
              "(alpha ~ 0 deg) inside this range - geometry limit reached; "
              "results beyond it are invalid.")

    if np.any(np.diff(r["damper_length"]) * np.sign(travel[-1] - travel[0]) > 0):
        print("\nWARNING: damper length is not monotonic with wheel travel - "
              "the rocker goes over centre in this range (regressive/lock-up risk).")

    if P["csv_path"]:
        import csv
        with open(P["csv_path"], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i in range(n):
                w.writerow([f"{r[c][i]:.5f}" for c in cols])
        print(f"\nWrote {P['csv_path']}")

    if P["plot"]:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(travel, r["damper_travel"])
        ax[0].set(xlabel="Wheel travel (mm)", ylabel="Damper travel (mm)",
                  title="Wheel vs damper travel")
        ax[1].plot(travel, r["mr_instant"], label="Instantaneous")
        ax[1].plot(travel, r["mr_finite"], "--", label="Finite (from ride height)")
        ax[1].set(xlabel="Wheel travel (mm)", ylabel="Motion ratio",
                  title="Motion ratio vs wheel travel")
        ax[1].legend()
        ax[2].plot(travel, r["alpha_deg"], label="alpha")
        ax[2].plot(travel, r["beta_deg"], label="beta")
        ax[2].set(xlabel="Wheel travel (mm)", ylabel="Angle (deg)",
                  title="Rocker angles")
        ax[2].legend()
        for a in ax:
            a.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()