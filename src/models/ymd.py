import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path

from ..structs.car import Car
from .tire import Tire
from ..utils.files import SUSPENSION_SKIDPAD_CONFIG_PATH


class YMD:

    def __init__(self):

        self.car = Car()
        self.tire = Tire()

        self.n_beta = 81
        self.n_delta = 81
        self.beta_limit = 12
        self.delta_limit = 12

        self.Mz_grid = None
        self.Ay_grid = None
        self.r_grid = None
        self.converged_grid = None
        self.R = None

        self.beta_range = np.linspace(-self.beta_limit, self.beta_limit, self.n_beta)
        self.delta_range = np.linspace(-self.delta_limit, self.delta_limit, self.n_delta)

    # ---------- Setup ----------
    def setup(self, sus_params=None, aero_params=None, engine_params=None):
        self.car.setup(sus_params=sus_params, aero_params=aero_params)
        self.tire.setup()
        print("Finished Setup")

    # ---------- Core point solve (constant radius) ----------
    #
    # Reformulation vs. the original relaxation loop:
    #
    # For a car pinned to a circular path of fixed radius R, the kinematic
    # relation r = V/R holds EXACTLY at every instant -- it is not something
    # that needs iterative convergence, only the force balance does. That
    # means alpha_f/alpha_r only depend on the *sign* of Ay (i.e. which way
    # around the circle the car is going), not on its magnitude:
    #
    #     alpha_f = beta + a*sign(Ay)/R - delta
    #     alpha_r = beta - b*sign(Ay)/R
    #
    # (since r/V = sign(Ay)/R identically once Ay = V*r and |r| = V/R).
    #
    # That collapses the original 2-variable coupled relaxation (Ay, r) --
    # which could diverge, land on Ay=0 and divide by zero, or let Ay/r fall
    # out of sync mid-iteration -- into a single-variable root find on the
    # signed residual  f(Ay) = Ay_calc(Ay) - Ay = 0, solved robustly with a
    # bracketing method (brentq) instead of fixed-point relaxation.

    def _axle_distances(self):
        sp = self.car.sus_params
        a = (1 - sp.weight_dist_front) * sp.wheelbase   # CG -> front axle
        b = sp.weight_dist_front * sp.wheelbase          # CG -> rear axle
        return a, b

    def _ay_residual_and_state(self, Ay, beta, delta, R, a, b):
        """Given a trial signed Ay, compute the force-balance Ay and full
        state. Returns (residual, state_dict)."""
        beta_r = math.radians(beta)
        delta_r = math.radians(delta)
        s = 1.0 if Ay >= 0 else -1.0

        alpha_f = math.degrees(beta_r + a * s / R - delta_r)
        alpha_r = math.degrees(beta_r - b * s / R)

        V = math.sqrt(abs(Ay) * R)

        FzFL, FzFR, FzRL, FzRR = self.car.calculate_corner_loads(Ay)
        DfFL, DfFR, DfRL, DfRR = self.car.calculate_downforce(V)

        FyFL = self.tire.calculate_lat_force_surface(alpha_f, 0.5, FzFL + DfFL, 68.9476)
        FyFR = self.tire.calculate_lat_force_surface(alpha_f, 0.5, FzFR + DfFR, 68.9476)
        FyRL = self.tire.calculate_lat_force_surface(alpha_r, 1.4, FzRL + DfRL, 68.9476)
        FyRR = self.tire.calculate_lat_force_surface(alpha_r, 1.4, FzRR + DfRR, 68.9476)

        Fy_front = FyFL + FyFR
        Fy_rear = FyRL + FyRR
        Ay_calc = (Fy_front + Fy_rear) / self.car.sus_params.mass

        state = {
            "Ay": Ay, "Ay_calc": Ay_calc, "V": V, "r": (Ay / V) if V > 0 else 0.0,
            "alpha_f": alpha_f, "alpha_r": alpha_r,
            "Fy_front": Fy_front, "Fy_rear": Fy_rear,
        }
        return Ay_calc - Ay, state

    @staticmethod
    def _find_bracket(f, lo, hi, n):
        """Scan [lo, hi] for a sign change in f (skipping x=0, the kinematic
        singularity in the sign(Ay) term). Returns (x1, x2) or None."""
        xs = np.linspace(lo, hi, n)
        xs = xs[np.abs(xs) > 1e-6]
        if len(xs) < 2:
            return None
        f_prev = f(xs[0])
        for k in range(1, len(xs)):
            f_curr = f(xs[k])
            if np.isfinite(f_prev) and np.isfinite(f_curr) and np.sign(f_prev) != np.sign(f_curr):
                return xs[k - 1], xs[k]
            f_prev = f_curr
        return None

    def solve_point_constant_radius(self, beta, delta, R, Ay_guess=None, Ay_search_max=40.0):
        """Solve for the self-consistent (Ay, Mz, r, alpha_f, alpha_r) at a
        constant-radius trim point via a robust 1D root find, rather than
        fixed-point relaxation. Returns a dict; result["converged"] is False
        (with NaNs) if no valid root could be bracketed -- callers should
        check this rather than trust a possibly-garbage value."""

        a, b = self._axle_distances()
        cache = {}

        def f(Ay):
            resid, state = self._ay_residual_and_state(Ay, beta, delta, R, a, b)
            cache["state"] = state
            return resid

        bracket = None
        if Ay_guess is not None and np.isfinite(Ay_guess) and Ay_guess != 0:
            window = max(2.0, 0.25 * abs(Ay_guess))
            bracket = self._find_bracket(f, Ay_guess - window, Ay_guess + window, n=12)
        if bracket is None:
            bracket = self._find_bracket(f, -Ay_search_max, Ay_search_max, n=80)

        if bracket is None:
            return {"converged": False, "beta": beta, "delta": delta,
                    "Ay": np.nan, "Mz": np.nan, "r": np.nan,
                    "alpha_f": np.nan, "alpha_r": np.nan, "V": np.nan}

        try:
            Ay_star = brentq(f, bracket[0], bracket[1], xtol=1e-6, maxiter=100)
        except (RuntimeError, ValueError):
            return {"converged": False, "beta": beta, "delta": delta,
                    "Ay": np.nan, "Mz": np.nan, "r": np.nan,
                    "alpha_f": np.nan, "alpha_r": np.nan, "V": np.nan}

        # one more evaluation exactly at the root so `cache` isn't just
        # "close to" the answer from brentq's last internal step
        f(Ay_star)
        state = cache["state"]
        Mz = a * state["Fy_front"] - b * state["Fy_rear"]

        return {"converged": True, "beta": beta, "delta": delta,
                "Ay": Ay_star, "Mz": Mz, "r": state["r"],
                "alpha_f": state["alpha_f"], "alpha_r": state["alpha_r"], "V": state["V"]}

    # ---------- Grid build ----------
    def build_ymd_constant_radius(self, R, Ay_search_max=40.0):
        n_b, n_d = len(self.beta_range), len(self.delta_range)
        Ay_grid = np.full((n_b, n_d), np.nan)
        Mz_grid = np.full((n_b, n_d), np.nan)
        r_grid = np.full((n_b, n_d), np.nan)
        converged = np.zeros((n_b, n_d), dtype=bool)

        n_failed = 0
        for i, beta in enumerate(self.beta_range):
            # warm-start the first delta of this row from the point directly
            # above (same delta, previous beta) if it converged
            row_guess = Ay_grid[i - 1, 0] if i > 0 and converged[i - 1, 0] else None

            for j, delta in enumerate(self.delta_range):
                guess = row_guess if j == 0 else (Ay_grid[i, j - 1] if converged[i, j - 1] else None)

                result = self.solve_point_constant_radius(beta, delta, R, Ay_guess=guess,
                                                            Ay_search_max=Ay_search_max)
                if result["converged"]:
                    Ay_grid[i, j] = result["Ay"]
                    Mz_grid[i, j] = result["Mz"]
                    r_grid[i, j] = result["r"]
                    converged[i, j] = True
                else:
                    n_failed += 1

        if n_failed:
            print(f"Warning: {n_failed}/{n_b * n_d} grid points failed to converge (left as NaN).")

        print("Max |Lateral Accel|: ", np.nanmax(np.abs(Ay_grid)))

        self.Ay_grid = Ay_grid
        self.Mz_grid = Mz_grid
        self.r_grid = r_grid
        self.converged_grid = converged
        self.R = R
        return Ay_grid, Mz_grid

    # ---------- Plotting ----------
    def plot_ymd(self, title, save_dir="figures/ymd"):
        fig, ax = plt.subplots(figsize=(16, 16))

        for i in range(self.Ay_grid.shape[0]):
            ax.plot(self.Ay_grid[i, :], self.Mz_grid[i, :], color="tab:blue", lw=0.75)
        for j in range(self.Ay_grid.shape[1]):
            ax.plot(self.Ay_grid[:, j], self.Mz_grid[:, j], color="tab:red", lw=0.75)

        ax.axhline(0, color="k", lw=0.75)
        ax.axvline(0, color="k", lw=0.75)
        ax.set_xlabel("Lateral Acceleration Ay [m/s^2]", fontsize=20)
        ax.set_ylabel("Yaw Moment Mz [Nm]", fontsize=20)
        ax.set_title(title, fontsize=20)
        ax.grid(True)
        fig.tight_layout()
        out_path = Path(save_dir).expanduser().resolve() / f"{title}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        return out_path

    # ---------- Trim line (Mz = 0) extraction ----------
    def find_trim_line(self, R=None, refine=True, h=1e-3):
        """Locate all Mz=0 crossings along each beta row. Uses linear
        interpolation between adjacent grid columns for the initial bracket
        (fixes the earlier version, which reported the nearest *grid node*
        rather than the actual crossing), then optionally polishes each
        crossing with the robust point solver and computes dMz/ddelta and
        dMz/dbeta there via true central differences (not grid interpolation).

        Handles multiple crossings per beta row, which do occur near the
        limit as the MMM curve folds back."""
        if self.Mz_grid is None:
            raise RuntimeError("Call build_ymd_constant_radius first.")
        R = R if R is not None else self.R
        if R is None:
            raise ValueError("R not provided and not stored from a previous build.")

        trim_points = []
        for i, beta in enumerate(self.beta_range):
            row = self.Mz_grid[i, :]
            for j in range(len(self.delta_range) - 1):
                Mz1, Mz2 = row[j], row[j + 1]
                if not (np.isfinite(Mz1) and np.isfinite(Mz2)):
                    continue
                if Mz1 == 0.0:
                    t = 0.0
                elif np.sign(Mz1) == np.sign(Mz2):
                    continue
                else:
                    t = -Mz1 / (Mz2 - Mz1)

                d1, d2 = self.delta_range[j], self.delta_range[j + 1]
                delta0 = d1 + t * (d2 - d1)
                Ay0 = self.Ay_grid[i, j] + t * (self.Ay_grid[i, j + 1] - self.Ay_grid[i, j])
                dMz_ddelta_secant = (Mz2 - Mz1) / (d2 - d1)

                point = {"beta": beta, "delta": delta0, "Ay": Ay0,
                         "Mz": 0.0, "dMz_ddelta": dMz_ddelta_secant,
                         "dMz_dbeta": np.nan, "refined": False}

                if refine:
                    refined = self._refine_trim_point(beta, d1, d2, R, h=h)
                    if refined is not None:
                        point.update(refined)

                trim_points.append(point)

        trim_points.sort(key=lambda p: p["beta"])
        return trim_points

    def _refine_trim_point(self, beta, delta_lo, delta_hi, R, h=1e-3):
        """Polish a bracketed Mz=0 crossing with the actual point solver, then
        take central differences on the real physics (not interpolated grid
        values) for dMz/ddelta and dMz/dbeta at that exact trim point."""

        def mz_of_delta(delta, guess=None):
            res = self.solve_point_constant_radius(beta, delta, R, Ay_guess=guess)
            return res

        res_lo = mz_of_delta(delta_lo)
        res_hi = mz_of_delta(delta_hi)
        if not (res_lo["converged"] and res_hi["converged"]):
            return None
        if np.sign(res_lo["Mz"]) == np.sign(res_hi["Mz"]):
            return None

        def f(delta):
            res = self.solve_point_constant_radius(beta, delta, R, Ay_guess=res_lo["Ay"])
            return res["Mz"] if res["converged"] else np.nan

        try:
            delta_star = brentq(f, delta_lo, delta_hi, xtol=1e-5, maxiter=100)
        except (RuntimeError, ValueError):
            return None

        center = self.solve_point_constant_radius(beta, delta_star, R, Ay_guess=res_lo["Ay"])
        if not center["converged"]:
            return None

        dp = self.solve_point_constant_radius(beta, delta_star + h, R, Ay_guess=center["Ay"])
        dm = self.solve_point_constant_radius(beta, delta_star - h, R, Ay_guess=center["Ay"])
        bp = self.solve_point_constant_radius(beta + h, delta_star, R, Ay_guess=center["Ay"])
        bm = self.solve_point_constant_radius(beta - h, delta_star, R, Ay_guess=center["Ay"])

        dMz_ddelta = ((dp["Mz"] - dm["Mz"]) / (2 * h)) if (dp["converged"] and dm["converged"]) else np.nan
        dMz_dbeta = ((bp["Mz"] - bm["Mz"]) / (2 * h)) if (bp["converged"] and bm["converged"]) else np.nan

        return {"delta": delta_star, "Ay": center["Ay"], "Mz": center["Mz"],
                "dMz_ddelta": dMz_ddelta, "dMz_dbeta": dMz_dbeta, "refined": True}

    # ---------- Steady-state balance along the trim line ----------
    def analyze_trim_line_balance(self, R=None, refine=True, break_threshold_deg=1.0):
        """Walk the Mz=0 trim line and report, at each trim point:

          - dMz/ddelta : control (steering) derivative -- front-axle yaw
            authority at that trim condition. This is what you asked for.
          - dMz/dbeta  : stability derivative -- the standard MMM read on
            self-correcting (dMz/dbeta < 0) vs. divergent/spin (> 0) yaw
            behavior at the limit.
          - ddelta/dAy : the classic SAE constant-radius steer gradient
            (secant along the trim line itself). Positive = understeer,
            negative = oversteer, ~0 = neutral. This -- not dMz/ddelta -- is
            the textbook-standard US/OS number; it's included here because
            dMz/ddelta by itself mainly tells you steering *sensitivity*,
            not which way the balance leans. See notes below.

        Trend to watch: if |dMz/ddelta| shrinks toward zero as Ay increases
        along the trim line, the front axle is running out of steering
        authority faster than the rear is losing grip -- a classic
        limit-understeer symptom. A roughly constant |dMz/ddelta| across the
        Ay range suggests the front/rear are saturating together.

        Branch handling: near the grip limit, a single beta can satisfy
        Mz=0 at more than one delta (the MMM curve folds back) -- verified
        this actually happens on a real grid, not just theoretically: about
        a third of the beta rows in testing produced 2-3 crossings.

        Two different bugs showed up here during testing, both now fixed:

        1. Picking "the" trim point per beta by smallest |delta| is NOT
           robust -- when multiple roots sit close together in delta (as
           they do for a lot of these betas), that static per-row rule can
           flip between genuinely unrelated branches from one beta to the
           next, producing a discontinuous, physically meaningless curve.
           Fixed with a proper numerical continuation: start from the beta
           nearest zero, seed on its smallest-|delta| root, then walk
           outward in both beta directions always choosing whichever root
           at the next beta is closest in delta to the previous step's
           choice. That keeps the walk on one continuous branch through
           bifurcations instead of jumping.

        2. Even with a single well-defined branch, computing ddelta/dAy by
           sorting points by Ay and differencing Ay-adjacent neighbors
           breaks down wherever Ay is non-monotonic along the branch (which
           does happen near the limit -- a real reversal, not just a stub
           artifact). Fixed by differentiating along the branch's *actual*
           parametrization (beta, which is monotonic by construction: it's
           beta_range) instead of re-sorting by Ay. ddelta/dAy can still
           legitimately spike or change sign right at a fold in Ay(beta);
           that reflects the trim manifold's real shape there, not a bug.

        The non-primary crossings are still returned (as "secondary") with
        their own dMz/ddelta and dMz/dbeta, but without a ddelta/dAy, since
        they aren't part of the one continuous branch being differentiated.

        Known remaining limitation: "nearest delta to the previous step"
        continuation can still jump at a genuine fold/merge point in the
        trim manifold -- verified this happens (not hypothetical): a branch
        can simply not exist yet at one beta and appear at the next, so
        continuation is forced onto a different sheet with no continuous
        option available. Rather than silently report a derivative computed
        across that seam, consecutive-beta steps where |delta| jumps by
        more than `break_threshold_deg` are treated as a break: the branch
        is split there and ddelta/dAy is left as NaN on both sides of the
        seam. This is a heuristic safeguard, not a rigorous bifurcation
        tracker -- if it fires often on your real tire data, it's a sign
        the beta/delta grid is too coarse to resolve the fold, and refining
        the grid (or moving to real pseudo-arclength continuation) is the
        next step up in rigor.
        """
        trim = self.find_trim_line(R=R, refine=refine)
        trim = [p for p in trim if p.get("refined")]

        by_beta = {}
        for p in trim:
            by_beta.setdefault(round(p["beta"], 6), []).append(p)
        betas_sorted = sorted(by_beta.keys())
        if not betas_sorted:
            return {"primary": [], "secondary": []}

        start = min(range(len(betas_sorted)), key=lambda i: abs(betas_sorted[i]))

        chosen = {}
        seed = min(by_beta[betas_sorted[start]], key=lambda p: abs(p["delta"]))
        chosen[betas_sorted[start]] = seed

        prev = seed
        for i in range(start + 1, len(betas_sorted)):
            b = betas_sorted[i]
            cand = min(by_beta[b], key=lambda p: abs(p["delta"] - prev["delta"]))
            chosen[b] = cand
            prev = cand

        prev = seed
        for i in range(start - 1, -1, -1):
            b = betas_sorted[i]
            cand = min(by_beta[b], key=lambda p: abs(p["delta"] - prev["delta"]))
            chosen[b] = cand
            prev = cand

        primary = [chosen[b] for b in betas_sorted]  # already beta-ordered
        secondary = [p for p in trim if p is not chosen[round(p["beta"], 6)]]

        # flag breaks: steps where the branch jumped rather than continued
        is_break = [False] * len(primary)
        n_breaks = 0
        for k in range(1, len(primary)):
            if abs(primary[k]["delta"] - primary[k - 1]["delta"]) > break_threshold_deg:
                is_break[k] = True
                n_breaks += 1
        if n_breaks:
            print(f"Note: {n_breaks} branch break(s) detected in the primary trim line "
                  f"(fold/merge in the trim manifold, or grid too coarse to resolve it). "
                  f"ddelta/dAy left as NaN across each break.")

        for k in range(len(primary)):
            if k == 0 or k == len(primary) - 1:
                primary[k]["ddelta_dAy"] = np.nan
                continue
            if is_break[k] or is_break[k + 1]:
                # the k-1->k or k->k+1 step crossed a detected branch break;
                # a derivative spanning it wouldn't mean anything physically
                primary[k]["ddelta_dAy"] = np.nan
                continue
            dAy = primary[k + 1]["Ay"] - primary[k - 1]["Ay"]
            ddelta = primary[k + 1]["delta"] - primary[k - 1]["delta"]
            primary[k]["ddelta_dAy"] = (ddelta / dAy) if dAy != 0 else np.nan

        secondary.sort(key=lambda p: p["Ay"])
        for p in secondary:
            p["ddelta_dAy"] = np.nan

        return {"primary": primary, "secondary": secondary}

    def plot_trim_balance(self, balance, title, save_dir="figures/ymd"):
        """Plot dMz/ddelta, dMz/dbeta, and ddelta/dAy vs. Ay along the
        primary trim line, so the understeer/oversteer trend is visible
        directly rather than read off a table."""
        primary = balance["primary"]
        Ay = np.array([p["Ay"] for p in primary])
        order = np.argsort(Ay)  # display order only; derivatives already computed along beta
        Ay = Ay[order]
        dMz_ddelta = np.array([p["dMz_ddelta"] for p in primary])[order]
        dMz_dbeta = np.array([p["dMz_dbeta"] for p in primary])[order]
        ddelta_dAy = np.array([p["ddelta_dAy"] for p in primary])[order]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(Ay, dMz_ddelta, "o-", color="tab:blue", ms=3)
        axes[0].axhline(0, color="k", lw=0.75)
        axes[0].set_ylabel("dMz/ddelta [Nm/deg]")
        axes[0].set_title("Control derivative (steering yaw authority)")
        axes[0].grid(True)

        axes[1].plot(Ay, dMz_dbeta, "o-", color="tab:red", ms=3)
        axes[1].axhline(0, color="k", lw=0.75)
        axes[1].set_ylabel("dMz/dbeta [Nm/deg]")
        axes[1].set_title("Stability derivative (<0 stable, >0 divergent)")
        axes[1].grid(True)

        axes[2].plot(Ay, ddelta_dAy, "o-", color="tab:green", ms=3)
        axes[2].axhline(0, color="k", lw=0.75)
        axes[2].set_ylabel("ddelta/dAy [deg/(m/s^2)]")
        axes[2].set_xlabel("Lateral Acceleration Ay [m/s^2]")
        axes[2].set_title("Classic SAE steer gradient (>0 understeer, <0 oversteer)")
        axes[2].grid(True)

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        out_path = Path(save_dir).expanduser().resolve() / f"{title}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        return out_path


if __name__ == "__main__":
    skidpad_time = 5.155
    skidpad_radius = 8.3975

    ymd = YMD()
    ymd.setup()

    ymd.build_ymd_constant_radius(skidpad_radius)
    ymd.plot_ymd(f"FSAE Michigan 2026 Skidpad YMD @ R = {skidpad_radius}")

    balance = ymd.analyze_trim_line_balance()
    primary = balance["primary"]
    print(f"\n{len(primary)} primary trim points found "
          f"({len(balance['secondary'])} secondary/fold-back points excluded from ddelta/dAy).\n")
    for p in primary:
        print(f"beta={p['beta']:6.2f}  delta={p['delta']:7.3f}  Ay={p['Ay']:7.3f}  "
              f"dMz/ddelta={p['dMz_ddelta']:9.2f}  dMz/dbeta={p['dMz_dbeta']:9.2f}  "
              f"ddelta/dAy={p['ddelta_dAy']:7.4f}")

    ymd.plot_trim_balance(balance)