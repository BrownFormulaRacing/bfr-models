from pathlib import Path

import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares
from scipy.interpolate import make_smoothing_spline, RegularGridInterpolator
import numpy as np
import math
from ..utils.files import DEFAULT_RUN_ID, DEFAULT_LAT_DATA_PATH, DEFAULT_LAT_SURFACE_PATH, DEFAULT_LAT_COEFFS_PATH, DEFAULT_LAT_MF52_COEFFS_PATH

CHANNELS = set(["ET", "FX", "FY", "AMBTMP", "FZ", "IA", "MX", "MZ", "N", "P", "RE", "RL", "RST", "SA", "SL", "SR", "TSTC", "TSTI", "TSTO", "V"])
FY_SCALING_FACTOR = 0.7

LAT_BNP89_COEFF_NAMES = [
    "a0", "a1", "a2", "a3", "a4", "a5", "a6",
    "a7", "a8", "a9", "a10", "a11", "a12", "a13"]
LAT_MF52_COEFF_NAMES = [
    "pCy1", "pDy1", "pDy2", "pDy3",
    "pEy1", "pEy2", "pEy3", "pEy4",
    "pKy1", "pKy2", "pKy3",
    "pHy1", "pHy2", "pHy3",
    "pVy1", "pVy2", "pVy3", "pVy4"]
LONG_MF52_COEFF_NAMES = []

# initial guess / bounds for the 18 lat mf5.2 coeffs (same order as LAT_MF52_COEFF_NAMES)
LAT_MF52_X0 = [1.4, 2.5, -0.2, 0, 0, 0, 0, 0, -40, 1.5, 0, 0, 0, 0, 0, 0, 0, 0]
LAT_MF52_LB = [1.0, 0.5, -2, -50, -10, -10, -10, -50, -200, 0.1, -10, -0.1, -0.1, -1, -0.5, -0.5, -5, -5]
LAT_MF52_UB = [2.0, 5.0, 2, 50, 1, 10, 10, 50, -1, 10, 10, 0.1, 0.1, 1, 0.5, 0.5, 5, 5]
LAT_MF52_MIN_POINTS = 500

SA_RANGE = np.linspace(-14, 14, 300)
FZ_BUCKETS = [230, 400, 630, 900, 1100]
IA_BUCKETS = [0, 2, 4]
P_BUCKETS = [41.3685, 55.1581, 68.9476, 82.7371, 96.5266] # in kPa (when hot) - (6, 8, 10, 12, 14) psi choices

# ordinal blue ramp, light -> heavy load
LOAD_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]

FZ_TOL = 50
IA_TOL = 0.5
P_TOL = 5

class Tire:
    def __init__(self, run_id=DEFAULT_RUN_ID, data_path=DEFAULT_LAT_DATA_PATH, coeffs_path=DEFAULT_LAT_COEFFS_PATH, surface_path=DEFAULT_LAT_SURFACE_PATH, mf52_coeffs_path=DEFAULT_LAT_MF52_COEFFS_PATH):

        self.run_id = run_id
        self.data_path = data_path
        self.coeffs_path = coeffs_path
        self.surface_path = surface_path
        self.mf52_coeffs_path = mf52_coeffs_path
        self.tire_id_substrings = ["43075", "16x7.5", "8 inch rim"]

        self.lat_mf4_coeffs = dict()
        self.lat_mf52_coeffs = dict() # pressure -> {coeff name -> value, "fz0" -> value}
        self.lat_bnp89_coeffs = dict()

        self.lat_splines = dict()
        self.lat_surface = None


    # ---------- Setup ----------
    def setup(self):
        try:
            # Load the data
            self.load_data()

            # Process the data
            self.process_data()

            if self.mf52_coeffs_path.is_file():
                self.load_lat_mf52_coeffs()
            else:
                self.fit_lat_mf52()
                self.save_lat_mf52_coeffs()

            self.print_lat_mf52_coeffs()
        except Exception as e:
            print("Failed to setup tire object: ", str(e))


    def load_lat_surface(self):
        with open(self.surface_path, "rb") as f:
            self.lat_surface = pickle.load(f)
            print("SURFACE: ", self.lat_surface)


    def save_lat_surface(self):
        with open(self.surface_path, "wb") as f:
            pickle.dump(self.lat_surface, f)


    # ---------- For Model Invocation ----------

    def calculate_lat_force_bnp89(self, alpha: float,  gamma: float, fz: float) -> float:
        """Pacejka BNP 1989 lateral force model.

        Args:
            vertical_load: normal (vertical) load, N. Must be positive -- the fit was
                done against abs(FZ) (see hoosier_r20_tire_model_lat.m section 2:
                `FZ_pos = abs(FZ_all)`), so a negative fz here is a caller bug,
                not a valid "unloaded tire" input.
            camber_angle: inclination angle, deg (gamma)
            slip_angle: slip angle, deg (alpha)

        Returns:
            Fy, N. Sign follows the SAE convention baked into the fit
            (D is negative -- see stage-1 fitting notes in the .m script).
        """

        if fz < 0:
            raise ValueError(f"fz must be a positive load magnitude, got {fz}")

        c = self.lat_bnp89_coeffs  # shorthand

        C = c["a0"]
        D = c["a1"] * fz**2 + c["a2"] * fz
        BCD = c["a3"] * math.sin(2 * math.atan(fz / c["a4"])) * (1 - c["a5"] * abs(gamma))
        E = c["a6"] * fz + c["a7"]
        SH = c["a8"] * gamma + c["a9"] * fz + c["a10"]
        SV = c["a11"] * fz * gamma + c["a12"] * fz + c["a13"]

        # B = BCD / (C*D); guard the degenerate case where D ~ 0 (e.g. fz ~ 0)
        if abs(C * D) < 1e-9:
            return 0.0
        B = BCD / (C * D)

        phi = alpha + SH
        return (D * math.sin(C * math.atan(B * phi - E * (B * phi - math.atan(B * phi)))) + SV) * 0.7 # Scaling factor


    @staticmethod
    def _lat_mf52_model(params, alpha, gamma, fz, fz0):
        """Pure MF5.2 lateral math against an explicit parameter vector.
        alpha, gamma in radians. Used internally by the fitter."""
        (pCy1, pDy1, pDy2, pDy3,
        pEy1, pEy2, pEy3, pEy4,
        pKy1, pKy2, pKy3,
        pHy1, pHy2, pHy3,
        pVy1, pVy2, pVy3, pVy4) = params

        dfz = (fz - fz0) / fz0

        Cy = pCy1
        muy = (pDy1 + pDy2 * dfz) * (1 - pDy3 * gamma**2)
        Dy = muy * fz

        Ky = (pKy1 * fz0 * np.sin(2 * np.arctan(fz / (pKy2 * fz0)))
            * (1 - pKy3 * np.abs(gamma)))

        denom = Cy * Dy
        By = np.divide(Ky, denom, out=np.zeros_like(np.asarray(denom, dtype=float)),
                    where=np.abs(denom) > 1e-9)

        SHy = (pHy1 + pHy2 * dfz) + pHy3 * gamma
        SVy = fz * ((pVy1 + pVy2 * dfz) + (pVy3 + pVy4 * dfz) * gamma)

        alpha_y = alpha + SHy

        # sign(alpha_y) to match the standard mf5.2 / .tir definition
        Ey = (pEy1 + pEy2 * dfz) * (1 - (pEy3 + pEy4 * gamma) * np.sign(alpha_y))
        Ey = np.minimum(Ey, 1.0)

        Bx = By * alpha_y
        return Dy * np.sin(Cy * np.arctan(Bx - Ey * (Bx - np.arctan(Bx)))) + SVy


    def _eval_lat_mf52(self, alpha, gamma, fz, p):
        # alpha, gamma in deg, fz in N (positive), p in kPa -> unscaled fy in N
        p_fit = min(self.lat_mf52_coeffs, key=lambda k: abs(k - p))
        c = self.lat_mf52_coeffs[p_fit]
        params = [c[name] for name in LAT_MF52_COEFF_NAMES]
        return self._lat_mf52_model(params, np.deg2rad(alpha), np.deg2rad(gamma), fz, c["fz0"])


    def calculate_lat_force_mf52(self, alpha, gamma, fz, p):
        # uses the fit at the closest tested pressure
        if np.any(np.asarray(fz) < 0):
            raise ValueError(f"fz must be a positive load magnitude, got {fz}")

        fy = self._eval_lat_mf52(alpha, gamma, fz, p) * FY_SCALING_FACTOR
        return float(fy) if np.ndim(fy) == 0 else fy


    def calculate_lat_force_mf4(self, alpha, gamma, fz):
        c = self.lat_mf4_coeffs[(fz, gamma)]

        B = c["B"] 
        C = c["C"]
        D = c["D"]
        E_pos = c["E_pos"] 
        E_neg = c["E_neg"]
        SH = c["SH"]
        SV = c["SV"]

        c = self.lat_mf4_coeffs

        E = np.where(alpha >= 0, E_pos, E_neg)
        phi = alpha + SH
        return D * np.sin(C * np.arctan(B * phi - E * (B * phi - np.arctan(B * phi)))) + SV


    def calculate_lat_force_surface(self, alpha, gamma, fz, p):
        query_point = np.array([alpha, fz, gamma, p])

        # Negative value incorporated so that Ay sign convention matches right turn
        result = self.lat_surface(query_point)[0] * FY_SCALING_FACTOR
        return result


    # ---------- Loading Coefficients ----------
    
    def load_lat_bnp89_coeffs(self) -> None:
        """Load fitted Pacejka BNP 1989 coefficients from a .mat file produced
        by hoosier_r20_tire_model_lat.m (the `tireParams` struct).

        Expects `tireParams.coeffs` to hold a0..a13 (14 values) and
        `tireParams.rms_error_N` for fit quality. Populates self.coeffs as a
        name -> value dict for easy access (e.g. self.coeffs["a3"]).
        """
        if self.coeffs_path is None or not self.coeffs_path.exists():
            raise FileNotFoundError(f"Tire coefficient file not found: {self.coeffs_path}")

        mat = loadmat(self.coeffs_path, struct_as_record=False, squeeze_me=True)
        self.lat_bnp89_coeffs = dict(zip(mat["coeff_names"], mat["coeff_values"]))


    def load_lat_mf52_coeffs(self):
        if self.mf52_coeffs_path is None or not self.mf52_coeffs_path.exists():
            raise FileNotFoundError(f"Tire coefficient file not found: {self.mf52_coeffs_path}")

        mat = loadmat(self.mf52_coeffs_path)
        names = [str(n).strip() for n in mat["coeff_names"]]

        self.lat_mf52_coeffs = dict()
        for p, fz0, rms, values in zip(mat["pressures"].ravel(), mat["fz0"].ravel(), mat["rms"].ravel(), np.atleast_2d(mat["coeff_values"])):
            coeffs = dict(zip(names, values.astype(float)))
            coeffs["fz0"] = float(fz0)
            coeffs["rms"] = float(rms)
            self.lat_mf52_coeffs[float(p)] = coeffs


    def save_lat_mf52_coeffs(self):
        pressures = sorted(self.lat_mf52_coeffs)
        savemat(self.mf52_coeffs_path, {
            "coeff_names": LAT_MF52_COEFF_NAMES,
            "pressures": np.array(pressures),
            "fz0": np.array([self.lat_mf52_coeffs[p]["fz0"] for p in pressures]),
            "rms": np.array([self.lat_mf52_coeffs[p]["rms"] for p in pressures]),
            "coeff_values": np.array([[self.lat_mf52_coeffs[p][n] for n in LAT_MF52_COEFF_NAMES] for p in pressures]),
        })


    def print_lat_mf52_coeffs(self):
        # .tir style names, alpha/gamma in rad, fz in N, tested (sae) sign convention, lmuy etc. = 1
        for p in sorted(self.lat_mf52_coeffs):
            c = self.lat_mf52_coeffs[p]
            print(f"\n[{self.run_id}] mf5.2 lateral coeffs @ P = {p:g} kPa ({p / 6.89476:.0f} psi)")
            print(f"rms error = {c['rms']:.1f} N")
            print(f"FNOMIN = {c['fz0']:.1f}")
            for name in LAT_MF52_COEFF_NAMES:
                print(f"{name.upper():<6} = {c[name]: .6e}")

    # ---------- For Data Management ----------

    def load_data(self):
        """
        Load and filter all matching TTC cornering runs from TYRE_DATA_PATH.
        Returns a dict of 1D numpy arrays: SA [deg], FY [N], FZ [N, positive], IA [deg].
        """

        def _mat_str(val):
            """Coerce a loadmat string field (char array / object array) to a plain str."""
            arr = np.asarray(val)
            return str(arr.item()).strip() if arr.size == 1 else str(arr).strip()

        run_data = dict()

        # Step 2: Iterating through all relevant files
        # Step 3: Proceed with relevant tyres
        info = loadmat(self.data_path, variable_names=["tireid"])
        tire_str = _mat_str(info["tireid"])
        if not all(sub in tire_str for sub in self.tire_id_substrings):
            return

        d = loadmat(self.data_path)

        for key in d.keys():
            if key in CHANNELS:
                sign_factor = -1 if key == "FZ" else 1
                run_data[key] = np.asarray(d[key]).squeeze() * sign_factor

        self.run_data = run_data
        return run_data


    def process_data(self):

        def _mask_fz_filter(run_data, recovery_thresh=1000.0):
            """
            Returns a boolean keep-mask that drops every sample from the first
            FZ < 0 point onward, until FZ climbs back above recovery_thresh.
            Handles multiple dropout regions in the same run.
            """
            keep = np.ones(np.asarray(run_data["FZ"]).shape, dtype=bool)
            valid = True

            for idx in range(len(run_data["FZ"])):
                if run_data["FZ"][idx] < 0:
                    valid = False
                if abs(run_data["SA"][idx]) - 0 > 0.1 and not valid:
                    valid = True
                keep[idx] = valid

            return keep

        def _avg_sa_hysteresis(run_data, sa_bin=0.2, gap=0.5, min_rate=1.0, k=5):
            # per sweep, average the +dsa and -dsa passes at the same sa (cancels lag, keeps camber/offset asymmetry)
            et = np.asarray(run_data["ET"])
            breaks = np.r_[0, np.where(np.diff(et) > gap)[0] + 1, len(et)]
            out = {ch: [] for ch in run_data}

            for s, e in zip(breaks[:-1], breaks[1:]):
                n = e - s
                if n < 2 * k + 1:
                    continue

                seg = {ch: np.asarray(run_data[ch])[s:e] for ch in run_data}

                # smoothed sweep rate, deg/s
                i = np.arange(n)
                lo, hi = np.clip(i - k, 0, n - 1), np.clip(i + k, 0, n - 1)
                rate = (seg["SA"][hi] - seg["SA"][lo]) / (seg["ET"][hi] - seg["ET"][lo])

                key = np.round(seg["SA"] / sa_bin).astype(int)
                key -= key.min()
                n_bins = key.max() + 1

                # drop dwells / turnarounds, bin each direction separately
                passes = []
                for m in (rate > min_rate, rate < -min_rate):
                    cnt = np.bincount(key[m], minlength=n_bins)
                    avg = {ch: np.bincount(key[m], weights=seg[ch][m], minlength=n_bins) / np.maximum(cnt, 1) for ch in seg}
                    passes.append((cnt, avg))

                both = (passes[0][0] > 0) & (passes[1][0] > 0)
                for ch in seg:
                    out[ch].append(0.5 * (passes[0][1][ch][both] + passes[1][1][ch][both]))

            result = {ch: np.concatenate(v) for ch, v in out.items()}

            sort_idx = np.argsort(result["ET"])
            return {ch: v[sort_idx] for ch, v in result.items()}

        if self.run_id == "B2356raw8":
            # Remove point where tires are warming up
            temperature_filter = (self.run_data["ET"] > 250) & (self.run_data["ET"] < 2000)

            for ch in self.run_data:
                self.run_data[ch] = self.run_data[ch][temperature_filter]
        elif self.run_id == "B2356raw9":
            # Remove point where tires are warming up
            pressure_filter = (self.run_data["ET"] > 500) | (self.run_data["P"] < 60)

            for ch in self.run_data:
                self.run_data[ch] = self.run_data[ch][pressure_filter]

        # Remove in-between tests (hella noise)
        fz_filter = _mask_fz_filter(self.run_data)
        for ch in self.run_data:
            self.run_data[ch] = self.run_data[ch][fz_filter]

        # Remove hysteresis
        self.run_data = _avg_sa_hysteresis(self.run_data)


    # ---------- For Plotting ----------

    def plot_data_over_time(self, category, title):
        channels = ["SA", "FY", "FZ", "IA", "TSTC", "P", "V"]

        elapsed_time = self.run_data["ET"]
        xlabel = "Elapsed time [s]"

        available = [ch for ch in channels if ch in self.run_data]
        if not available:
            return

        fig, axes = plt.subplots(
            len(available), 1, figsize=(20, 6 * len(available)), sharex=True
        )
        if len(available) == 1:
            axes = [axes]

        for ax, ch in zip(axes, available):
            y = np.asarray(self.run_data[ch]).squeeze()

            if y.shape != elapsed_time.shape:
                ax.text(
                    0.5, 0.5, f"{ch}: shape mismatch ({y.shape} vs {elapsed_time.shape})",
                    transform=ax.transAxes, ha="center", va="center",
                )
                continue

            ax.plot(elapsed_time, y, linewidth=0.6)
            ax.set_ylabel(ch)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(xlabel)
        fig.suptitle(self.run_id)
        fig.tight_layout()
        fig.savefig(Path(Path.cwd() / f"figures/tyres/{category}/[{self.run_id}] {title}.png").expanduser().resolve(), dpi=150)


    def plot_fy_over_sa_per_fz(self, category, title, fz_buckets, ia, p):        
        fig, ax = plt.subplots(figsize=(9, 6))

        cmap = plt.get_cmap("jet", len(fz_buckets))
        for j, curr_fz in enumerate(fz_buckets):
            idx = (np.abs(self.run_data["FZ"] - curr_fz) < FZ_TOL) & (np.abs(self.run_data["IA"] - ia) < IA_TOL) & (np.abs(self.run_data["P"] - p) < P_TOL)

            sa_data = self.run_data["SA"][idx]
            fy_data = self.run_data["FY"][idx]

            if len(sa_data) > 0:
                ax.scatter(sa_data, fy_data, s=5, color=cmap(j), alpha=0.10)

            fy_pred = self._eval_lat_mf52(SA_RANGE, ia, curr_fz, p)
            ax.plot(SA_RANGE, fy_pred, color=cmap(j), linewidth=2, label=f"F_Z = {curr_fz:g} N")

        ax.axhline(0, linestyle="--", color="k", alpha=0.25)
        ax.axvline(0, linestyle="--", color="k", alpha=0.25)
        ax.set_xlabel("slip angle  α  [deg]")
        ax.set_ylabel("lateral force  F_Y  [N]")
        ax.set_title(f"Fy VS SA @ IA = {ia}, P = {p}")
        ax.legend(loc="best")
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(Path(Path.cwd() / f"figures/tyres/{category}/[{self.run_id}] {title} @ IA = {ia}, P = {p}.png").expanduser().resolve())


    def plot_fy_over_sa_per_ia(self, category, title, fz, ia_buckets, p):

        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.get_cmap("jet", len(ia_buckets))

        for j, curr_ia in enumerate(ia_buckets):
            idx = (np.abs(self.run_data["FZ"] - fz) < FZ_TOL) & (np.abs(self.run_data["IA"] - curr_ia) < IA_TOL) & (np.abs(self.run_data["P"] - p) < P_TOL)

            sa_data = self.run_data["SA"][idx]
            fy_data = self.run_data["FY"][idx]

            if len(sa_data) > 0:
                ax.scatter(sa_data, fy_data, s=5, color=cmap(j), alpha=0.50)

            fy_pred = self._eval_lat_mf52(SA_RANGE, curr_ia, fz, p)
            ax.plot(SA_RANGE, fy_pred, color=cmap(j), linewidth=2, label=f"IA = {curr_ia:g} N")

        ax.axhline(0, linestyle="--", color="k", alpha=0.25)
        ax.axvline(0, linestyle="--", color="k", alpha=0.25)
        ax.set_xlabel("slip angle  α  [deg]")
        ax.set_ylabel("lateral force  F_Y  [N]")
        ax.set_title(f"Fy VS SA @ FZ = {fz}, P = {p}")
        ax.legend(loc="best")
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(Path(Path.cwd() / f"figures/tyres/{category}/[{self.run_id}] {title} @ FZ = {fz}, P = {p}.png").expanduser().resolve())


    def plot_fy_over_sa_per_p(self, category, title, fz, ia, p_buckets):

        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.get_cmap("jet", len(p_buckets))

        for j, curr_p in enumerate(p_buckets):
            idx = (np.abs(self.run_data["FZ"] - fz) < FZ_TOL) & (np.abs(self.run_data["IA"] - ia) < IA_TOL) & (np.abs(self.run_data["P"] - curr_p) < P_TOL)

            sa_data = self.run_data["SA"][idx]
            fy_data = self.run_data["FY"][idx]

            if len(sa_data) > 0:
                ax.scatter(sa_data, fy_data, s=5, color=cmap(j), alpha=0.50)

            fy_pred = self._eval_lat_mf52(SA_RANGE, ia, fz, curr_p)
            ax.plot(SA_RANGE, fy_pred, color=cmap(j), linewidth=2, label=f"P = {curr_p:g} N")

        ax.axhline(0, linestyle="--", color="k", alpha=0.25)
        ax.axvline(0, linestyle="--", color="k", alpha=0.25)
        ax.set_xlabel("slip angle  α  [deg]")
        ax.set_ylabel("lateral force  F_Y  [N]")
        ax.set_title(f"Fy VS SA @ FZ = {fz}, IA = {ia}")
        ax.legend(loc="best")
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(Path(Path.cwd() / f"figures/tyres/{category}/[{self.run_id}] {title} @ FZ = {fz}, IA = {ia}.png").expanduser().resolve())


    def plot_peak_mu_over_ia(self, category, title, fz_buckets, p_buckets, ia_max=4):
        # camber thrust is +fy for +ia in this data, so the +fy peak is the leaning-into-turn side
        sa = np.linspace(-12, 12, 1201)[:, None] # tested sa range only
        ia = np.linspace(0, ia_max, 41)[None, :]
        colors = [LOAD_RAMP[i] for i in np.linspace(0, len(LOAD_RAMP) - 1, len(fz_buckets)).round().astype(int)]

        fig, axes = plt.subplots(1, len(p_buckets), figsize=(6 * len(p_buckets), 5.5), sharey=True, squeeze=False)
        axes = axes[0]

        for ax, p in zip(axes, p_buckets):
            for j, fz in enumerate(fz_buckets):
                fy = self._eval_lat_mf52(sa, ia, fz, p)
                ax.plot(ia[0], fy.max(axis=0) / fz, color=colors[j], linewidth=2, label=f"F_Z = {fz:g} N")
                ax.plot(-ia[0], -fy.min(axis=0) / fz, color=colors[j], linewidth=2)

            ax.axvline(0, color="k", linewidth=0.8, alpha=0.25)
            ax.set_xlabel("camber [deg]  (+ leaning into turn, − leaning away)")
            ax.set_title(f"P = {p:.0f} kPa ({p / 6.89476:.0f} psi)")
            ax.grid(True, alpha=0.3)

        # measured peaks, one pair per sweep
        d = self.run_data
        breaks = np.r_[0, np.where(np.diff(d["ET"]) > 5)[0] + 1, len(d["ET"])]
        for s, e in zip(breaks[:-1], breaks[1:]):
            fz, ia_sweep, p = d["FZ"][s:e].mean(), d["IA"][s:e].mean(), d["P"][s:e].mean()
            j = np.argmin(np.abs(np.asarray(fz_buckets) - fz))
            k = np.argmin(np.abs(np.asarray(p_buckets) - p))
            if abs(fz_buckets[j] - fz) > FZ_TOL or abs(p_buckets[k] - p) > P_TOL:
                continue

            axes[k].scatter([ia_sweep, -ia_sweep], [d["FY"][s:e].max() / fz, -d["FY"][s:e].min() / fz],
                            s=40, color=colors[j], edgecolors="white", linewidths=1, zorder=3)

        axes[0].set_ylabel("peak μ_y  [F_Y,peak / F_Z]")
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color="gray", marker="o", linestyle="none", markeredgecolor="white"))
        labels.append("measured sweep peak")
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)

        fig.suptitle(f"[{self.run_id}] {title} (mf5.2 fit, unscaled)")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(Path(Path.cwd() / f"figures/tyres/{category}/[{self.run_id}] {title}.png").expanduser().resolve())



    # ---------- Fitting Coefficients ----------

    def fit_lat_mf52(self, pressures=P_BUCKETS):
        # fits all 18 mf5.2 lateral coeffs in one least squares, one set per tested pressure
        self.lat_mf52_coeffs = dict()

        for p in pressures:
            idx = np.abs(self.run_data["P"] - p) < P_TOL
            if idx.sum() < LAT_MF52_MIN_POINTS:
                continue

            alpha = np.deg2rad(self.run_data["SA"][idx])
            gamma = np.deg2rad(self.run_data["IA"][idx])
            fz = self.run_data["FZ"][idx]
            fy = self.run_data["FY"][idx]
            fz0 = float(np.round(np.median(fz), -1))

            result = least_squares(
                lambda x: self._lat_mf52_model(x, alpha, gamma, fz, fz0) - fy,
                LAT_MF52_X0, bounds=(LAT_MF52_LB, LAT_MF52_UB),
                x_scale="jac", loss="soft_l1", f_scale=50, max_nfev=20000)

            resid = self._lat_mf52_model(result.x, alpha, gamma, fz, fz0) - fy

            coeffs = dict(zip(LAT_MF52_COEFF_NAMES, result.x))
            coeffs["fz0"] = fz0
            coeffs["rms"] = float(np.sqrt(np.mean(resid**2)))
            self.lat_mf52_coeffs[float(p)] = coeffs

        if not self.lat_mf52_coeffs:
            raise ValueError("Not enough points to construct an mf5.2 fit")

        return self.lat_mf52_coeffs


    def fit_lat_spline(self):
        for fz in FZ_BUCKETS:
            for ia in IA_BUCKETS:
                for p in P_BUCKETS:
                    idx = (np.abs(self.run_data["FZ"] - fz) < FZ_TOL) & \
                        (np.abs(self.run_data["IA"] - ia) < IA_TOL) & \
                        (np.abs(self.run_data["P"] - p) < P_TOL)

                    sa_data = self.run_data["SA"][idx]
                    fy_data = self.run_data["FY"][idx]
                    if len(sa_data) < 15:
                        continue

                    sa_clean, inverse_indices = np.unique(sa_data, return_inverse=True)
                    fy_clean = np.bincount(inverse_indices, weights=fy_data) / np.bincount(inverse_indices)

                    w = np.ones_like(sa_clean)
                    w[(np.abs(sa_clean) >= 7.5) & (np.abs(sa_clean) < 10)] = 0.5
                    w[np.abs(sa_clean) >= 10] = 0.25

                    self.lat_splines[(fz, ia, p)] = make_smoothing_spline(sa_clean, fy_clean, w=w, lam=35)

        # --- Build the actual tensor grid RegularGridInterpolator expects ---
        sa_axis = np.unique(SA_RANGE)
        fz_axis = np.unique(FZ_BUCKETS)
        ia_axis = np.unique(IA_BUCKETS)
        p_axis  = np.unique(P_BUCKETS)

        fy_grid = np.full((len(sa_axis), len(fz_axis), len(ia_axis), len(p_axis)), np.nan)

        for i, fz in enumerate(fz_axis):
            for j, ia in enumerate(ia_axis):
                for k, p in enumerate(p_axis):
                    spline = self.lat_splines.get((fz, ia, p))
                    if spline is not None:
                        fy_grid[:, i, j, k] = spline(sa_axis)

        if np.isnan(fy_grid).all():
            raise ValueError("Not enough points to construct a spline fit")

        self.lat_surface = RegularGridInterpolator(
            (sa_axis, fz_axis, ia_axis, p_axis),
            fy_grid,
            bounds_error=False,
            fill_value=None,
        )
        return self.lat_surface


    @DeprecationWarning
    def fit_lat_mf4(self, alpha, fy, ia, fz):
        """Fit a Pacejka curve with SEPARATE curvature E for alpha>=0 vs alpha<0,
        since MF5.2's Ey has an explicit sign(alpha) split. Returns B, C, D,
        E_pos, E_neg, SH, SV.
        """

        sign = np.where(alpha >= 0, 1.0, -1.0)

        def model(p, alpha, sign):
            B, C, D, E_pos, E_neg, SH, SV = p
            E = np.where(sign >= 0, E_pos, E_neg)
            phi = alpha + SH
            return D * np.sin(C * np.arctan(B * phi - E * (B * phi - np.arctan(B * phi)))) + SV

        def resid(p):
            return model(p, alpha, sign) - fy

        D0 = np.max(np.abs(fy))
        near0 = np.abs(alpha) < 1.0
        slope0 = np.polyfit(alpha[near0], fy[near0], 1)[0] if near0.sum() > 3 else D0
        B0 = abs(slope0) / (1.3 * D0) if D0 > 1e-6 else 10.0

        x0 = [B0, 1.3, D0 * np.sign(slope0), -0.5, -0.5, 0.0, 0.0]
        lower = [0.1, 0.8, -D0 * 2, -3.0, -3.0, -0.3, -D0 * 0.3]
        upper = [200.0, 2.2, D0 * 2, 1.0, 1.0, 0.3, D0 * 0.3]

        result = least_squares(resid, x0, bounds=(lower, upper), x_scale="jac",
                                xtol=1e-12, ftol=1e-12, max_nfev=5000)
        B, C, D, E_pos, E_neg, SH, SV = result.x
        
        self.lat_mf4_coeffs[(fz, ia)] = dict(B=B, C=C, D=D, E_pos=E_pos, E_neg=E_neg, SH=SH, SV=SV, success=result.success)
    


if __name__ == "__main__":
    tire = Tire()

    tire.setup()

    result = tire.calculate_lat_force_mf52(5, 0.5, 1000, 68.9476)

    tire.plot_fy_over_sa_per_fz("lateral", "Fy vs SA per Fz", [180, 400, 630, 1000], 4, 82.7371)
    tire.plot_fy_over_sa_per_ia("lateral", "Fy vs SA per IA", 1000, [0, 1], 82.7371)
    tire.plot_fy_over_sa_per_p("lateral", "Fy VS SA per P", 1000, 1.5, P_BUCKETS)
    tire.plot_peak_mu_over_ia("lateral", "Peak mu vs IA per Fz", [220, 440, 665, 890, 1110], sorted(tire.lat_mf52_coeffs))
