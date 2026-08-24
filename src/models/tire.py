from pathlib import Path

import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares
from scipy.interpolate import make_splrep, make_smoothing_spline, RectBivariateSpline, RegularGridInterpolator
import numpy as np
import math

DEFAULT_RUN_ID = "B2356run8"
DEFAULT_DATA_PATH = Path(Path.cwd() / f"data/tyres/RunData_Cornering_Matlab_SI_Round9/{DEFAULT_RUN_ID}.mat").expanduser().resolve()
DEFAULT_COEFFS_PATH = Path(Path.cwd() / f"data/coeffs/lateral/[{DEFAULT_RUN_ID}] hoosier_r20_lat_mf52_coeffs.mat").expanduser().resolve()

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
    "pVy1", "pVy2", "pVy3", "pVy4", "fz0"]
LONG_MF52_COEFF_NAMES = []

SA_RANGE = np.linspace(-14, 14, 300)
FZ_BUCKETS = [230, 400, 630, 900, 1100]
IA_BUCKETS = [0, 2, 4]
P_BUCKETS = [41.3685, 55.1581, 68.9476, 82.7371] # in kPa (when hot) - (6, 8, 10, 12) psi choices

FZ_TOL = 50
IA_TOL = 0.5
P_TOL = 5

class Tire:
    def __init__(self, run_id=DEFAULT_RUN_ID, data_path=DEFAULT_DATA_PATH, coeffs_path=DEFAULT_COEFFS_PATH):

        self.run_id = run_id
        self.data_path = data_path
        self.coeffs_path = coeffs_path
        self.tire_id_substrings = ["43075", "16x7.5", "8 inch rim"]

        self.lat_mf4_coeffs = dict()
        self.lat_mf52_coeffs = dict()
        self.lat_bnp89_coeffs = dict()

        self.lat_splines = dict()
        self.lat_surface = None


    # ---------- Setup ----------
    def setup(self):
        # Load the data
        self.load_data()

        # Process the data
        self.process_data()

        # Perform fitting
        self.fit_lat_spline()


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
    def _lat_mf52_model(params, alpha, gamma, fz):
        """Pure MF5.2 lateral math against an explicit parameter vector.
        alpha, gamma in radians. Used internally by the fitter."""
        (pCy1, pDy1, pDy2, pDy3,
        pEy1, pEy2, pEy3, pEy4,
        pKy1, pKy2, pKy3,
        pHy1, pHy2, pHy3,
        pVy1, pVy2, pVy3, pVy4, fz0) = params

        dfz = (fz - fz0) / fz0

        Cy = pCy1
        muy = (pDy1 + pDy2 * dfz) * (1 - pDy3 * gamma**2)
        Dy = muy * fz

        Ey = (pEy1 + pEy2 * dfz) * (1 - (pEy3 + pEy4 * gamma) * np.sign(alpha))
        Ey = np.minimum(Ey, 1.0)

        Ky = (pKy1 * fz0 * np.sin(2 * np.arctan(fz / (pKy2 * fz0)))
            * (1 - pKy3 * np.abs(gamma)))

        denom = Cy * Dy
        By = np.divide(Ky, denom, out=np.zeros_like(np.asarray(denom, dtype=float)),
                    where=np.abs(denom) > 1e-9)

        SHy = (pHy1 + pHy2 * dfz) + pHy3 * gamma
        SVy = fz * ((pVy1 + pVy2 * dfz) + (pVy3 + pVy4 * dfz) * gamma)

        alpha_y = alpha + SHy
        Bx = By * alpha_y
        return Dy * np.sin(Cy * np.arctan(Bx - Ey * (Bx - np.arctan(Bx)))) + SVy


    def calculate_lat_force_mf52(self, alpha: float, gamma: float, fz: float) -> float:
        """Public API: evaluate MF5.2 lateral force using previously fitted
        coefficients for `run_id`. slip_angle/camber_angle in degrees, matching
        the BNP-model convention used elsewhere in this class."""

        if fz < 0:
            raise ValueError(f"fz must be a positive load magnitude, got {fz}")

        c = self.lat_mf52_coeffs

        params = [c[name] for name in LAT_MF52_COEFF_NAMES]
        return float(self._lat_mf52_model(params, alpha, gamma, fz))


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
        return self.lat_surface(query_point)[0] * FY_SCALING_FACTOR


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
        """Load fitted MF5.2 coefficients from a .mat file produced
        by hoosier_r20_tire_model_lat.m (the `tireParams` struct).
        """
        if self.coeffs_path is None or not self.coeffs_path.exists():
            raise FileNotFoundError(f"Tire coefficient file not found: {self.coeffs_path}")

        mat = loadmat(self.coeffs_path, struct_as_record=False, squeeze_me=True)

        default_coeffs = dict(zip(mat["coeff_names"], mat["coeff_values"]))
        default_coeffs["fz0"] = mat["fz0"]

        self.lat_mf52_coeffs = default_coeffs

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

        def _avg_sa_hysteresis(run_data, sa_match_tol=0.05,
                           fz_match_tol=25.0, ia_match_tol=0.1):
            sa = np.asarray(run_data["SA"])
            fy = np.asarray(run_data["FY"])
            fz = np.asarray(run_data["FZ"])
            ia = np.asarray(run_data["IA"])
            n = len(sa)

            used = np.zeros(n, dtype=bool)
            out = {ch: [] for ch in run_data}

            for idx in range(len(fz)):
                if used[idx] or sa[idx] < 0:
                    continue

                mask = (~used) & (np.abs(fz - fz[idx]) < fz_match_tol) & (np.abs(ia - ia[idx]) < ia_match_tol)
                mask[idx] = False  # don't match to self
                candidate_idxs = np.where(mask)[0]

                if len(candidate_idxs) == 0:
                    continue

                mirrored_idx = candidate_idxs[np.argmin(np.abs(sa[candidate_idxs] + sa[idx]))]
                if abs(sa[mirrored_idx] + sa[idx]) > sa_match_tol:
                    continue

                used[idx] = used[mirrored_idx] = True
                fy_avg = (abs(fy[idx]) + abs(fy[mirrored_idx])) / 2.0

                for i, fy_val in ((idx, -fy_avg), (mirrored_idx, fy_avg)):
                    for ch in run_data:
                        out[ch].append(np.asarray(run_data[ch])[i] if ch != "FY" else fy_val)

            result = {ch: np.asarray(v) for ch, v in out.items()}

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

            query_points = np.transpose(
                np.array([
                    SA_RANGE,
                    curr_fz * np.ones_like(SA_RANGE),
                    ia * np.ones_like(SA_RANGE),
                    p * np.ones_like(SA_RANGE) 
                ])
            )
            fy_pred = self.lat_surface(query_points)
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

            query_points = np.transpose(
                np.array([
                    SA_RANGE,
                    fz * np.ones_like(SA_RANGE),
                    curr_ia * np.ones_like(SA_RANGE),
                    p * np.ones_like(SA_RANGE) 
                ])
            )
            fy_pred = self.lat_surface(query_points)
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

            query_points = np.transpose(
                np.array([
                    SA_RANGE,
                    fz * np.ones_like(SA_RANGE),
                    ia * np.ones_like(SA_RANGE),
                    curr_p * np.ones_like(SA_RANGE) 
                ])
            )
            fy_pred = self.lat_surface(query_points)
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



    # ---------- Fitting Coefficients ----------

    @DeprecationWarning
    def fit_lat_mf52(self, fz0=None, verbose=True):
        """
        Fit MF 5.2 pure-slip lateral coefficients (Table 10, TNO MF-Tyre/MF-Swift
        convention) via bounded nonlinear least squares.

        Model:
            Cy  = pCy1
            muy = (pDy1 + pDy2*dfz) * (1 - pDy3*gamma^2)
            Dy  = muy * Fz
            Ey  = (pEy1 + pEy2*dfz) * (1 - (pEy3 + pEy4*gamma)*sign(alpha))   [Ey <= 1]
            Ky  = pKy1*Fz0 * sin(2*arctan(Fz / (pKy2*Fz0))) * (1 - pKy3*|gamma|)
            By  = Ky / (Cy*Dy)
            SHy = (pHy1 + pHy2*dfz) + pHy3*gamma
            SVy = Fz * ((pVy1 + pVy2*dfz) + (pVy3 + pVy4*dfz)*gamma)
            alpha_y = alpha + SHy
            Fy0 = Dy*sin(Cy*arctan(By*alpha_y - Ey*(By*alpha_y - arctan(By*alpha_y)))) + SVy

        All lambda_* scaling factors are fixed at 1.0 (not fit). alpha, gamma
        are converted from degrees (as stored in run_data) to radians internally.

        Args:
            fz0: nominal load, N. Defaults to the median FZ in the dataset.

        Returns:
            dict of fitted pCy1, pDy1-3, pEy1-4, pKy1-3, pHy1-3, pVy1-4
        """

        spline_results = self.fit_lat_spline()
        first_stage_coeffs = dict()


        for (fz, ia) in spline_results:
            first_stage_coeffs[(fz, ia)] = self.fit_lat_mf4(spline_results[(fz, ia)][0], ia, fz, spline_results[(fz, ia)][1])


        base = {k: v for k, v in first_stage_coeffs.items() if k[1] == 0.0}
        fz_arr = np.array([k[0] for k in base])
        D_arr = np.array([v["D"] for v in base.values()])
        B_arr = np.array([v["B"] for v in base.values()])
        C_arr = np.array([v["C"] for v in base.values()])
        Ky_arr = B_arr * C_arr * D_arr
        SH_arr = np.array([v["SH"] for v in base.values()])
        SV_arr = np.array([v["SV"] for v in base.values()])
        Ey_pos_arr = np.array([v["E_pos"] for v in base.values()])
        Ey_neg_arr = np.array([v["E_neg"] for v in base.values()])

        # pCy1: just average the per-bucket C (should be fairly load-independent)
        pCy1 = float(np.mean(C_arr))

        if fz0 is None:
            fz0 = float(np.median(fz_arr))
        dfz_arr = (fz_arr - fz0) / fz0

        # pDy1, pDy2: muy = pDy1 + pDy2*dfz  (fit at gamma=0, so pDy3 term drops out)
        muy_arr = D_arr / fz_arr
        pDy2, pDy1 = np.polyfit(dfz_arr, muy_arr, 1)  # returns [slope, intercept]

        # pKy1, pKy2: Ky = pKy1*Fz0*sin(2*atan(Fz/(pKy2*Fz0)))  -- nonlinear, small fit
        def ky_resid(p):
            pKy1, pKy2 = p
            return pKy1 * fz0 * np.sin(2 * np.arctan(fz_arr / (pKy2 * fz0))) - Ky_arr

        ky_guess = np.clip(Ky_arr[-1] / fz0, 0.15, 195.0)
        ky_fit = least_squares(ky_resid, x0=[ky_guess, 1.0], bounds=([0.1, 0.1], [200, 20]))
        pKy1, pKy2 = ky_fit.x

        # pHy1, pHy2: SH = pHy1 + pHy2*dfz
        pHy2, pHy1 = np.polyfit(dfz_arr, SH_arr, 1)

        # pVy1, pVy2: SV/Fz = pVy1 + pVy2*dfz
        pVy2, pVy1 = np.polyfit(dfz_arr, SV_arr / fz_arr, 1)

        # pEy1, pEy2: Ey = pEy1 + pEy2*dfz  (at gamma=0, sign(alpha) folded in separately if you want per-sign E)
        Ey_base_arr = (Ey_pos_arr + Ey_neg_arr) / 2.0
        pEy2, pEy1 = np.polyfit(dfz_arr, Ey_base_arr, 1)
        pEy3_gamma0 = float(np.mean((Ey_neg_arr - Ey_pos_arr) / (Ey_pos_arr + Ey_neg_arr + 1e-12)))

        # print(f"pCy1={pCy1:.4f}  pDy1={pDy1:.4f} pDy2={pDy2:.4f}  pKy1={pKy1:.4f} pKy2={pKy2:.4f}")
        # print(f"pHy1={pHy1:.4f} pHy2={pHy2:.4f}  pVy1={pVy1:.4f} pVy2={pVy2:.4f}  pEy1={pEy1:.4f} pEy2={pEy2:.4f}")

        pDy3_samples, pKy3_samples, pHy3_samples, ey_ratio_samples, gamma_for_ey = [], [], [], [], []
        pVy_diff_samples, pVy_dfz_samples = [], []

        for (fz, ia), v in first_stage_coeffs.items():
            if ia == 0.0:
                continue

            gamma = ia
            dfz = (fz - fz0) / fz0

            # --- pDy3: muy = (pDy1+pDy2*dfz)*(1 - pDy3*gamma^2)
            muy = v["D"] / fz
            base_muy = pDy1 + pDy2 * dfz
            if abs(base_muy) > 1e-6:
                pDy3_samples.append((1 - muy / base_muy) / gamma**2)

            # --- pKy3: Ky = Ky_base(Fz) * (1 - pKy3*|gamma|)
            Ky_meas = v["B"] * v["C"] * v["D"]
            Ky_base = pKy1 * fz0 * np.sin(2 * np.arctan(fz / (pKy2 * fz0)))
            if abs(Ky_base) > 1e-6:
                pKy3_samples.append((1 - Ky_meas / Ky_base) / abs(gamma))

            # --- pHy3: SH = (pHy1+pHy2*dfz) + pHy3*gamma
            base_SH = pHy1 + pHy2 * dfz
            pHy3_samples.append((v["SH"] - base_SH) / gamma)

            # --- pVy3, pVy4: SV/Fz = (pVy1+pVy2*dfz) + (pVy3+pVy4*dfz)*gamma
            base_SVfz = pVy1 + pVy2 * dfz
            slope_term = ((v["SV"] / fz) - base_SVfz) / gamma   # == pVy3 + pVy4*dfz
            pVy_diff_samples.append(slope_term)
            pVy_dfz_samples.append(dfz)

            # --- pEy3, pEy4: ratio = pEy3 + pEy4*gamma, where
            # ratio = (Ey_neg - Ey_pos) / (Ey_pos + Ey_neg)
            denom = v["E_pos"] + v["E_neg"]
            if abs(denom) > 1e-6:
                ey_ratio_samples.append((v["E_neg"] - v["E_pos"]) / denom)
                gamma_for_ey.append(gamma)

        pDy3 = float(np.median(pDy3_samples)) if pDy3_samples else 0.0
        pKy3 = float(np.median(pKy3_samples)) if pKy3_samples else 0.0
        pHy3 = float(np.median(pHy3_samples)) if pHy3_samples else 0.0

        pVy4, pVy3 = np.polyfit(pVy_dfz_samples, pVy_diff_samples, 1) if len(pVy_dfz_samples) > 1 else (0.0, 0.0)
        pEy4, pEy3 = np.polyfit(gamma_for_ey, ey_ratio_samples, 1) if len(gamma_for_ey) > 1 else (0.0, float(pEy3_gamma0))

        fitted = [pCy1, pDy1, pDy2, pDy3, pEy1, pEy2, pEy3, pEy4, pKy1, pKy2, pKy3, pHy1, pHy2, pHy3, pVy1, pVy2, pVy3, pVy4, fz0]
        self.lat_mf52_coeffs = dict(zip(LAT_MF52_COEFF_NAMES, fitted))

        savemat(self.coeffs_path, {
            "coeff_names": LAT_MF52_COEFF_NAMES,
            "coeff_values": fitted,
        })


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

    tire.load_data()
    tire.process_data()

    tire.fit_lat_spline()

    tire.plot_fy_over_sa_per_fz("lateral", "Fy vs SA per Fz", [180, 400, 630, 1000], 4, 82.7371)
    tire.plot_fy_over_sa_per_ia("lateral", "Fy vs SA per IA", 1000, [0, 1], 82.7371)
    tire.plot_fy_over_sa_per_p("lateral", "Fy VS SA per P", 1000, 1.5, P_BUCKETS)
