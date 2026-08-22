from pathlib import Path

import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.io import loadmat
import numpy as np
import math

"""
Pacejka BNP 1989 Magic Formula Structure
%
%   C   = a0                                       (shape factor)
%   D   = a1*Fz^2 + a2*Fz                          (peak force — load sensitivity)
%   BCD = a3*sin(2*atan(Fz/a4))*(1 - a5*|gamma|)   (cornering stiffness; negative in sae convention)
%   E   = a6*Fz + a7                               (curvature factor)
%   SH  = a8*gamma + a9*Fz + a10                   (horizontal shift — plysteer/camber)
%   SV  = a11*Fz*gamma + a12*Fz + a13              (vertical shift — conicity/camber thrust)
%
%   phi = alpha + SH
%   FY  = D*sin(C*atan(B*phi - E*(B*phi - atan(B*phi)))) + SV
%
% note: B = BCD / (C*D)
%       slip angle (alpha) and camber (gamma) in degrees, forces in newtons
"""

DEFAULT_DATA_PATH = Path(Path.cwd() / "data/tyres/RunData_Cornering_Matlab_SI_Round9").expanduser().resolve()
DEFAULT_COEFFS_PATH = Path(Path.cwd() / "data/coeffs/lateral/[B2356run9] hoosier_r20_tire_params_lat.mat").expanduser().resolve()
CHANNELS = set(["ET", "FX", "FY", "AMBTMP", "FZ", "IA", "MX", "MZ", "N", "P", "RE", "RL", "RST", "SA", "SL", "SR", "TSTC", "TSTI", "TSTO", "V"])

class Tire:

    def __init__(self, data_path=DEFAULT_DATA_PATH, coeffs_path=DEFAULT_COEFFS_PATH):
        # TODO: fill in real values -- placeholders below are 0.0 so the class
        # is at least constructible; using these for a real sim run will
        # silently zero out camber gain until they're set.
        self.data_path = data_path
        self.coeffs_path = coeffs_path
        self.tire_id_substrings = ["43075", "16x7.5", "8 inch rim"]

        self.COEFF_NAMES = (
            "a0", "a1", "a2", "a3", "a4", "a5", "a6",
            "a7", "a8", "a9", "a10", "a11", "a12", "a13",
        )
        self.load_coefficients()


        # ---------- For Model Invocation ----------

        def calculate_lat_force(self, vertical_load: float, slip_angle: float,  camber_angle: float) -> float:
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
            if self.coeffs is None:
                raise RuntimeError("Tire coefficients not loaded -- call load_coefficients() first")
            if vertical_load < 0:
                raise ValueError(f"fz must be a positive load magnitude, got {vertical_load}")
    
            c = self.coeffs  # shorthand
    
            C = c["a0"]
            D = c["a1"] * vertical_load**2 + c["a2"] * vertical_load
            BCD = c["a3"] * math.sin(2 * math.atan(vertical_load / c["a4"])) * (1 - c["a5"] * abs(camber_angle))
            E = c["a6"] * vertical_load + c["a7"]
            SH = c["a8"] * camber_angle + c["a9"] * vertical_load + c["a10"]
            SV = c["a11"] * vertical_load * camber_angle + c["a12"] * vertical_load + c["a13"]
    
            # B = BCD / (C*D); guard the degenerate case where D ~ 0 (e.g. fz ~ 0)
            if abs(C * D) < 1e-9:
                return 0.0
            B = BCD / (C * D)
    
            phi = slip_angle + SH
            return (D * math.sin(C * math.atan(B * phi - E * (B * phi - math.atan(B * phi)))) + SV) * 0.7 # Scaling factor
    
    
    def load_coefficients(self) -> None:
        """Load fitted Pacejka BNP 1989 coefficients from a .mat file produced
        by hoosier_r20_tire_model_lat.m (the `tireParams` struct).

        Expects `tireParams.coeffs` to hold a0..a13 (14 values) and
        `tireParams.rms_error_N` for fit quality. Populates self.coeffs as a
        name -> value dict for easy access (e.g. self.coeffs["a3"]).
        """
        if self.coeffs_path is None or not self.coeffs_path.exists():
            raise FileNotFoundError(f"Tire coefficient file not found: {self.coeffs_path}")

        mat = loadmat(self.coeffs_path, struct_as_record=False, squeeze_me=True)
        raw_coeffs = mat["coeffs"]

        if len(raw_coeffs) != len(self.COEFF_NAMES):
            raise ValueError(
                f"Expected {len(self.COEFF_NAMES)} coefficients "
                f"(a0-a{len(self.COEFF_NAMES) - 1}), got {len(raw_coeffs)} "
                f"from {self.coeffs_path}. Check the MATLAB script's coeff_names still "
                f"matches the fit vector p_fit."
            )
        self.coeffs = dict(zip(self.COEFF_NAMES, raw_coeffs.tolist()))

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

        data_dict = defaultdict(dict)

        # Step 1: Read the relevant files
        run_files = self.data_path.rglob("*")

        # Step 2: Iterating through all relevant files
        for fpath in run_files:

            # Step 3: Proceed with relevant tyres
            info = loadmat(fpath, variable_names=["tireid"])
            tire_str = _mat_str(info["tireid"])
            if not all(sub in tire_str for sub in self.tire_id_substrings):
                continue

            d = loadmat(fpath)

            for key in d.keys():
                if key in CHANNELS:
                    sign_factor = -1 if key == "FZ" else 1
                    data_dict[Path(fpath).stem][key] = np.asarray(d[key]).squeeze() * sign_factor
        self.data_dict = data_dict
        return data_dict


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

        for run_id, run_data in self.data_dict.items():
            if run_id == "B2356raw8":
                # Remove point where tires are warming up
                temperature_filter = (run_data["ET"] > 250) & (run_data["ET"] < 2000)

                for ch in run_data:
                    run_data[ch] = run_data[ch][temperature_filter]
            elif run_id == "B2356raw9":
                # Remove point where tires are warming up
                pressure_filter = (run_data["ET"] > 500) | (run_data["P"] < 60)

                for ch in run_data:
                    run_data[ch] = run_data[ch][pressure_filter]

            # Remove in-between tests (hella noise)
            fz_filter = _mask_fz_filter(run_data)
            for ch in run_data:
                run_data[ch] = run_data[ch][fz_filter]

        # Remove hysteresis
        for run_id in self.data_dict:
            self.data_dict[run_id] = _avg_sa_hysteresis(self.data_dict[run_id])


    # ---------- For Plotting ----------

    def plot_data_over_time(self, category, title):
        channels = ["SA", "FY", "FZ", "IA", "TSTC", "P"]

        for run_id, run_data in self.data_dict.items():
            elapsed_time = run_data["ET"]
            xlabel = "Elapsed time [s]"


            available = [ch for ch in channels if ch in run_data]
            if not available:
                continue

            fig, axes = plt.subplots(
                len(available), 1, figsize=(20, 6 * len(available)), sharex=True
            )
            if len(available) == 1:
                axes = [axes]

            for ax, ch in zip(axes, available):
                y = np.asarray(run_data[ch]).squeeze()

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
            fig.suptitle(run_id)
            fig.tight_layout()
            fig.savefig(Path(Path.cwd() / f"figures/tyres/{category}/[{run_id}] {title}.png").expanduser().resolve(), dpi=150)


    def plot_fy_over_sa(self, category, title):
        fz_buckets = [230, 630, 900, 1100]

        # sa_vec = np.linspace(-14, 14, 300)
        fz_tol = 27.5
        cmap = plt.get_cmap("jet", len(fz_buckets))

        fig, ax = plt.subplots(figsize=(9, 6))
        for run_id, run_data in self.data_dict.items():
            for k, fz in enumerate(fz_buckets):
                idx = (np.abs(run_data["FZ"] - fz) < fz_tol) & (np.abs(run_data["IA"]) < 0.5)
                # if idx.sum() < 30:
                #     print(f"skipping F_Z = {fz} N (only {idx.sum()} points nearby)")
                #     continue

                ax.scatter(run_data["SA"][idx], run_data["FY"][idx], s=5, color=cmap(k), alpha=0.20)

                # fy_pred = pacejka_lat_force(p_fit, sa_vec, fz * np.ones_like(sa_vec), 0.0)
                # ax.plot(sa_vec, fy_pred, color=cmap(k), linewidth=2, label=f"F_Z = {fz:g} N")

            ax.axhline(0, linestyle="--", color="k", alpha=0.25)
            ax.axvline(0, linestyle="--", color="k", alpha=0.25)
            ax.set_xlabel("slip angle  α  [deg]")
            ax.set_ylabel("lateral force  F_Y  [N]")
            ax.set_title("lateral force vs slip angle (lines = model, dots = measured)")
            ax.legend(loc="best")
            ax.grid(True)

            fig.tight_layout()
            fig.savefig(Path(Path.cwd() / f"figures/tyres/{category}/[{run_id}] {title}.png").expanduser().resolve())


    # Fits MF 6.1 coefficients
    def fit_coefficients(self):

        # Step 1: Establish names for all coefficients
        # Step 2: Identify UB and LW for all coefficients
        # Step 3: Fitting process?

        return


if __name__ == "__main__":
    tire = Tire()

    tire.load_data()
    tire.plot_data_over_time("unprocessed", "Unprocessed Tire Data")

    tire.process_data()

    tire.plot_data_over_time("processed", "Processed Tire Data")
    tire.plot_fy_over_sa("lateral", "Fy vs SA")