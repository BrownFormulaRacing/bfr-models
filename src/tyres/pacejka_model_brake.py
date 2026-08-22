import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares
from pathlib import Path
from .pacejka_model_helper import load_data

TYRE_DATA_PATH = Path("../../data/tyres/RunData_DriveBrake_Matlab_SI_Round9").expanduser().resolve()

# tire filter — Hoosier 43100 18.0x6.0-10 R20, 7" rim
TIRE_ID_SUBSTRINGS = ["43100", "18.0x6.0", "7 inch rim"]

FZ_MIN, FZ_MAX = 100.0, 1300.0   # [N] valid load window
SL_MAX = 0                     # [-] valid slip ratio window (unitless, not degrees)
SL_MIN = -0.2
PRESSURE_TARGET = 83.0           # [kPa]
PRESSURE_TOL = 5.0               # [kPa]
DECIMATE = 1                     # keep every Nth sample after filtering

_coeffs_cache = None

def pacejka_brake_force(p, kappa, Fz):
    """
    Pacejka BNP 1989 pure brake slip magic formula.
    kappa is slip ratio [-] (unitless, matches SL_MAX convention above —
    no radians/degrees ambiguity here, unlike the lateral model).
    Fz in N.

    Note: this formula only uses b0-b12 (13 of the 14 coefficients).
    b13 is carried for array-shape parity with the lateral model's .mat
    convention but is unused in the standard BNP89 brake form —
    it's hardcoded to 0 in generate_coeffs() below rather than fit.
    """
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13 = p

    C = b0
    D = Fz * (b1 * Fz + b2)
    BCD = (b3 * Fz**2 + b4 * Fz) * np.exp(-b5 * Fz)
    B = np.divide(BCD, C * D, out=np.zeros_like(np.asarray(BCD, dtype=float)), where=(C * D) != 0)
    E = b6 * Fz**2 + b7 * Fz + b8
    Sh = b9 * Fz + b10
    Sv = b11 * Fz + b12

    kx = kappa + Sh
    Fx = D * np.sin(C * np.arctan(B * kx - E * (B * kx - np.arctan(B * kx)))) + Sv
    return Fx

def filter_data(data):
    for run_id in data:
        SL = data[run_id]["SL"]
        SA = data[run_id]["SA"]
        FZ = data[run_id]["FZ"]
        P = data[run_id]["P"]

        valid = (
            (FZ > FZ_MIN) & (FZ < FZ_MAX)
            & (SL > SL_MIN)
            & (SL < SL_MAX)
            & (np.abs(P - PRESSURE_TARGET) < PRESSURE_TOL)
        )


        for entry_id in data[run_id]:
            data[run_id][entry_id] = data[run_id][entry_id][valid][::DECIMATE]


def generate_coeffs(run_data, run_id, n_starts=8, seed=0):
    global _coeffs_cache
    if _coeffs_cache is not None:
        return _coeffs_cache

    SL, FX, FZ, IA = run_data["SL"], run_data["FX"], run_data["FZ"], run_data["IA"]

    print("RETRIEVED RUN DATA: ", run_data)

    # ---- stage 1: anchor peak-force load curve D(Fz) ----
    fz_bin_centers = np.arange(150, 1251, 50)
    fz_bin_tol = 55.0
    fzc, dpk = [], []
    for fz in fz_bin_centers:
        sel = (np.abs(FZ - fz) < fz_bin_tol) & (np.abs(SL) > 0.155) & (np.abs(IA) < 0.6)
        if sel.sum() > 40:
            v = np.sort(np.abs(FX[sel]))
            fzc.append(fz)
            dpk.append(v[max(0, round(0.90 * len(v)) - 1)])
    fzc = np.asarray(fzc, dtype=float)
    dpk = np.asarray(dpk, dtype=float)

    A = np.column_stack([fzc**2, fzc])
    ab, *_ = np.linalg.lstsq(A, -dpk, rcond=None)
    b1_fix, b2_fix = ab
    print(f"stage 1 — anchored load curve: b1 = {b1_fix:.4e}, b2 = {b2_fix:.4f}")

    # ---- stage 1b: anchor initial slope BCD(Fz) via near-origin regression ----
    # This is the key fix — directly ties b3/b4/b5 to the data's actual
    # dFx/dkappa near kappa=0, instead of letting stage 2 guess them freely.
    sl_lin_tol = 0.02  # [-] "linear near origin" window — tune against your data
    fzc_slope, slope_data = [], []
    for fz in fz_bin_centers:
        sel = (np.abs(FZ - fz) < fz_bin_tol) & (np.abs(SL) < sl_lin_tol) & (np.abs(IA) < 0.6)
        if sel.sum() > 20:
            # least-squares slope through origin: slope = sum(SR*FX)/sum(SR^2)
            slope = (SL[sel] @ FX[sel]) / (SL[sel] @ SL[sel])
            fzc_slope.append(fz)
            slope_data.append(abs(slope))
    fzc_slope = np.asarray(fzc_slope, dtype=float)
    slope_data = np.asarray(slope_data, dtype=float)

    if fzc_slope.size == 0:
        raise ValueError(
            f"run_id={run_id}: stage 1b found no Fz bins with enough near-origin "
            f"data (sel.sum() > 20 never satisfied with sl_lin_tol={sl_lin_tol}). "
            f"This run file likely has too few samples for a per-file fit — "
            f"consider pooling multiple run files, widening sl_lin_tol, or "
            f"lowering the sel.sum() threshold for sparse single-file fits."
        )

    print(f"stage 1b — slope anchor points: {len(fzc_slope)} bins, "
          f"slope range {slope_data.min():.0f} - {slope_data.max():.0f} N/unit-kappa")

    def bcd_model(params, Fz):
        b3, b4, b5 = params
        return (b3 * Fz**2 + b4 * Fz) * np.exp(-b5 * Fz)

    def bcd_residuals(params):
        return bcd_model(params, fzc_slope) - slope_data

    # bounds now scoped to physically sane slopes (tens of thousands, not millions)
    bcd_lb = np.array([-1.0, 0.0, 1e-6])
    bcd_ub = np.array([1.0, 200.0, 1e-3])
    bcd_x0 = np.array([0.0, 80.0, 1e-4])

    bcd_result = least_squares(bcd_residuals, bcd_x0, bounds=(bcd_lb, bcd_ub), method="trf")
    b3_fix, b4_fix, b5_fix = bcd_result.x
    print(f"stage 1b — anchored slope curve: b3 = {b3_fix:.4e}, "
          f"b4 = {b4_fix:.4f}, b5 = {b5_fix:.4e}")

    # sanity check: print BCD at your four reference loads
    for fz_check in [200, 600, 1000, 1200]:
        bcd_check = bcd_model([b3_fix, b4_fix, b5_fix], fz_check)
        print(f"          BCD at Fz={fz_check}: {bcd_check:.1f} N/unit-kappa")

    # ---- stage 2: fit ONLY the remaining shape parameters ----
    # free vector order: [b0, b6, b7, b8, b9, b10, b11, b12]  (8 params)
    # b1, b2, b3, b4, b5 all fixed from stage 1 / stage 1b now
    def expand(free):
        b0, b6, b7, b8, b9, b10, b11, b12 = free
        return np.array([b0, b1_fix, b2_fix, b3_fix, b4_fix, b5_fix,
                          b6, b7, b8, b9, b10, b11, b12, 0.0])

    def residuals(free):
        p = expand(free)
        model = pacejka_brake_force(p, SL, FZ)
        return (FX - model) / FZ

    #        b0    b6     b7    b8    b9   b10   b11   b12
    lb = np.array([1.4, -1e-3, -2.0, -3.0, -5.0, -50, -5.0, -200])
    ub = np.array([2.2,  1e-3,  2.0,  0.8,  5.0,  50,  5.0,  200])

    p_nom_free = np.array([1.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    rng = np.random.default_rng(seed)
    best_cost = np.inf
    best_free = p_nom_free

    print(f"stage 2 — fitting shape parameters ({n_starts} starts)...")
    for s in range(n_starts):
        x0 = p_nom_free if s == 0 else rng.uniform(lb, ub)
        try:
            result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", loss="linear")
        except Exception as err:
            print(f"  start {s+1}/{n_starts}  skipped ({err})")
            continue

        cost = 2 * result.cost
        print(f"  start {s+1}/{n_starts}  normalized cost = {cost:.5f}")
        if cost < best_cost:
            best_cost = cost
            best_free = result.x

    p_fit = expand(best_free)
    rms_err = np.sqrt(np.mean((FX - pacejka_brake_force(p_fit, SL, FZ)) ** 2))
    print(f"\nbest normalized cost = {best_cost:.5f}")
    print(f"fit complete — rms error: {rms_err:.2f} N\n")

    coeff_names = ["b0 (C)", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
                   "b8", "b9", "b10", "b11", "b12", "b13 (unused)"]
    print("fitted coefficients:")
    for name, val in zip(coeff_names, p_fit):
        print(f"  {name:<14} = {val:12.6f}")

    savemat(Path(f"../../data/coeffs/brake/[{run_id}] hoosier_r20_tire_params_brake.mat").expanduser().resolve(), {
        "coeffs": p_fit,
        "coeff_names": coeff_names,
        "rms_error_N": rms_err,
        "tire": "Hoosier 43100 18.0x6.0-10 R20",
        "source": "FSAE TTC Round 9, Calspan Tire Research Facility",
        "model": "Pacejka BNP 1989 Brake",
    })
    print("\ntire parameters saved to hoosier_r20_tire_params_brake.mat")

    _coeffs_cache = p_fit
    return p_fit


def plot_fx_vs_sl(run_data, run_id, fn_buckets):
    """
    Brake force vs slip ratio, one curve per Fz bucket in fn_buckets [N].
    Model curves overlaid on measured scatter near each bucket.
    """
    SL, FX, FZ, IA = run_data["SL"], run_data["FX"], run_data["FZ"], run_data["IA"]
    p_fit = generate_coeffs(run_data, run_id)

    sl_lo, sl_hi = SL.min(), SL.max()
    sl_vec = np.linspace(sl_lo, sl_hi, 300)
    fz_tol = 60.0
    cmap = plt.get_cmap("jet", len(fn_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, fz in enumerate(fn_buckets):
        idx = (np.abs(FZ - fz) < fz_tol) & (np.abs(IA) < 0.5)
        if idx.sum() < 30:
            print(f"skipping F_Z = {fz} N (only {idx.sum()} points nearby)")
            continue

        ax.scatter(SL[idx], FX[idx], s=5, color=cmap(k), alpha=0.20)

        fx_pred = pacejka_brake_force(p_fit, sl_vec, fz * np.ones_like(sl_vec))
        ax.plot(sl_vec, fx_pred, color=cmap(k), linewidth=2, label=f"F_Z = {fz:g} N")

    ax.axhline(0, linestyle="--", color="k", alpha=0.25)
    ax.axvline(0, linestyle="--", color="k", alpha=0.25)
    ax.set_xlabel("slip ratio  κ  [-]")
    ax.set_ylabel("brake force  F_X  [N]")
    ax.set_title("brake force vs slip ratio (lines = model, dots = measured)")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(Path(f"../../figures/tyres/brake/[{run_id}] Fx vs SL (Brake).png").expanduser().resolve(), dpi=150)
    plt.close(fig)


def plot_fx_vs_fn(run_data, run_id, sl_buckets):
    """
    Brake force vs normal force, one curve per slip ratio in sl_buckets [-].
    Model-only continuous Fz sweep — direct visualization of brake
    load sensitivity (sub-linear rise of Fx with Fz).
    """
    p_fit = generate_coeffs(run_data, run_id)

    fz_vec = np.linspace(100, 1300, 300)
    cmap = plt.get_cmap("viridis", len(sl_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, sl in enumerate(sl_buckets):
        fx_pred = np.abs(pacejka_brake_force(p_fit, sl * np.ones_like(fz_vec), fz_vec))
        ax.plot(fz_vec, fx_pred, color=cmap(k), linewidth=2, label=f"κ = {sl:g}")

    ax.set_xlabel("normal force  F_Z  [N]")
    ax.set_ylabel("brake force  |F_X|  [N]")
    ax.set_title("brake force vs normal force — load sensitivity")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(Path(f"../../figures/tyres/brake/[{run_id}] Fx vs Fn (Brake).png").expanduser().resolve(), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data = load_data(TYRE_DATA_PATH, TIRE_ID_SUBSTRINGS, start_idx=0)
    filter_data(data)

    for run_id in data:
        try:
            _coeffs_cache = None

            if len(data[run_id]["SL"]) > 0:
                generate_coeffs(data[run_id], run_id)
                plot_fx_vs_sl(data[run_id], run_id, fn_buckets=[200, 600, 1000, 1200])
                plot_fx_vs_fn(data[run_id], run_id, sl_buckets=[1.92, 4, 8, 12])
        except Exception as e:
            print(str(e))