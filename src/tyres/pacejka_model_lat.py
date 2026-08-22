import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares
from pathlib import Path
from pacejka_model_helper import load_data

TYRE_DATA_PATH = Path("../../data/tyres/RunData_Cornering_Matlab_SI_Round9").expanduser().resolve()  # folder containing the B2356run*.mat TTC run files

# tire filter — Hoosier 43075 16x7.5-10 R20, 8" rim
TIRE_ID_SUBSTRINGS = ["43075", "16x7.5", "8 inch rim"]

FZ_MIN, FZ_MAX = 100.0, 1300.0   # [N] valid load window
SA_MAX = 14.0                    # [deg] valid slip angle window
PRESSURE_TARGET = 83.0           # [kPa]
PRESSURE_TOL = 5.0               # [kPa]
DECIMATE = 1                     # keep every Nth sample after filtering

_coeffs_cache = None
_coeffs_cache_4param = None

def filter_data(data):
    for run_id in data:
        SL = data[run_id]["SL"]
        SA = data[run_id]["SA"]
        FZ = data[run_id]["FZ"]
        P = data[run_id]["P"]

        valid = (
            (FZ > FZ_MIN) & (FZ < FZ_MAX)
            & (np.abs(SA) < SA_MAX)
            & (np.abs(P - PRESSURE_TARGET) < PRESSURE_TOL)
            # & (np.abs(SL) < 0.05)
        )

        for entry_id in data[run_id]:
            data[run_id][entry_id] = data[run_id][entry_id][valid][::DECIMATE]

def pacejka_lat_force(p, alpha_deg, Fz, gamma_deg=0.0):
    """
    Pacejka BNP 1989 lateral magic formula.
    alpha_deg, gamma_deg in DEGREES (matches this fit's calibration). Fz in N.
    Vectorized: alpha_deg / Fz / gamma_deg may be arrays (broadcastable).
    """
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = p

    C = a0
    D = a1 * Fz**2 + a2 * Fz
    BCD = a3 * np.sin(2 * np.arctan(Fz / a4)) * (1 - a5 * np.abs(gamma_deg))
    B = np.divide(BCD, C * D, out=np.zeros_like(np.asarray(BCD, dtype=float)), where=(C * D) != 0)
    E = a6 * Fz + a7
    SH = a8 * gamma_deg + a9 * Fz + a10
    SV = a11 * Fz * gamma_deg + a12 * Fz + a13

    phi = alpha_deg + SH
    FY = D * np.sin(C * np.arctan(B * phi - E * (B * phi - np.arctan(B * phi)))) + SV
    return FY


def generate_coeffs(run_data, run_id, n_starts=8, seed=0):
    """
    Two-stage fit:
      stage 1 — anchor D(Fz) = a1*Fz^2 + a2*Fz to the data's peak |FY| per load bin
      stage 2 — multi-start bounded nonlinear least squares on the remaining
                 12 shape parameters, with a1/a2 held fixed at stage-1 values
    Returns the fitted 14-element coefficient array and caches it.
    """
    global _coeffs_cache
    if _coeffs_cache is not None:
        return _coeffs_cache

    SA, FY, FZ, IA = run_data["SA"], run_data["FY"], run_data["FZ"], run_data["IA"]

    # ---- stage 1: anchor peak-force load curve ----
    fz_bin_centers = np.arange(150, 1251, 50)
    fz_bin_tol = 55.0
    fzc, dpk = [], []
    for fz in fz_bin_centers:
        sel = (np.abs(FZ - fz) < fz_bin_tol) & (np.abs(SA) > 5) & (np.abs(IA) < 0.6)
        if sel.sum() > 40:
            v = np.sort(np.abs(FY[sel]))
            fzc.append(fz)
            dpk.append(v[max(0, round(0.90 * len(v)) - 1)])  # 90th percentile proxy
    fzc = np.asarray(fzc, dtype=float)
    dpk = np.asarray(dpk, dtype=float)

    A = np.column_stack([fzc**2, fzc])
    ab, *_ = np.linalg.lstsq(A, -dpk, rcond=None)
    a1_fix, a2_fix = ab
    print(f"stage 1 — anchored load curve: a1 = {a1_fix:.4e}, a2 = {a2_fix:.4f}")
    print(f"          peak mu  {abs(a1_fix*200 + a2_fix):.2f} (at 200 N) -> "
          f"{abs(a1_fix*1200 + a2_fix):.2f} (at 1200 N)")

    # ---- stage 2: multi-start fit of remaining 12 shape parameters ----
    # free vector order: [a0, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]
    def expand(free):
        a0, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = free
        return np.array([a0, a1_fix, a2_fix, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13])

    def residuals(free):
        p = expand(free)
        model = pacejka_lat_force(p, SA, FZ, IA)
        return (FY - model) / FZ  # load-normalized residual

    #        a0    a3     a4    a5      a6    a7    a8    a9     a10   a11   a12    a13
    lb = np.array([1.2, -5000, 100, 0.0, -5e-4, -1.0, -2, -0.5,  -0.5, -2, -0.05,  -20])
    ub = np.array([1.8, 0,     5000, 0.5, 5e-4,  1.5,  2,  0.5,   0.5,  2,  0.05,   20])

    p_nom_free = np.array([1.45, -800, 1900, 0.01, 0.0, 0.60, 0.0, 1e-5, 0.0, 0.0, 0.0, 0.0])

    rng = np.random.default_rng(seed)
    best_cost = np.inf
    best_free = p_nom_free

    print(f"stage 2 — fitting shape parameters ({n_starts} starts)...")
    for s in range(n_starts):
        x0 = p_nom_free if s == 0 else rng.uniform(lb, ub)
        try:
            result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", loss="cauchy")
        except Exception as err:
            print(f"  start {s+1}/{n_starts}  skipped ({err})")
            continue

        cost = 2 * result.cost  # least_squares cost is 0.5*sum(res^2)
        print(f"  start {s+1}/{n_starts}  normalized cost = {cost:.5f}")
        if cost < best_cost:
            best_cost = cost
            best_free = result.x

    p_fit = expand(best_free)
    rms_err = np.sqrt(np.mean((FY - pacejka_lat_force(p_fit, SA, FZ, IA)) ** 2))
    print(f"\nbest normalized cost = {best_cost:.5f}")
    print(f"fit complete — rms error: {rms_err:.2f} N\n")

    coeff_names = ["a0 (C)", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
                   "a8", "a9", "a10", "a11", "a12", "a13"]
    print("fitted coefficients:")
    for name, val in zip(coeff_names, p_fit):
        print(f"  {name:<14} = {val:12.6f}")

    savemat(Path(f"../../data/coeffs/lateral/[{run_id}] hoosier_r20_tire_params_lat.mat").expanduser().resolve(), {
        "coeffs": p_fit,
        "coeff_names": coeff_names,
        "rms_error_N": rms_err,
        "tire": "Hoosier 43075 16x7.5-10 R20",
        "source": "FSAE TTC Round 9, Calspan Tire Research Facility",
        "model": "Pacejka BNP 1989 Lateral",
    })
    print("\ntire parameters saved to hoosier_r20_tire_params.mat")

    _coeffs_cache = p_fit
    return p_fit


def generate_coeffs_4param(run_data, run_id, n_starts=8, seed=0):
    """
    Simplified 4-parameter Pacejka fit: B, C, D, E only — no Sh/Sv shift
    terms (consistent with the confirmed anti-symmetric behavior from the
    full fit) and no explicit Fz-dependence in the coefficients themselves.

    To still use the full multi-load dataset, this fits against the
    NORMALIZED friction coefficient mu = Fy/Fz rather than raw Fy. This
    means the fit assumes a single representative mu-vs-alpha SHAPE across
    all loads — it discards the load-sensitivity behavior (D(Fz) sub-linear
    scaling, stiffness-vs-load) that the full 14-coefficient model captures.
    Use this for a compact single-curve approximation, not as a replacement
    for the full model when load sensitivity matters to your analysis.
    """
    global _coeffs_cache_4param
    if _coeffs_cache_4param is not None:
        return _coeffs_cache_4param

    SA, FY, FZ, IA = run_data["SA"], run_data["FY"], run_data["FZ"], run_data["IA"]
    mu = FY / FZ  # normalized lateral force [-]

    # ---- stage 1: anchor D as the representative peak |mu| across all loads ----
    sel = (np.abs(SA) > 5) & (np.abs(IA) < 0.6)
    print("MU Shape: ", mu.shape)
    if sel.sum() < 40:
        raise ValueError(
            f"run_id={run_id}: not enough near-saturation points ({sel.sum()}) "
            f"to anchor D for the 4-parameter fit."
        )

    mu_sel = mu[sel]
    idx_sorted = np.argsort(np.abs(mu_sel))
    D_fix = mu_sel[idx_sorted[max(0, round(0.90 * len(mu_sel)) - 1)]]  # signed now
    print(f"stage 1 — anchored peak mu: D = {D_fix:.4f}")

    # ---- stage 2: fit B, C, E (D held fixed from stage 1) ----
    def pacejka_4param(p, alpha_deg, D):
        B, C, E = p
        return D * np.sin(C * np.arctan(B * alpha_deg - E * (B * alpha_deg - np.arctan(B * alpha_deg))))

    def residuals(free):
        model = pacejka_4param(free, SA, D_fix)
        return mu - model

    #        B     C     E
    lb = np.array([0.05, 1.2, -3.0])
    ub = np.array([2.0,  1.9,  1.0])

    p_nom_free = np.array([0.3, 1.5, 0.5])

    rng = np.random.default_rng(seed)
    best_cost = np.inf
    best_free = p_nom_free

    print(f"stage 2 — fitting B, C, E ({n_starts} starts)...")
    for s in range(n_starts):
        x0 = p_nom_free if s == 0 else rng.uniform(lb, ub)
        try:
            result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", loss="cauchy")
        except Exception as err:
            print(f"  start {s+1}/{n_starts}  skipped ({err})")
            continue

        cost = 2 * result.cost
        print(f"  start {s+1}/{n_starts}  cost = {cost:.6f}")
        if cost < best_cost:
            best_cost = cost
            best_free = result.x

    B_fit, C_fit, E_fit = best_free
    p_fit = np.array([B_fit, C_fit, D_fix, E_fit])

    mu_pred = pacejka_4param(best_free, SA, D_fix)
    rms_err_mu = np.sqrt(np.mean((mu - mu_pred) ** 2))
    rms_err_N = np.sqrt(np.mean((FY - mu_pred * FZ) ** 2))  # back-converted to force units for comparability
    print(f"\nbest cost = {best_cost:.6f}")
    print(f"fit complete — rms error: {rms_err_mu:.4f} (mu, unitless), "
          f"{rms_err_N:.2f} N (approx, back-converted)\n")

    coeff_names = ["B", "C", "D", "E"]
    print("fitted coefficients (4-parameter model):")
    for name, val in zip(coeff_names, p_fit):
        print(f"  {name:<4} = {val:12.6f}")

    savemat(Path(f"../../data/coeffs/lateral/[{run_id}] hoosier_r20_tire_params_lat_4param.mat").expanduser().resolve(), {
        "coeffs": p_fit,
        "coeff_names": coeff_names,
        "rms_error_mu": rms_err_mu,
        "rms_error_N_approx": rms_err_N,
        "tire": "Hoosier 43075 16x7.5-10 R20",
        "source": "FSAE TTC Round 9, Calspan Tire Research Facility",
        "model": "Pacejka 4-Parameter (B,C,D,E; normalized mu=Fy/Fz, no load-dependence)",
    })
    print(f"\ntire parameters saved to [{run_id}] hoosier_r20_tire_params_lat_4param.mat")

    _coeffs_cache_4params = p_fit
    return p_fit


def pacejka_lat_force_4param(p, alpha_deg, Fz):
    """
    Evaluate the 4-parameter model: returns Fy in N given Fz.
    p = [B, C, D, E]; D is peak MU (unitless), so Fy = mu(alpha) * Fz.
    """
    B, C, D, E = p
    mu = D * np.sin(C * np.arctan(B * alpha_deg - E * (B * alpha_deg - np.arctan(B * alpha_deg))))
    return mu * Fz


def plot_fy_vs_sa(run_data, run_id, fn_buckets):
    """
    Lateral force vs slip angle, one curve per Fz bucket in fn_buckets [N].
    Model curves (zero camber) overlaid on measured scatter near each bucket.
    """
    SA, FY, FZ, IA = run_data["SA"], run_data["FY"], run_data["FZ"], run_data["IA"]
    p_fit = generate_coeffs(run_data, run_id)

    sa_vec = np.linspace(-14, 14, 300)
    fz_tol = 60.0
    cmap = plt.get_cmap("jet", len(fn_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, fz in enumerate(fn_buckets):
        idx = (np.abs(FZ - fz) < fz_tol) & (np.abs(IA) < 0.5)
        # if idx.sum() < 30:
        #     print(f"skipping F_Z = {fz} N (only {idx.sum()} points nearby)")
        #     continue

        ax.scatter(SA[idx], FY[idx], s=5, color=cmap(k), alpha=0.20)

        fy_pred = pacejka_lat_force(p_fit, sa_vec, fz * np.ones_like(sa_vec), 0.0)
        ax.plot(sa_vec, fy_pred, color=cmap(k), linewidth=2, label=f"F_Z = {fz:g} N")

    ax.axhline(0, linestyle="--", color="k", alpha=0.25)
    ax.axvline(0, linestyle="--", color="k", alpha=0.25)
    ax.set_xlabel("slip angle  α  [deg]")
    ax.set_ylabel("lateral force  F_Y  [N]")
    ax.set_title("lateral force vs slip angle (lines = model, dots = measured)")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(Path(f"../../figures/tyres/lateral/[{run_id}] Fy vs SA.png").expanduser().resolve())

def plot_fy_vs_sa_4param(run_data, run_id, fn_buckets):
    """
    Lateral force vs slip angle, one curve per Fz bucket in fn_buckets [N].
    Model curves (zero camber) overlaid on measured scatter near each bucket.
    """
    SA, FY, FZ, IA = run_data["SA"], run_data["FY"], run_data["FZ"], run_data["IA"]
    p_fit = generate_coeffs_4param(run_data, run_id)

    sa_vec = np.linspace(-SA_MAX, SA_MAX, 300)
    fz_tol = 60.0
    cmap = plt.get_cmap("jet", len(fn_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, fz in enumerate(fn_buckets):
        idx = (np.abs(FZ - fz) < fz_tol) & (np.abs(IA) < 0.5)
        # if idx.sum() < 30:
        #     print(f"skipping F_Z = {fz} N (only {idx.sum()} points nearby)")
        #     continue

        ax.scatter(SA[idx], FY[idx], s=5, color=cmap(k), alpha=0.20)

        fy_pred = pacejka_lat_force_4param(p_fit, sa_vec, fz * np.ones_like(sa_vec))
        ax.plot(sa_vec, fy_pred, color=cmap(k), linewidth=2, label=f"F_Z = {fz:g} N")

    ax.axhline(0, linestyle="--", color="k", alpha=0.25)
    ax.axvline(0, linestyle="--", color="k", alpha=0.25)
    ax.set_xlabel("slip angle  α  [deg]")
    ax.set_ylabel("lateral force  F_Y  [N]")
    ax.set_title("lateral force vs slip angle (lines = model, dots = measured)")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(Path(f"../../figures/tyres/lateral/[{run_id}] Fy vs SA - 4 Param.png").expanduser().resolve())


def plot_fy_vs_fn(run_data, run_id, sa_buckets):
    """
    Lateral force vs normal force, one curve per slip angle in sa_buckets [deg].
    Model-only (continuous Fz sweep, zero camber) — this is the direct
    visualization of load sensitivity: each curve's sub-linear rise with Fz
    shows the diminishing-returns behavior discussed earlier.
    """
    p_fit = generate_coeffs(run_data, run_id)

    fz_vec = np.linspace(100, 1300, 300)
    cmap = plt.get_cmap("viridis", len(sa_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, sa in enumerate(sa_buckets):
        fy_pred = np.abs(pacejka_lat_force(p_fit, sa * np.ones_like(fz_vec), fz_vec, 0.0))
        ax.plot(fz_vec, fy_pred, color=cmap(k), linewidth=2, label=f"α = {sa:g}°")

    ax.set_xlabel("normal force  F_Z  [N]")
    ax.set_ylabel("lateral force  |F_Y|  [N]")
    ax.set_title("lateral force vs normal force — load sensitivity")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(Path(f"../../figures/tyres/lateral/[{run_id}] Fy vs FN.png").expanduser().resolve())


def plot_naive(x_points, y_points):
    """
    Quick diagnostic scatter: Fy (y-axis) vs SA (x-axis), no model overlay.
    Useful for eyeballing raw data before/independent of a Pacejka fit.
    """
    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x_points, y_points, s=5, alpha=0.25, color="tab:blue")

    ax.axhline(0, linestyle="--", color="k", alpha=0.25)
    ax.axvline(0, linestyle="--", color="k", alpha=0.25)
    ax.set_xlabel("slip angle  α  [deg]")
    ax.set_ylabel("lateral force  F_Y  [N]")
    ax.set_title("lateral force vs slip angle (raw data)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(Path("../../figures/tyres/lateral/Fy vs SA (naive).png").expanduser().resolve())
    plt.close(fig)


if __name__ == "__main__":
    data = load_data(TYRE_DATA_PATH, TIRE_ID_SUBSTRINGS, start_idx=750)
    filter_data(data)

    print("[B2356run8] Data: ", data["B2356run8"]["SA"].shape)

    for run_id in data:
        _coeffs_cache = None

        generate_coeffs(data[run_id], run_id)
        plot_fy_vs_sa(data[run_id], run_id, fn_buckets=[200, 600, 1000, 1200])
        plot_fy_vs_fn(data[run_id], run_id, sa_buckets=[1.92, 4, 8, 12])

        _coeffs_cache_4param = None
        generate_coeffs_4param(data[run_id], run_id)
        plot_fy_vs_sa_4param(data[run_id], run_id, fn_buckets=[200, 300, 650, 1000, 1200])