# scales hoosier 43164 longitudinal and combined slip coeffs onto the 43075, which has no round 9 drive/brake data.
# fits mf5.2 lat + long + combined to the 43164 round 9 cornering and drive/brake runs, takes the 43075 / 43164
# lat ratios (peak mu, cornering stiffness) and applies them to the 43164 fits, mf lambda / normalized slip style.
# run from repo root: python -m src.models.tire_converter

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares

from .tire import Tire, LAT_MF52_COEFF_NAMES, P_BUCKETS, P_TOL, IA_RAMP
from ..utils.files import (DEFAULT_RUN_ID, LAT_COEFFS_DIR, LONG_COEFFS_DIR, COMB_COEFFS_DIR, LAT_DATA_DIR, LONG_DATA_DIR,
                           DEFAULT_LAT_MF52_COEFFS_PATH, DEFAULT_LONG_MF52_COEFFS_PATH, DEFAULT_COMB_MF52_COEFFS_PATH)

SRC_TIRE_ID = ["43164", "20.5x7.0-13"]
SRC_RIM = "8 inch rim" # same rim as the 43075 reference fit (run8), None pools 7 and 8 inch
SRC_NAME = "Hoosier 43164 20.5x7.0-13 R20"
REF_NAME = "Hoosier 43075 16x7.5-10 R20"

V_TARGET, V_TOL = 40.0, 3.0 # kph, run8 was tested at 40 kph
FZ_MIN = 100.0 # n
FZ_MAX_LAT = 1300.0 # n, run8 tops out at ~1120 n so drop the 43164's 1550 n sweeps
SA_PURE = 0.5 # deg, pure longitudinal window
SA_COMB_MAX = 7.0 # deg, drive/brake runs sweep slip ratio at sa = 0, -3, -6
SL_SWEEP_MIN = 0.1 # slip ratio range a run needs to count as a drive/brake sweep
KAPPA_FY0 = 0.01 # |kappa| window for the drive/brake session's own fy0

# 43075 tested load range, ratios are only taken where both lat fits are backed by data
FZ_GRID = np.linspace(220, 1110, 60)
TABLE_FZ = [220, 440, 670, 890, 1110]

LONG_MF52_COEFF_NAMES = [
    "pCx1", "pDx1", "pDx2", "pDx3",
    "pEx1", "pEx2", "pEx3", "pEx4",
    "pKx1", "pKx2", "pKx3",
    "pHx1", "pHx2",
    "pVx1", "pVx2"]

# initial guess / bounds for the 15 long mf5.2 coeffs (same order as LONG_MF52_COEFF_NAMES)
LONG_MF52_X0 = [1.6, 2.5, -0.2, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0]
LONG_MF52_LB = [1.0, 0.5, -2, -50, -10, -10, -10, -1, 1, -50, -5, -0.05, -0.05, -0.5, -0.5]
LONG_MF52_UB = [2.5, 5.0, 2, 50, 1, 10, 10, 1, 200, 50, 5, 0.05, 0.05, 0.5, 0.5]
LONG_MF52_MIN_POINTS = 500

COMB_MF52_COEFF_NAMES = [
    "rBx1", "rBx2", "rCx1", "rEx1", "rEx2", "rHx1",
    "rBy1", "rBy2", "rBy3", "rCy1", "rEy1", "rEy2", "rHy1", "rHy2",
    "rVy1", "rVy2", "rVy3", "rVy4", "rVy5", "rVy6"]
GX_NAMES = COMB_MF52_COEFF_NAMES[:6]

# initial guess / bounds for the 20 combined coeffs (same order as COMB_MF52_COEFF_NAMES).
# the weightings are even in their B terms and svyk only sees the sign of rVy1-3 * rVy5 * rVy6,
# so the B's and rVy4-6 are kept >= 0
COMB_MF52_X0 = [12, 8, 1.0, 0, 0, 0, 8, 5, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 1.9, 10]
COMB_MF52_LB = [0, 0, 0.5, -10, -10, -0.1, 0, 0, -0.1, 0.5, -10, -10, -0.1, -0.1, -1, -1, -50, 0, 0, 0]
COMB_MF52_UB = [100, 100, 2.0, 1, 10, 0.1, 100, 100, 0.1, 2.0, 1, 10, 0.1, 0.1, 1, 1, 50, 100, 10, 100]
# only sa = 0, -3, -6 tested, so the slip angle asymmetry of the weightings can't be seen
COMB_MF52_FIXED = {"rHx1": 0.0, "rBy3": 0.0}

PLOT_FZ = [220, 650, 1100]
COMB_PLOT_FZ = 650
COMB_PLOT_SA = [0, -3, -6]
FZ_PLOT_TOL = 75
SL_RANGE = np.linspace(-0.25, 0.25, 300)
KPA_PER_PSI = 6.89476

REF_COLOR = "#2a78d6" # 43075
SRC_COLOR = "#eb6834" # 43164


# ---------- model ----------

def long_mf52_model(params, kappa, gamma, fz, fz0):
    """Pure MF5.2 longitudinal math. kappa [-], gamma [rad], fz [N]."""
    (pCx1, pDx1, pDx2, pDx3,
     pEx1, pEx2, pEx3, pEx4,
     pKx1, pKx2, pKx3,
     pHx1, pHx2,
     pVx1, pVx2) = params

    dfz = (fz - fz0) / fz0

    Cx = pCx1
    mux = (pDx1 + pDx2 * dfz) * (1 - pDx3 * gamma**2)
    Dx = mux * fz

    Kx = fz * (pKx1 + pKx2 * dfz) * np.exp(pKx3 * dfz)

    denom = Cx * Dx
    Bx = np.divide(Kx, denom, out=np.zeros_like(np.asarray(denom, dtype=float)),
                   where=np.abs(denom) > 1e-9)

    SHx = pHx1 + pHx2 * dfz
    SVx = fz * (pVx1 + pVx2 * dfz)

    kappa_x = kappa + SHx

    Ex = (pEx1 + pEx2 * dfz + pEx3 * dfz**2) * (1 - pEx4 * np.sign(kappa_x))
    Ex = np.minimum(Ex, 1.0)

    Bk = Bx * kappa_x
    return Dx * np.sin(Cx * np.arctan(Bk - Ex * (Bk - np.arctan(Bk)))) + SVx


def eval_long(coeffs, kappa, gamma_deg, fz):
    c = coeffs
    return long_mf52_model([c[n] for n in LONG_MF52_COEFF_NAMES], kappa, np.deg2rad(gamma_deg), fz, c["fz0"])


def weight_mf52(B, C, E, s, sh):
    # mf5.2 combined slip cosine weighting, exactly 1 at s = 0
    E = np.minimum(E, 1.0)

    def shape(x):
        return np.cos(C * np.arctan(B * x - E * (B * x - np.arctan(B * x))))

    return shape(s + sh) / shape(sh)


def comb_mf52_model(c, mu_y, alpha, kappa, gamma, fz, fz0):
    """MF5.2 combined slip terms (Gxa, Gyk, SVyk) from a coeff dict, so that
    Fx = Gxa * Fx0 and Fy = Gyk * Fy0 + SVyk. alpha, gamma [rad], kappa [-],
    mu_y = Dy / Fz of the pure lat fit."""
    dfz = (fz - fz0) / fz0

    Bxa = c["rBx1"] * np.cos(np.arctan(c["rBx2"] * kappa))
    gxa = weight_mf52(Bxa, c["rCx1"], c["rEx1"] + c["rEx2"] * dfz, alpha, c["rHx1"])

    Byk = c["rBy1"] * np.cos(np.arctan(c["rBy2"] * (alpha - c["rBy3"])))
    gyk = weight_mf52(Byk, c["rCy1"], c["rEy1"] + c["rEy2"] * dfz, kappa, c["rHy1"] + c["rHy2"] * dfz)

    dvyk = mu_y * fz * (c["rVy1"] + c["rVy2"] * dfz + c["rVy3"] * gamma) * np.cos(np.arctan(c["rVy4"] * alpha))
    svyk = dvyk * np.sin(c["rVy5"] * np.arctan(c["rVy6"] * kappa))

    return gxa, gyk, svyk


def lat_params(c):
    return [c[n] for n in LAT_MF52_COEFF_NAMES]


def scale_lat(c, l_mu, l_k):
    # mf lambda scaling of a lat fit, lmuy on peak mu and sv, lky on cornering stiffness
    s = dict(c)
    for n in ["pDy1", "pDy2", "pVy1", "pVy2", "pVy3", "pVy4"]:
        s[n] = c[n] * l_mu
    s["pKy1"] = c["pKy1"] * l_k
    return s


def lat_traits(c, fz):
    # zero camber peak mu and cornering stiffness [n/rad] of an mf5.2 lat fit
    dfz = (fz - c["fz0"]) / c["fz0"]
    mu = c["pDy1"] + c["pDy2"] * dfz
    ky = c["pKy1"] * c["fz0"] * np.sin(2 * np.arctan(fz / (c["pKy2"] * c["fz0"])))
    return mu, ky


def long_traits(c, fz):
    # zero camber peak mu, slip stiffness / fz, curvature and shifts of an mf5.2 long fit
    dfz = (fz - c["fz0"]) / c["fz0"]
    mu = c["pDx1"] + c["pDx2"] * dfz
    kx = (c["pKx1"] + c["pKx2"] * dfz) * np.exp(c["pKx3"] * dfz)
    ex = c["pEx1"] + c["pEx2"] * dfz + c["pEx3"] * dfz**2
    shx = c["pHx1"] + c["pHx2"] * dfz
    svx = c["pVx1"] + c["pVx2"] * dfz
    return mu, kx, ex, shx, svx


# ---------- data ----------

def run_num(path):
    return int(path.stem.removeprefix("B2356run"))


def load_runs(data_dir, lateral):
    # every 43164 run in data_dir, cut down to the reference speed (and pure slip for lat)
    runs = dict()
    for fpath in sorted(Path(Path.cwd() / data_dir).glob("B2356run*.mat"), key=run_num):
        tire_id = str(np.asarray(loadmat(fpath, variable_names=["tireid"])["tireid"]).item()).strip()
        if not all(sub in tire_id for sub in SRC_TIRE_ID):
            continue

        print(f"  {fpath.stem:<12} {tire_id:<42} ", end="")
        if SRC_RIM is not None and SRC_RIM not in tire_id:
            print("skipped, rim")
            continue

        tire = Tire(run_id=fpath.stem, data_path=fpath)
        tire.tire_id_substrings = SRC_TIRE_ID
        d = tire.load_data()

        if not lateral and np.ptp(d["SL"]) < SL_SWEEP_MIN:
            print("skipped, no slip ratio sweep")
            continue

        at_speed = np.abs(d["V"] - V_TARGET) < V_TOL
        if not at_speed.any():
            print(f"skipped, nothing at {V_TARGET:g} kph")
            continue
        d = {ch: v[at_speed] for ch, v in d.items()}

        if lateral:
            # same cleanup as the 43075 fit so the lat comparison is like for like
            tire.run_data = d
            tire.process_data()
            d = tire.run_data
            keep = (d["FZ"] > FZ_MIN) & (d["FZ"] < FZ_MAX_LAT)
        else:
            keep = (d["FZ"] > FZ_MIN) & (np.abs(d["SA"]) < SA_COMB_MAX)

        d = {ch: v[keep] for ch, v in d.items()}
        if len(d["ET"]) == 0:
            print("skipped, no usable data")
            continue

        print(f"{len(d['ET'])} pts")
        runs[fpath.stem] = d

    if not runs:
        raise ValueError(f"no usable {SRC_NAME} runs in {data_dir}")
    return runs


def merge(runs):
    channels = set.intersection(*(set(d) for d in runs.values()))
    return {ch: np.concatenate([d[ch] for d in runs.values()]) for ch in channels}


def coeffs_path(coeffs_dir, runs, name):
    tag = "B2356run" + "+".join(str(run_num(Path(r))) for r in runs)
    return Path(Path.cwd() / f"{coeffs_dir}[{tag}] {name}").expanduser().resolve()


def at_pressure(coeffs, p):
    near = min(coeffs, key=lambda k: abs(k - p))
    return coeffs[near] if abs(near - p) < P_TOL else None


def save_coeffs(path, coeffs, names, fields=("rms",), **info):
    # same layout as Tire.save_lat_mf52_coeffs, plus any other per pressure fields
    pressures = sorted(coeffs)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {
        "coeff_names": names,
        "pressures": np.array(pressures),
        "fz0": np.array([coeffs[p]["fz0"] for p in pressures]),
        **{f: np.array([coeffs[p][f] for p in pressures]) for f in fields},
        "coeff_values": np.array([[coeffs[p][n] for n in names] for p in pressures]),
        **info,
    })


def load_coeffs(path):
    # any lat / long / mz / combined coeff file back into {pressure: {name: value, "fz0": ...}}
    mat = loadmat(path)
    names = [str(n).strip() for n in mat["coeff_names"]]
    out = dict()
    for p, fz0, values in zip(mat["pressures"].ravel(), mat["fz0"].ravel(), np.atleast_2d(mat["coeff_values"])):
        out[float(p)] = {**dict(zip(names, values.astype(float))), "fz0": float(fz0)}
    return out


def print_coeffs(title, coeffs, names):
    for p in sorted(coeffs):
        c = coeffs[p]
        print(f"\n{title} @ P = {p:g} kPa ({p / KPA_PER_PSI:.0f} psi)")
        for f in ("rms", "rms_fx", "rms_fy"):
            if f in c:
                print(f"{f} = {c[f]:.1f} N")
        print(f"FNOMIN = {c['fz0']:.1f}")
        for name in names:
            print(f"{name.upper():<6} = {c[name]: .6e}")


# ---------- fitting ----------

def fit_long_mf52(d, pressures=P_BUCKETS):
    # fits all 15 mf5.2 long coeffs in one least squares, one set per tested pressure
    coeffs = dict()

    for p in pressures:
        idx = np.abs(d["P"] - p) < P_TOL
        if idx.sum() < LONG_MF52_MIN_POINTS:
            continue

        kappa = d["SL"][idx]
        gamma = np.deg2rad(d["IA"][idx])
        fz = d["FZ"][idx]
        fx = d["FX"][idx]
        fz0 = float(np.round(np.median(fz), -1))

        result = least_squares(
            lambda x: long_mf52_model(x, kappa, gamma, fz, fz0) - fx,
            LONG_MF52_X0, bounds=(LONG_MF52_LB, LONG_MF52_UB),
            x_scale="jac", loss="soft_l1", f_scale=50, max_nfev=20000)

        resid = long_mf52_model(result.x, kappa, gamma, fz, fz0) - fx

        c = dict(zip(LONG_MF52_COEFF_NAMES, result.x))
        c["fz0"] = fz0
        c["rms"] = float(np.sqrt(np.mean(resid**2)))
        coeffs[float(p)] = c

    if not coeffs:
        raise ValueError("Not enough points to construct an mf5.2 long fit")

    return coeffs


def fit_comb_mf52(d, src_lat, src_long, pressures=P_BUCKETS):
    """
    Per pressure, on top of the 43164 pure fits:
      1. fy0 of the drive/brake session, lmuy / lky on the cornering lat fit from the |kappa| ~ 0 points.
         the tire makes ~15 % less fy here than in its cornering runs and Gyk has to be 1 at kappa = 0
      2. Gxa against fx and Gyk + SVyk against fy, every slip angle at once
    """
    x0, lb, ub = (np.array(v, dtype=float) for v in (COMB_MF52_X0, COMB_MF52_LB, COMB_MF52_UB))
    x_fixed = np.array([COMB_MF52_FIXED.get(n, 0.0) for n in COMB_MF52_COEFF_NAMES])
    free = np.array([n not in COMB_MF52_FIXED for n in COMB_MF52_COEFF_NAMES])
    is_x = np.array([n in GX_NAMES for n in COMB_MF52_COEFF_NAMES])
    fit_x, fit_y = free & is_x, free & ~is_x

    coeffs = dict()

    for p in pressures:
        idx = np.abs(d["P"] - p) < P_TOL
        lat, lon = at_pressure(src_lat, p), at_pressure(src_long, p)
        if idx.sum() < LONG_MF52_MIN_POINTS or lat is None or lon is None:
            continue

        alpha = np.deg2rad(d["SA"][idx])
        gamma = np.deg2rad(d["IA"][idx])
        kappa = d["SL"][idx]
        fz, fx, fy = d["FZ"][idx], d["FX"][idx], d["FY"][idx]
        fz0 = lat["fz0"]

        # 1. session fy0
        k0 = np.abs(kappa) < KAPPA_FY0
        l_fit = least_squares(
            lambda l: Tire._lat_mf52_model(lat_params(scale_lat(lat, *l)), alpha[k0], gamma[k0], fz[k0], fz0) - fy[k0],
            [1.0, 1.0], bounds=([0.5, 0.5], [1.5, 1.5])).x
        lat_s = lat_params(scale_lat(lat, *l_fit))
        fy0 = Tire._lat_mf52_model(lat_s, alpha, gamma, fz, fz0)
        mu_y = Tire._lat_mf52_terms(lat_s, alpha, gamma, fz, fz0)[1] / fz
        fx0 = eval_long(lon, kappa, d["IA"][idx], fz)

        # 2. weightings
        def model(x, mask):
            full = x_fixed.copy()
            full[mask] = x
            return comb_mf52_model(dict(zip(COMB_MF52_COEFF_NAMES, full)), mu_y, alpha, kappa, gamma, fz, fz0)

        def resid_y(x):
            _, gyk, svyk = model(x, fit_y)
            return gyk * fy0 + svyk - fy

        rx = least_squares(lambda x: model(x, fit_x)[0] * fx0 - fx, x0[fit_x], bounds=(lb[fit_x], ub[fit_x]),
                           x_scale="jac", loss="soft_l1", f_scale=50, max_nfev=20000)
        ry = least_squares(resid_y, x0[fit_y], bounds=(lb[fit_y], ub[fit_y]),
                           x_scale="jac", loss="soft_l1", f_scale=50, max_nfev=20000)

        full = x_fixed.copy()
        full[fit_x], full[fit_y] = rx.x, ry.x
        c = dict(zip(COMB_MF52_COEFF_NAMES, full))
        gxa, gyk, svyk = comb_mf52_model(c, mu_y, alpha, kappa, gamma, fz, fz0)

        c["fz0"] = fz0
        c["rms_fx"] = float(np.sqrt(np.mean((gxa * fx0 - fx) ** 2)))
        c["rms_fy"] = float(np.sqrt(np.mean((gyk * fy0 + svyk - fy) ** 2)))
        c["rms_fx_pure"] = float(np.sqrt(np.mean((fx0 - fx) ** 2)))
        c["rms_fy_pure"] = float(np.sqrt(np.mean((fy0 - fy) ** 2)))
        c["lmuy_session"], c["lky_session"] = (float(v) for v in l_fit)
        coeffs[float(p)] = c

    if not coeffs:
        raise ValueError("Not enough points to construct an mf5.2 combined fit")

    return coeffs


def convert(ref_lat, src_lat, src_long):
    """
    MF lambda scaling, per 43075 pressure:
      lmu(fz) = mu_y 43075 / mu_y 43164   -> peak mu_x and SVx
      lk(fz)  = Ky 43075 / Ky 43164       -> slip stiffness Kx
    shape (C, E), camber and SHx stay as the 43164's. a Cy ratio isn't used since Cy and Ey
    trade off in the lat fits (the ratio swings 0.87 - 1.06 between pressures).
    everything is refit on the 43075 fz0 so the long coeffs share dfz with its lat fit.
    """
    out = dict()

    for p in sorted(ref_lat):
        ref, lat, lon = ref_lat[p], at_pressure(src_lat, p), at_pressure(src_long, p)
        if lat is None or lon is None:
            print(f"  no 43164 fit near P = {p:g} kPa, skipped")
            continue

        fz0 = ref["fz0"]
        dfz = (FZ_GRID - fz0) / fz0

        mu_r, ky_r = lat_traits(ref, FZ_GRID)
        mu_s, ky_s = lat_traits(lat, FZ_GRID)
        l_mu, l_k = mu_r / mu_s, ky_r / ky_s

        mux, kx, ex, shx, svx = long_traits(lon, FZ_GRID)
        kx_target = kx * l_k

        c = dict()
        c["pCx1"] = lon["pCx1"]
        c["pDx2"], c["pDx1"] = np.polyfit(dfz, mux * l_mu, 1)
        c["pDx3"] = lon["pDx3"]
        c["pEx3"], c["pEx2"], c["pEx1"] = np.polyfit(dfz, ex, 2)
        c["pEx4"] = lon["pEx4"]

        k_fit = least_squares(lambda x: (x[0] + x[1] * dfz) * np.exp(x[2] * dfz) - kx_target,
                              [np.interp(fz0, FZ_GRID, kx_target), 0, 0], x_scale="jac")
        c["pKx1"], c["pKx2"], c["pKx3"] = k_fit.x

        c["pHx2"], c["pHx1"] = np.polyfit(dfz, shx, 1)
        c["pVx2"], c["pVx1"] = np.polyfit(dfz, svx * l_mu, 1)
        c["fz0"] = fz0
        c["rms"] = lon["rms"] # no 43075 long data to check against, carry the 43164 fit error

        out[p] = c

        # how well the mf5.2 load forms hold the scaled curves
        mu_c, kx_c, *_ = long_traits(c, FZ_GRID)
        err_mu = np.max(np.abs(mu_c / (mux * l_mu) - 1)) * 100
        err_k = np.max(np.abs(kx_c / kx_target - 1)) * 100

        print(f"\n@ P = {p:g} kPa ({p / KPA_PER_PSI:.0f} psi)  refit err: mu {err_mu:.2f} %, kx {err_k:.2f} %")
        print(f"  {'fz [N]':>7} {'lmu':>6} {'lk':>6} {'mu_x 43164':>11} {'mu_x 43075':>11} "
              f"{'kx 43164':>10} {'kx 43075':>10}")
        for fz in TABLE_FZ:
            i = np.argmin(np.abs(FZ_GRID - fz))
            print(f"  {fz:>7.0f} {l_mu[i]:>6.3f} {l_k[i]:>6.3f} {mux[i]:>11.3f} {mu_c[i]:>11.3f} "
                  f"{kx[i] * fz / 1e3:>8.1f} k {kx_c[i] * fz / 1e3:>8.1f} k")

    if not out:
        raise ValueError("no pressure has a 43075 lat fit and both 43164 fits")

    return out


def convert_comb(ref_lat, src_lat, src_comb):
    """
    Normalized slip transfer per 43075 pressure. the weightings are taken as equal once slip is
    divided by slip at peak (~ mu / K), so
      s_alpha = (Ky / mu_y) 43075 / (Ky / mu_y) 43164 drive/brake session  -> terms on alpha
      s_kappa = lk / lmu, the ratio the long transfer used for Kx / mu_x     -> terms on kappa
    terms that multiply a slip scale by s, slip shifts divide by it, C, E, rVy1-3 and rVy5 carry over.
    taken at the 43075 fz0 and rebased onto it.
    """
    out = dict()

    for p in sorted(ref_lat):
        ref, lat, comb = ref_lat[p], at_pressure(src_lat, p), at_pressure(src_comb, p)
        if lat is None or comb is None:
            print(f"  no 43164 fit near P = {p:g} kPa, skipped")
            continue

        fz0 = ref["fz0"]
        mu_r, ky_r = lat_traits(ref, fz0)
        mu_s, ky_s = lat_traits(lat, fz0)
        s_kappa = (ky_r / ky_s) / (mu_r / mu_s)
        s_alpha = s_kappa * comb["lmuy_session"] / comb["lky_session"]

        # dfz on the 43164 fz0 = a + b * dfz on the 43075 fz0
        b = fz0 / comb["fz0"]
        a = b - 1

        def rebase(c0, c1):
            return c0 + c1 * a, c1 * b

        c = dict()
        c["rBx1"], c["rBx2"], c["rCx1"] = comb["rBx1"] * s_alpha, comb["rBx2"] * s_kappa, comb["rCx1"]
        c["rEx1"], c["rEx2"] = rebase(comb["rEx1"], comb["rEx2"])
        c["rHx1"] = comb["rHx1"] / s_alpha
        c["rBy1"], c["rBy2"], c["rBy3"] = comb["rBy1"] * s_kappa, comb["rBy2"] * s_alpha, comb["rBy3"] / s_alpha
        c["rCy1"] = comb["rCy1"]
        c["rEy1"], c["rEy2"] = rebase(comb["rEy1"], comb["rEy2"])
        c["rHy1"], c["rHy2"] = (v / s_kappa for v in rebase(comb["rHy1"], comb["rHy2"]))
        c["rVy1"], c["rVy2"] = rebase(comb["rVy1"], comb["rVy2"])
        c["rVy3"], c["rVy4"] = comb["rVy3"], comb["rVy4"] * s_alpha
        c["rVy5"], c["rVy6"] = comb["rVy5"], comb["rVy6"] * s_kappa

        c["fz0"] = fz0
        c["rms_fx"], c["rms_fy"] = comb["rms_fx"], comb["rms_fy"] # no 43075 combined data, carry the 43164 fit error
        c["s_alpha"], c["s_kappa"] = s_alpha, s_kappa
        out[p] = c

        print(f"  P = {p:g} kPa ({p / KPA_PER_PSI:.0f} psi)  s_alpha = {s_alpha:.3f}  s_kappa = {s_kappa:.3f}")

    if not out:
        raise ValueError("no pressure has a 43075 lat fit and a 43164 combined fit")

    return out


# ---------- plotting ----------

def plot_conversion(d_long, ref_lat, src_lat, src_long, ref_long, p):
    ref, lat, lon, conv = ref_lat[p], at_pressure(src_lat, p), at_pressure(src_long, p), ref_long[p]
    psi = p / KPA_PER_PSI

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # fx vs sl, 43164 data + fit and the converted 43075
    ax = axes[0]
    at_p = (np.abs(d_long["P"] - p) < P_TOL) & (np.abs(d_long["IA"]) < 0.5)
    for fz in PLOT_FZ:
        idx = at_p & (np.abs(d_long["FZ"] - fz) < FZ_PLOT_TOL)
        ax.scatter(d_long["SL"][idx], d_long["FX"][idx], s=4, color=SRC_COLOR, alpha=0.2, linewidths=0)

        fx_src = eval_long(lon, SL_RANGE, 0, fz)
        fx_ref = eval_long(conv, SL_RANGE, 0, fz)
        ax.plot(SL_RANGE, fx_src, color=SRC_COLOR, linewidth=2)
        ax.plot(SL_RANGE, fx_ref, color=REF_COLOR, linewidth=2)
        ax.annotate(f"{fz:g} N", (SL_RANGE[-1], fx_src[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color="0.3")

    ax.plot([], [], color=SRC_COLOR, linewidth=2, label="43164 fit (dots = data)")
    ax.plot([], [], color=REF_COLOR, linewidth=2, label="43075 converted")
    ax.axhline(0, linestyle="--", color="k", alpha=0.25)
    ax.axvline(0, linestyle="--", color="k", alpha=0.25)
    ax.set_xlim(SL_RANGE[0], SL_RANGE[-1] * 1.18)
    ax.set_xlabel("slip ratio  κ  [-]")
    ax.set_ylabel("longitudinal force  F_X  [N]")
    ax.set_title("Fx vs SL @ IA = 0")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # peak mu and stiffness vs load, lat dashed, long solid
    mu_ry, ky_ry = lat_traits(ref, FZ_GRID)
    mu_sy, ky_sy = lat_traits(lat, FZ_GRID)
    mu_sx, kx_s, *_ = long_traits(lon, FZ_GRID)
    mu_rx, kx_r, *_ = long_traits(conv, FZ_GRID)

    for ax, ys, ylabel, title in [
        (axes[1], (mu_ry, mu_sy, mu_sx, mu_rx), "peak μ  [-]", "peak friction vs load"),
        (axes[2], (np.abs(ky_ry) / 1e3, np.abs(ky_sy) / 1e3, kx_s * FZ_GRID / 1e3, kx_r * FZ_GRID / 1e3),
         "slip stiffness  [kN / unit slip]", "|Ky| [per rad] and Kx vs load"),
    ]:
        ax.plot(FZ_GRID, ys[0], color=REF_COLOR, linewidth=2, linestyle="--", label="43075 lat (fit)")
        ax.plot(FZ_GRID, ys[1], color=SRC_COLOR, linewidth=2, linestyle="--", label="43164 lat (fit)")
        ax.plot(FZ_GRID, ys[2], color=SRC_COLOR, linewidth=2, label="43164 long (fit)")
        ax.plot(FZ_GRID, ys[3], color=REF_COLOR, linewidth=2, label="43075 long (converted)")
        ax.set_xlabel("normal force  F_Z  [N]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"[{DEFAULT_RUN_ID}] 43164 -> 43075 long conversion @ P = {p:g} kPa ({psi:.0f} psi)")
    fig.tight_layout()
    fig.savefig(Path(Path.cwd() / f"figures/tyres/longitudinal/[{DEFAULT_RUN_ID}] Long conversion @ P = {p:g}.png").expanduser().resolve(), dpi=150)
    plt.close(fig)


def comb_forces(lat, lon, comb, alpha_deg, kappa, fz):
    # combined fx, fy at zero camber from pure + combined coeff dicts
    a = np.deg2rad(alpha_deg)
    fy0 = Tire._lat_mf52_model(lat_params(lat), a, 0.0, fz, lat["fz0"])
    mu_y = Tire._lat_mf52_terms(lat_params(lat), a, 0.0, fz, lat["fz0"])[1] / fz
    gxa, gyk, svyk = comb_mf52_model(comb, mu_y, a, kappa, 0.0, fz, comb["fz0"])
    return gxa * eval_long(lon, kappa, 0, fz), gyk * fy0 + svyk


def plot_comb_conversion(d, ref_lat, src_lat, src_long, ref_long, src_comb, ref_comb, p):
    comb = at_pressure(src_comb, p)
    lat_s = scale_lat(at_pressure(src_lat, p), comb["lmuy_session"], comb["lky_session"])
    lon = at_pressure(src_long, p)
    fz = COMB_PLOT_FZ
    psi = p / KPA_PER_PSI
    ref_ramp = ["#86b6ef", "#3987e5", "#1c5cab"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    at = (np.abs(d["P"] - p) < P_TOL) & (np.abs(d["IA"]) < 0.5) & (np.abs(d["FZ"] - fz) < FZ_PLOT_TOL)

    # 43164 fx and fy vs sl at each tested slip angle, data + combined fit
    for ax, i, ch, sas in [(axes[0], 0, "FX", COMB_PLOT_SA), (axes[1], 1, "FY", COMB_PLOT_SA[1:])]:
        for sa in sas:
            color = IA_RAMP[COMB_PLOT_SA.index(sa)]
            idx = at & (np.abs(d["SA"] - sa) < 0.5)
            ax.scatter(d["SL"][idx], d[ch][idx], s=4, color=color, alpha=0.25, linewidths=0)
            ax.plot(SL_RANGE, comb_forces(lat_s, lon, comb, sa, SL_RANGE, fz)[i], color=color, linewidth=2,
                    label=f"SA = {sa:g} deg")
        ax.axhline(0, linestyle="--", color="k", alpha=0.25)
        ax.axvline(0, linestyle="--", color="k", alpha=0.25)
        ax.set_xlabel("slip ratio  κ  [-]")
        ax.set_ylabel(f"{'longitudinal' if ch == 'FX' else 'lateral'} force  F_{ch[1]}  [N]")
        ax.set_title(f"43164 {ch[0]}{ch[1].lower()} vs SL @ F_Z = {fz} N, IA = 0 (lines = fit, dots = data)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # friction envelope, fy vs fx swept over sl at constant slip angle
    ax = axes[2]
    for j, sa in enumerate([-2, -4, -6]):
        fx_s, fy_s = comb_forces(lat_s, lon, comb, sa, SL_RANGE, fz)
        fx_r, fy_r = comb_forces(ref_lat[p], ref_long[p], ref_comb[p], sa, SL_RANGE, fz)
        ax.plot(fx_s, fy_s, color=IA_RAMP[j], linewidth=2)
        ax.plot(fx_r, fy_r, color=ref_ramp[j], linewidth=2)
        ax.annotate(f"{sa:g} deg", (0, fy_r.max()), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color="0.3")

    ax.plot([], [], color=IA_RAMP[1], linewidth=2, label="43164 (drive/brake session fit)")
    ax.plot([], [], color=ref_ramp[1], linewidth=2, label="43075 (converted)")
    ax.set_xlabel("longitudinal force  F_X  [N]")
    ax.set_ylabel("lateral force  F_Y  [N]")
    ax.set_title(f"Fy vs Fx at constant SA, swept over SL @ F_Z = {fz} N")
    ax.legend(loc="lower center")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"[{DEFAULT_RUN_ID}] 43164 -> 43075 combined slip conversion @ P = {p:g} kPa ({psi:.0f} psi)")
    fig.tight_layout()
    path = Path(Path.cwd() / f"figures/tyres/combined/[{DEFAULT_RUN_ID}] Combined conversion @ P = {p:g}.png").expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"{SRC_NAME} cornering runs:")
    lat_runs = load_runs(LAT_DATA_DIR, lateral=True)
    print(f"{SRC_NAME} drive/brake runs:")
    long_runs = load_runs(LONG_DATA_DIR, lateral=False)

    # 43164 lat, same fitter as the 43075 so the ratios only see the tire
    src = Tire(run_id="43164",
               mf52_coeffs_path=coeffs_path(LAT_COEFFS_DIR, lat_runs, "hoosier_43164_r20_lat_mf52_coeffs.mat"))
    src.run_data = merge(lat_runs)
    src.fit_lat_mf52()
    src.save_lat_mf52_coeffs()

    # 43164 long, drive/brake runs keep every slip angle but the pure fit only uses sa ~ 0
    d_all = merge(long_runs)
    d_long = {ch: v[np.abs(d_all["SA"]) < SA_PURE] for ch, v in d_all.items()}
    src_long = fit_long_mf52(d_long)
    save_coeffs(coeffs_path(LONG_COEFFS_DIR, long_runs, "hoosier_43164_r20_long_mf52_coeffs.mat"),
                src_long, LONG_MF52_COEFF_NAMES, tire=SRC_NAME,
                source="FSAE TTC Round 9, Calspan Tire Research Facility", model="MF5.2 pure longitudinal")

    print("\n43164 fit rms:")
    for p in sorted(src.lat_mf52_coeffs):
        lon = at_pressure(src_long, p)
        long_rms = f"{lon['rms']:.1f} N" if lon else "-"
        print(f"  P = {p:g} kPa  lat {src.lat_mf52_coeffs[p]['rms']:.1f} N  long {long_rms}")

    # 43075 reference lat
    ref = Tire(mf52_coeffs_path=DEFAULT_LAT_MF52_COEFFS_PATH)
    ref.load_lat_mf52_coeffs()

    print("\nscaling 43164 long onto the 43075 (lambdas from the lat fits, 43075 / 43164):")
    ref_long = convert(ref.lat_mf52_coeffs, src.lat_mf52_coeffs, src_long)

    save_coeffs(DEFAULT_LONG_MF52_COEFFS_PATH, ref_long, LONG_MF52_COEFF_NAMES, tire=REF_NAME,
                source=f"scaled from {SRC_NAME} round 9 drive/brake by the 43075 / 43164 lat mf5.2 ratios",
                model="MF5.2 pure longitudinal")

    print_coeffs(f"[{DEFAULT_RUN_ID}] mf5.2 longitudinal coeffs (43075, converted)", ref_long, LONG_MF52_COEFF_NAMES)
    print(f"\nsaved to {DEFAULT_LONG_MF52_COEFFS_PATH}")

    for p in ref_long:
        plot_conversion(d_long, ref.lat_mf52_coeffs, src.lat_mf52_coeffs, src_long, ref_long, p)

    # 43164 combined, normalized against the drive/brake session's own fy0
    src_comb = fit_comb_mf52(d_all, src.lat_mf52_coeffs, src_long)
    save_coeffs(coeffs_path(COMB_COEFFS_DIR, long_runs, "hoosier_43164_r20_comb_mf52_coeffs.mat"),
                src_comb, COMB_MF52_COEFF_NAMES, fields=("rms_fx", "rms_fy", "lmuy_session", "lky_session"),
                tire=SRC_NAME, source="FSAE TTC Round 9, Calspan Tire Research Facility",
                model="MF5.2 combined slip (Gxa, Gyk, SVyk)")

    print("\n43164 combined fit rms (pure slip curves alone in brackets):")
    for p, c in sorted(src_comb.items()):
        print(f"  P = {p:g} kPa  fx {c['rms_fx']:.0f} N ({c['rms_fx_pure']:.0f})  fy {c['rms_fy']:.0f} N "
              f"({c['rms_fy_pure']:.0f})  session lmuy {c['lmuy_session']:.3f}  lky {c['lky_session']:.3f}")

    print("\nscaling 43164 combined onto the 43075 (normalized slip):")
    ref_comb = convert_comb(ref.lat_mf52_coeffs, src.lat_mf52_coeffs, src_comb)

    save_coeffs(DEFAULT_COMB_MF52_COEFFS_PATH, ref_comb, COMB_MF52_COEFF_NAMES,
                fields=("rms_fx", "rms_fy", "s_alpha", "s_kappa"), tire=REF_NAME,
                source=f"scaled from {SRC_NAME} round 9 drive/brake by normalized slip (43075 / 43164 lat mf5.2)",
                model="MF5.2 combined slip (Gxa, Gyk, SVyk)")

    print_coeffs(f"[{DEFAULT_RUN_ID}] mf5.2 combined coeffs (43075, converted)", ref_comb, COMB_MF52_COEFF_NAMES)
    print(f"\nsaved to {DEFAULT_COMB_MF52_COEFFS_PATH}")

    for p in ref_comb:
        plot_comb_conversion(d_all, ref.lat_mf52_coeffs, src.lat_mf52_coeffs, src_long, ref_long, src_comb, ref_comb, p)
