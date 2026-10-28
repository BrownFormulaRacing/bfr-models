# exports the full 43075 mf5.2 set (pure long / lat / mz + combined) in iso signs for applied racing dynamics.
# every fit is in the ttc's sae axes (x fwd, y right, z down). iso (tydex w-axis) is that frame turned 180 deg
# about x, so fy, fz, mz and slip angle flip while fx, slip ratio and camber keep their sign.
# run from repo root: python -m src.models.tire_iso_export

import numpy as np
from scipy.io import savemat

from .tire import Tire, LAT_MF52_COEFF_NAMES, MZ_MF52_COEFF_NAMES
from .tire_converter import LONG_MF52_COEFF_NAMES, COMB_MF52_COEFF_NAMES, KPA_PER_PSI, REF_NAME, load_coeffs
from ..utils.files import DEFAULT_LONG_MF52_COEFFS_PATH, DEFAULT_COMB_MF52_COEFFS_PATH, DEFAULT_ISO_MF52_COEFFS_PATH

# terms even in slip angle (offsets, asymmetry) flip with fy / mz, terms odd in it keep their sign
ISO_FLIP = {
    "long": [],
    "lat": ["pEy3", "pEy4", "pHy1", "pHy2", "pHy3", "pVy1", "pVy2", "pVy3", "pVy4"],
    "mz": ["qDz6", "qDz7", "qDz8", "qDz9", "qEz4", "qEz5", "qHz1", "qHz2", "qHz3", "qHz4"],
    "comb": ["rHx1", "rBy3", "rVy1", "rVy2", "rVy3"],
}
GROUPS = {"long": LONG_MF52_COEFF_NAMES, "lat": LAT_MF52_COEFF_NAMES, "mz": MZ_MF52_COEFF_NAMES, "comb": COMB_MF52_COEFF_NAMES}

# .tir sections, combined terms sit with their force
TIR_SECTIONS = [
    ("LONGITUDINAL_COEFFICIENTS", [("long", LONG_MF52_COEFF_NAMES), ("comb", COMB_MF52_COEFF_NAMES[:6])]),
    ("LATERAL_COEFFICIENTS", [("lat", LAT_MF52_COEFF_NAMES), ("comb", COMB_MF52_COEFF_NAMES[6:])]),
    ("ALIGNING_COEFFICIENTS", [("mz", MZ_MF52_COEFF_NAMES)]),
]


def to_iso(coeffs, group):
    # 0.0 - v so a held-at-zero term doesn't come out as -0
    return {p: {n: 0.0 - v if n in ISO_FLIP[group] else v for n, v in c.items()} for p, c in coeffs.items()}


if __name__ == "__main__":
    tire = Tire()
    tire.load_lat_mf52_coeffs()
    tire.load_mz_mf52_coeffs()

    sae = {
        "long": load_coeffs(DEFAULT_LONG_MF52_COEFFS_PATH),
        "lat": tire.lat_mf52_coeffs,
        "mz": tire.mz_mf52_coeffs,
        "comb": load_coeffs(DEFAULT_COMB_MF52_COEFFS_PATH),
    }

    # one .tir worth of coeffs per pressure, so only pressures every group has, all on the same fnomin
    pressures = sorted(set.intersection(*(set(c) for c in sae.values())))
    for p in pressures:
        fz0s = {g: sae[g][p]["fz0"] for g in sae}
        if len(set(fz0s.values())) > 1:
            raise ValueError(f"FNOMIN differs between groups at P = {p:g} kPa: {fz0s}")

    iso = {g: to_iso(sae[g], g) for g in sae}
    r0 = sae["mz"][pressures[0]]["r0"]

    DEFAULT_ISO_MF52_COEFFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    savemat(DEFAULT_ISO_MF52_COEFFS_PATH, {
        "convention": "ISO (TYDEX W-axis): x forward, y left, z up. alpha, gamma in rad, all scaling factors 1",
        "tire": f"{REF_NAME}, 8 inch rim",
        "pressures": np.array(pressures),
        "fz0": np.array([sae["lat"][p]["fz0"] for p in pressures]),
        "r0": r0,
        **{f"{g}_names": names for g, names in GROUPS.items()},
        **{f"{g}_values": np.array([[iso[g][p][n] for n in names] for p in pressures]) for g, names in GROUPS.items()},
    })

    for p in pressures:
        print(f"\n$ {REF_NAME} @ P = {p:g} kPa ({p / KPA_PER_PSI:.0f} psi), ISO")
        print(f"FNOMIN          = {sae['lat'][p]['fz0']:.1f}")
        print(f"UNLOADED_RADIUS = {r0:.4f}")
        for section, parts in TIR_SECTIONS:
            print(f"[{section}]")
            for g, names in parts:
                for name in names:
                    print(f"{name.upper():<6} = {iso[g][p][name]: .6e}")

    print(f"\nsaved to {DEFAULT_ISO_MF52_COEFFS_PATH}")
