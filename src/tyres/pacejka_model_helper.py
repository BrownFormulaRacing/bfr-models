import os
import glob
import numpy as np
from scipy.io import loadmat
from pathlib import Path

RUN_FILE_PATTERN_MATCH = "B2356run*.mat"

def _mat_str(val):
    """Coerce a loadmat string field (char array / object array) to a plain str."""
    arr = np.asarray(val)
    return str(arr.item()).strip() if arr.size == 1 else str(arr).strip()

def load_data(dir, tire_id_substrings, start_idx=1500):
    """
    Load and filter all matching TTC cornering runs from TYRE_DATA_PATH.
    Returns a dict of 1D numpy arrays: SA [deg], FY [N], FZ [N, positive], IA [deg].
    """
    data_dict = dict()

    # Step 1: Read the relevant files
    run_files = sorted(glob.glob(os.path.join(dir, RUN_FILE_PATTERN_MATCH)))
    if not run_files:
        raise FileNotFoundError(f"no {RUN_FILE_PATTERN_MATCH} files found in '{dir}'")

    # Step 2: Iterating through all relevant files
    for fpath in run_files:

        # Step 3: Proceed with relevant tyres
        info = loadmat(fpath, variable_names=["tireid"])
        tire_str = _mat_str(info["tireid"])
        if not all(sub in tire_str for sub in tire_id_substrings):
            continue

        d = loadmat(fpath) #, variable_names=["SA", "SR", "FX", "FY", "FZ", "IA", "P"])

        # NOTE: Prints the keys out
        # print(list(d.keys()))

        # Step 4: Pull data AND pre-process by removing cold-to-warm break-in
        SL_all = np.asarray(d["SR"]).squeeze()[start_idx:]
        SA_all = np.asarray(d["SA"]).squeeze()[start_idx:]
        FX_all = np.asarray(d["FX"]).squeeze()[start_idx:]
        FY_all = np.asarray(d["FY"]).squeeze()[start_idx:]
        FZ_all = np.asarray(d["FZ"]).squeeze()[start_idx:]
        IA_all = np.asarray(d["IA"]).squeeze()[start_idx:]
        P_all = np.asarray(d["P"]).squeeze()[start_idx:]
        FZ_pos = np.abs(FZ_all)  # TTC stores FZ negative

        print(f"  loaded: {os.path.basename(fpath)}  ({tire_str})")

        # Step 5: 
        data_dict[Path(fpath).stem] = { "SL": SL_all, "SA": SA_all, "FX": FX_all, "FY": FY_all, "FZ": FZ_pos, "IA": IA_all, "P": P_all}

    return data_dict


