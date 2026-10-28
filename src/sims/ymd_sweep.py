#!/usr/bin/env python3
"""
Perpetual yaw moment diagram (YMD) sweep runner for ARD.

    python ymd_sweep.py --check
    python ymd_sweep.py --show-sweeps
    python ymd_sweep.py                  # uses ymd_sweep.json, saves to results/ymd/

YMD doesn't use a track. Needs ard_common.py in the same folder.
"""
from ard_common import RunnerSpec, main

SPEC = RunnerSpec(
    name="YMD",
    simulation_type="ymd",       # as used in ARD's YMD API example
    needs_track=False,
    config_file="ymd_sweep.json",
    results_dir="results/ymd",
)

if __name__ == "__main__":
    main(SPEC)