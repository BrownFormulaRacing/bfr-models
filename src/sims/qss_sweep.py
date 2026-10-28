#!/usr/bin/env python3
"""
Perpetual QSS lap time sweep runner for ARD.

    python qss_sweep.py --check
    python qss_sweep.py --show-sweeps
    python qss_sweep.py                  # uses qss_sweep.json, saves to results/qss/

Needs ard_common.py in the same folder. Run with --help for every option.
"""
from ard_common import RunnerSpec, main

SPEC = RunnerSpec(
    name="QSS",
    simulation_type="lts",       # ARD's lap time (QSS) type, as used in the API examples
    needs_track=True,
    config_file="qss_sweep.json",
    results_dir="results/qss",
)

if __name__ == "__main__":
    main(SPEC)