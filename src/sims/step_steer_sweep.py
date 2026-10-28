#!/usr/bin/env python3
"""
Perpetual step steer sweep runner for ARD.

ARD's public docs don't show the step steer type name or its settings, so before the
first run:
    python step_steer_sweep.py --schema         # lists every simulation type ARD accepts
Then set "simulation_type" in step_steer_sweep.json to the step steer entry, and either
fill in simulation_config or point simulation_config_preset at a step steer config you
saved in the app (python step_steer_sweep.py --list-presets shows them).

The runner checks the type against ARD's schema at startup and refuses to start if
ARD doesn't recognise it, so a wrong guess costs nothing.

    python step_steer_sweep.py           # uses step_steer_sweep.json, saves to results/step_steer/

Needs ard_common.py in the same folder.
"""
from ard_common import RunnerSpec, main

SPEC = RunnerSpec(
    name="Step steer",
    simulation_type="step_steer",   # placeholder: confirm with --schema, override in the config
    needs_track=False,
    config_file="step_steer_sweep.json",
    results_dir="results/step_steer",
)

if __name__ == "__main__":
    main(SPEC)