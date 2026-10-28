"""
Shared engine for the ARD sweep runners (qss_sweep.py, ymd_sweep.py, step_steer_sweep.py).

Keep this file in the same folder as the runners. Each runner only says which simulation
type it sweeps, whether that type needs a track, and which config/results folder to use.

Common commands (work with every runner):
    --check                          verify API key, show plan/credits
    --schema                         list ARD's simulation types and POST /sweeps fields (no key needed)
    --list-presets                   saved simulation/sweep configs from the app
    --list-params VEHICLE AMBIENT    sweepable parameter paths (add --search spring)
    --show-sweeps                    print each sweep after merging with defaults (no key needed)
"""
import argparse
import copy
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "https://api.appliedracingdynamics.com/external-api"
OPENAPI_URL = f"{BASE_URL}/openapi.json"
DONE_STATES = {"complete", "completed", "finished", "success", "succeeded", "done"}
FAILED_STATES = {"failed", "error", "errored", "cancelled", "canceled"}

log = logging.getLogger("ard")
STOP = False


@dataclass
class RunnerSpec:
    name: str              # shown in logs and run names, e.g. "QSS"
    simulation_type: str   # ARD type each sweep combination runs, e.g. "lts", "ymd"
    needs_track: bool      # whether this type uses a track
    config_file: str       # default config path
    results_dir: str       # default output folder


# ---------------------------------------------------------------- stop handling

def _handle_stop(signum, frame):
    global STOP
    if STOP:
        log.warning("Second interrupt: exiting immediately.")
        sys.exit(1)
    STOP = True
    log.info("Stop requested. No new sweeps will start; running ones are finished (Ctrl+C again to force).")


# ---------------------------------------------------------------- API client

class ARDError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status

    @property
    def is_config_error(self):
        """A 4xx other than 429 means the request itself is wrong; retrying won't help."""
        return self.status is not None and 400 <= self.status < 500 and self.status != 429


class ARDClient:
    def __init__(self, api_key, timeout=60):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self.timeout = timeout

    def request(self, method, path, max_retries=8, **kwargs):
        url = f"{BASE_URL}{path}"
        backoff = 5
        last_status = None
        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as e:
                log.warning("Network error on %s %s (%s); retry %d in %ds", method, path, e, attempt, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            last_status = r.status_code
            # ARD refuses with 429 + Retry-After when too many sims are running in the team.
            if r.status_code == 429:
                wait = _retry_after(r, backoff)
                log.info("Concurrency/rate limit hit (429); waiting %ds", wait)
                time.sleep(wait)
                backoff = min(backoff * 2, 300)
                continue
            if r.status_code >= 500:
                log.warning("Server error %d on %s %s; retry %d in %ds", r.status_code, method, path, attempt, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            if r.status_code >= 400:
                raise ARDError(f"{method} {path} -> {r.status_code}: {r.text}", status=r.status_code)
            return r.json() if r.content else {}
        raise ARDError(f"{method} {path}: gave up after {max_retries} attempts",
                       status=last_status if last_status == 429 else None)

    def test_auth(self):            return self.request("POST", "/test-auth")
    def me(self):                   return self.request("GET", "/me")
    def presets(self):              return self.request("GET", "/simulation-configs")
    def current_vehicle(self):      return self.request("GET", "/current-vehicle")
    def vehicle_by_name(self, n):   return self.request("GET", "/vehicle", params={"name": n})
    def run_sweep(self, body):      return self.request("POST", "/sweeps", json=body)
    def status(self, sim_id):       return self.request("GET", f"/simulation-status/{sim_id}")
    def sweep_table(self, sim_id):  return self.request("GET", f"/simulations/{sim_id}/sweep-table")
    def cancel(self, sim_id):       return self.request("POST", f"/simulations/{sim_id}/cancel")

    def sweepable_params(self, vehicle_id, ambient_id, search=None, kind="all"):
        params = {"ambient_conditions_id": ambient_id, "kind": kind}
        if search:
            params["search"] = search
        return self.request("GET", f"/vehicles/{vehicle_id}/sweepable-parameters", params=params)


# ---------------------------------------------------------------- helpers

def _retry_after(response, default):
    try:
        return max(1, int(float(response.headers.get("Retry-After", default))))
    except (TypeError, ValueError):
        return default


def find_key(obj, keys):
    """Depth-first search for the first matching key in nested dicts/lists."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        for v in obj.values():
            found = find_key(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_key(item, keys)
            if found is not None:
                return found
    return None


def strip_comments(obj):
    """Drop keys starting with '_' so config files can carry notes."""
    if isinstance(obj, dict):
        return {k: strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_comments(v) for v in obj]
    return obj


def deep_merge(base, override):
    """Dicts merge key by key; anything else (including lists) in override replaces base."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def safe_name(label):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "sweep"


def interruptible_sleep(seconds):
    end = time.time() + seconds
    while not STOP and time.time() < end:
        time.sleep(min(1.0, max(0.0, end - time.time())))


def classify_status(st):
    state = str(find_key(st, ["status", "state"]) or "").lower().replace("_", " ").strip()
    if state in DONE_STATES or state.replace(" ", "") in DONE_STATES:
        return "done", state
    if state in FAILED_STATES:
        return "failed", state
    return "running", state


# ---------------------------------------------------------------- public schema (no key needed)

def fetch_schema():
    r = requests.get(OPENAPI_URL, timeout=60)
    r.raise_for_status()
    return r.json().get("components", {}).get("schemas", {})


def simulation_types(schemas):
    return schemas.get("SimulationType", {}).get("enum")


def print_schema():
    schemas = fetch_schema()
    print("Simulation types ARD accepts:")
    for t in simulation_types(schemas) or ["(SimulationType enum not found)"]:
        print(f"  {t}")
    sweep_in = schemas.get("SweepIn", {})
    print("\nPOST /sweeps (SweepIn) fields:")
    required = set(sweep_in.get("required", []))
    for name, prop in sweep_in.get("properties", {}).items():
        kind = prop.get("type") or prop.get("$ref", "").split("/")[-1] or "see docs"
        print(f"  {name:28s} {kind}{'  (required)' if name in required else ''}")
        if prop.get("description"):
            print(f"      {prop['description'].splitlines()[0]}")


def check_simulation_type(sim_type, force=False):
    """Make sure ARD actually knows this simulation type before spending anything."""
    try:
        valid = simulation_types(fetch_schema())
    except Exception as e:  # schema check is a convenience; don't block on it
        log.warning("Could not fetch ARD's schema to check the simulation type (%s); continuing.", e)
        return
    if not valid:
        log.warning("ARD's schema did not list simulation types; continuing without the check.")
        return
    if sim_type not in valid:
        msg = (f"Simulation type {sim_type!r} is not one ARD accepts. Valid types: {', '.join(valid)}. "
               f"Set 'simulation_type' in the config to the right one.")
        if force:
            log.warning(msg + " Continuing because of --force.")
        else:
            sys.exit(msg)


# ---------------------------------------------------------------- config

def resolve_presets(client, sweeps):
    """Fill simulation_config / sweep_parameters from presets saved in the ARD app, by name."""
    wanted = any(s.get("simulation_config_preset") or s.get("sweep_preset") for s in sweeps)
    if not wanted:
        return
    presets = client.presets()
    by_name = {}
    for p in presets if isinstance(presets, list) else []:
        name = p.get("name") or p.get("title")
        if name:
            by_name[name] = p

    def lookup(name, label):
        if name not in by_name:
            sys.exit(f"[{label}] preset {name!r} not found. Available: {', '.join(sorted(by_name)) or 'none'}")
        return by_name[name]

    for s in sweeps:
        if s.get("simulation_config_preset"):
            preset = lookup(s["simulation_config_preset"], s["label"])
            cfg = preset.get("config")
            if cfg is None:
                sys.exit(f"[{s['label']}] preset has no 'config': {json.dumps(preset)[:300]}")
            s["request"]["simulation_config"] = cfg
            log.info("[%s] simulation_config from preset %r", s["label"], s["simulation_config_preset"])
        if s.get("sweep_preset"):
            preset = lookup(s["sweep_preset"], s["label"])
            dims = find_key(preset, ["sweep_parameters", "dimensions"])
            if dims is None:
                sys.exit(f"[{s['label']}] could not find sweep dimensions in preset: {json.dumps(preset)[:300]}")
            s["request"]["sweep_parameters"] = dims
            log.info("[%s] sweep_parameters from preset %r", s["label"], s["sweep_preset"])


def build_sweeps(cfg, spec):
    defaults = cfg.get("defaults", {})
    raw = cfg.get("sweeps") or []
    type_field = cfg.get("type_field", "sweep_simulation_type")
    sim_type = cfg.get("simulation_type", spec.simulation_type)

    sweeps, seen = [], set()
    for i, entry in enumerate(raw, start=1):
        if not entry.get("enabled", True):
            continue
        label = entry.get("label") or f"sweep_{i}"
        if label in seen:
            sys.exit(f"Duplicate sweep label in config: {label!r}")
        seen.add(label)
        sweep = deep_merge(defaults, entry)
        sweep["label"] = label
        req = sweep.setdefault("request", {})
        req.setdefault(type_field, sim_type)

        track_id = (req.get("track") or {}).get("id")
        if spec.needs_track and not track_id:
            sys.exit(f"[{label}] {spec.name} needs a track: set request.track.id")
        if not spec.needs_track and "track" in req and not track_id:
            del req["track"]  # placeholder track not needed for this type
        sweeps.append(sweep)
    if not sweeps:
        sys.exit("No enabled sweeps in config.")
    return sweeps, sim_type


def resolve_vehicle_id(client, sweep, cache):
    """Fixed id, latest version by name, or whatever is selected in the app. Cached per cycle."""
    if sweep.get("use_current_vehicle"):
        key, fetch = ("current",), client.current_vehicle
    elif sweep.get("vehicle_name"):
        key, fetch = ("name", sweep["vehicle_name"]), (lambda: client.vehicle_by_name(sweep["vehicle_name"]))
    else:
        vid = (sweep["request"].get("vehicle") or {}).get("id")
        if not vid:
            raise ARDError(f"Sweep {sweep['label']!r} has no vehicle_name and no request.vehicle.id")
        return vid
    if key not in cache:
        v = fetch()
        vid = v.get("id") if isinstance(v, dict) else None
        if vid is None:
            raise ARDError(f"Could not resolve a vehicle id from: {json.dumps(v)[:300]}")
        cache[key] = vid
    return cache[key]


# ---------------------------------------------------------------- running

def run_batch(client, due, cfg, spec, out_dir, last_vehicle, disabled):
    """Submit due sweeps (up to max_parallel at once), poll them, save results."""
    max_parallel = max(1, int(cfg.get("max_parallel", 1)))
    poll_s = cfg.get("poll_seconds", 15)
    max_wait_s = cfg.get("max_wait_seconds", 6 * 3600)
    cancel_on_stop = cfg.get("cancel_on_stop", False)
    prefix = cfg.get("name_prefix", f"Auto {spec.name}")

    queue = list(due)
    running = {}

    while queue or running:
        while queue and len(running) < max_parallel and not STOP:
            sweep, vid = queue.pop(0)
            label = sweep["label"]
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            body = copy.deepcopy(sweep["request"])
            body["vehicle"] = {**(body.get("vehicle") or {}), "id": vid}
            body["name"] = f"{prefix} | {label} | {stamp}"  # names must be unique in ARD

            log.info("[%s] submitting on vehicle %s", label, vid)
            try:
                resp = client.run_sweep(body)
            except ARDError as e:
                if e.is_config_error:
                    log.error("[%s] rejected, disabling it for this run: %s", label, e)
                    disabled.add(label)
                else:
                    log.warning("[%s] could not submit (%s); will retry next cycle", label, e)
                continue

            sim_id = find_key(resp, ["simulation_id", "id"])
            if sim_id is None:
                log.error("[%s] no simulation id in response: %s", label, json.dumps(resp)[:500])
                continue
            running[sim_id] = {"sweep": sweep, "vehicle_id": vid, "body": body,
                               "stamp": stamp, "started": time.time(), "state": None}

        if not running:
            break

        time.sleep(poll_s)

        for sim_id in list(running):
            job = running[sim_id]
            label = job["sweep"]["label"]
            try:
                outcome, state = classify_status(client.status(sim_id))
            except ARDError as e:
                log.warning("[%s] status check failed (%s); will retry", label, e)
                continue
            if state != job["state"]:
                log.info("[%s] sim %s status: %s", label, sim_id, state or "?")
                job["state"] = state

            if outcome == "done":
                try:
                    table = client.sweep_table(sim_id)
                except ARDError as e:
                    log.error("[%s] could not fetch sweep table for sim %s: %s", label, sim_id, e)
                    del running[sim_id]
                    continue
                folder = out_dir / safe_name(label)
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"sweep_{sim_id}_{job['stamp']}.json"
                path.write_text(json.dumps({
                    "runner": spec.name, "label": label, "simulation_id": sim_id,
                    "vehicle_id": job["vehicle_id"], "request": job["body"], "sweep_table": table,
                }, indent=2))
                log.info("[%s] saved %s", label, path)
                last_vehicle[label] = job["vehicle_id"]
                del running[sim_id]
            elif outcome == "failed":
                log.error("[%s] sim %s ended as '%s'", label, sim_id, state)
                del running[sim_id]
            elif STOP and cancel_on_stop:
                log.info("[%s] cancelling sim %s on request", label, sim_id)
                try:
                    client.cancel(sim_id)
                except ARDError as e:
                    log.warning("[%s] cancel failed: %s", label, e)
                del running[sim_id]
            elif time.time() - job["started"] > max_wait_s:
                log.error("[%s] sim %s exceeded %ds; leaving it running in ARD", label, sim_id, max_wait_s)
                del running[sim_id]


def run_forever(client, cfg, spec, sweeps, out_dir):
    interval = cfg.get("interval_seconds", 60)
    max_cycles = cfg.get("max_cycles")
    skip_unchanged = cfg.get("skip_if_vehicle_unchanged", True)
    last_vehicle, disabled = {}, set()
    cycle = 0

    while not STOP:
        if max_cycles and cycle >= max_cycles:
            log.info("Reached max_cycles=%d", max_cycles)
            break

        cache, due = {}, []
        for sweep in sweeps:
            label = sweep["label"]
            if label in disabled:
                continue
            try:
                vid = resolve_vehicle_id(client, sweep, cache)
            except ARDError as e:
                log.error("[%s] %s", label, e)
                continue
            # Same vehicle version + same sweep = same answer, so don't spend credits re-running it.
            if skip_unchanged and last_vehicle.get(label) == vid:
                continue
            due.append((sweep, vid))

        if len(disabled) == len(sweeps):
            log.error("Every sweep has been rejected by the API; fix the config and restart.")
            break

        if due:
            cycle += 1
            log.info("%s cycle %d: %d sweep(s) due: %s", spec.name, cycle, len(due),
                     ", ".join(s["label"] for s, _ in due))
            run_batch(client, due, cfg, spec, out_dir, last_vehicle, disabled)
        else:
            log.info("No %s sweeps due (vehicles unchanged); waiting %ds", spec.name, interval)

        if not STOP:
            interruptible_sleep(interval)

    log.info("%s runner stopped after %d cycle(s).", spec.name, cycle)


# ---------------------------------------------------------------- entry point

def main(spec: RunnerSpec):
    p = argparse.ArgumentParser(description=f"Perpetual ARD {spec.name} sweep runner")
    p.add_argument("--config", default=spec.config_file)
    p.add_argument("--out", default=spec.results_dir)
    p.add_argument("--check", action="store_true", help="verify API key and print plan/credits")
    p.add_argument("--schema", action="store_true", help="list simulation types and POST /sweeps fields")
    p.add_argument("--list-presets", action="store_true", help="print saved simulation/sweep configs")
    p.add_argument("--list-params", nargs=2, metavar=("VEHICLE_ID", "AMBIENT_ID"),
                   help="print sweepable parameter paths for a vehicle")
    p.add_argument("--search", help="filter for --list-params (e.g. 'spring', 'wing')")
    p.add_argument("--show-sweeps", action="store_true", help="print each sweep after merging with defaults")
    p.add_argument("--force", action="store_true", help="run even if the simulation type check fails")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s [{spec.name}] %(message)s")

    if args.schema:
        print_schema()
        return
    if args.show_sweeps:
        cfg = strip_comments(json.loads(Path(args.config).read_text()))
        print(json.dumps(build_sweeps(cfg, spec)[0], indent=2))
        return

    api_key = os.environ.get("ARD_API_KEY")
    if not api_key:
        sys.exit("Set the ARD_API_KEY environment variable first.")
    client = ARDClient(api_key)

    if args.check:
        client.test_auth()
        print(json.dumps(client.me(), indent=2))
        return
    if args.list_presets:
        print(json.dumps(client.presets(), indent=2))
        return
    if args.list_params:
        vid, aid = (int(x) for x in args.list_params)
        print(json.dumps(client.sweepable_params(vid, aid, search=args.search), indent=2))
        return

    cfg = strip_comments(json.loads(Path(args.config).read_text()))
    sweeps, sim_type = build_sweeps(cfg, spec)
    check_simulation_type(sim_type, force=args.force)
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    client.test_auth()
    resolve_presets(client, sweeps)
    log.info("Authenticated. Account: %s", json.dumps(client.me())[:300])
    log.info("Loaded %d sweep(s) of type %r: %s", len(sweeps), sim_type, ", ".join(s["label"] for s in sweeps))
    run_forever(client, cfg, spec, sweeps, Path(args.out))