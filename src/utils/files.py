from pathlib import Path

# Directories
SUSPENSION_CONFIG_DIR = "config/suspension/"
AERO_CONFIG_DIR = "config/aero/"
ENGINE_CONFIG_DIR = "config/engine/"

LAT_COEFFS_DIR = "data/coeffs/lateral/"
LAT_SURFACE_DIR = "data/surfaces/lateral/"
LONG_COEFFS_DIR = "data/coeffs/longitudinal/"
COMB_COEFFS_DIR = "data/coeffs/combined/"
ISO_COEFFS_DIR = "data/coeffs/iso/"

LAT_DATA_DIR = "data/tyres/RunData_Cornering_Matlab_SI_Round9/"
LONG_DATA_DIR = "data/tyres/RunData_DriveBrake_Matlab_SI_Round9/"

# Files
DEFAULT_CONFIG_FILE = "default.json"
SKIDPAD_CONFIG_FILE = "skidpad.json"

DEFAULT_LAT_SURFACE_FILE = "hoosier_r20_lat_surface.pkl"
DEFAULT_LAT_COEFFS_FILE = "hoosier_r20_lat_surface.mat"
DEFAULT_LAT_MF52_COEFFS_FILE = "hoosier_r20_lat_mf52_coeffs.mat"
DEFAULT_LAT_MF52_TUNED_COEFFS_FILE = "hoosier_r20_lat_mf52_coeffs_tuned.mat"
DEFAULT_LONG_MF52_COEFFS_FILE = "hoosier_r20_long_mf52_coeffs.mat"
DEFAULT_MZ_MF52_COEFFS_FILE = "hoosier_r20_mz_mf52_coeffs.mat"
DEFAULT_COMB_MF52_COEFFS_FILE = "hoosier_r20_comb_mf52_coeffs.mat"
DEFAULT_ISO_MF52_COEFFS_FILE = "hoosier_r20_mf52_coeffs_iso.mat"

# Constants
DEFAULT_RUN_ID = "B2356run8"
DEFAULT_MZ_RUN_IDS = ["B2356run8", "B2356run9"]
DEFAULT_MZ_RUN_TAG = "B2356run8+9"

# Paths
SUSPENSION_DEFAULT_CONFIG_PATH = Path(Path.cwd() / f"{SUSPENSION_CONFIG_DIR}{DEFAULT_CONFIG_FILE}").expanduser().resolve()
SUSPENSION_SKIDPAD_CONFIG_PATH = Path(Path.cwd() / f"{SUSPENSION_CONFIG_DIR}{SKIDPAD_CONFIG_FILE}").expanduser().resolve()

AERO_DEFAULT_CONFIG_PATH = Path(Path.cwd() / f"{AERO_CONFIG_DIR}{DEFAULT_CONFIG_FILE}").expanduser().resolve()

DEFAULT_LAT_COEFFS_PATH = Path(Path.cwd() / f"{LAT_COEFFS_DIR}[{DEFAULT_RUN_ID}] {DEFAULT_LAT_COEFFS_FILE}").expanduser().resolve()
DEFAULT_LAT_MF52_COEFFS_PATH = Path(Path.cwd() / f"{LAT_COEFFS_DIR}[{DEFAULT_RUN_ID}] {DEFAULT_LAT_MF52_COEFFS_FILE}").expanduser().resolve()
DEFAULT_LAT_MF52_TUNED_COEFFS_PATH = Path(Path.cwd() / f"{LAT_COEFFS_DIR}[{DEFAULT_RUN_ID}] {DEFAULT_LAT_MF52_TUNED_COEFFS_FILE}").expanduser().resolve()
DEFAULT_LONG_MF52_COEFFS_PATH = Path(Path.cwd() / f"{LONG_COEFFS_DIR}[{DEFAULT_RUN_ID}] {DEFAULT_LONG_MF52_COEFFS_FILE}").expanduser().resolve()
DEFAULT_MZ_MF52_COEFFS_PATH = Path(Path.cwd() / f"{LAT_COEFFS_DIR}[{DEFAULT_MZ_RUN_TAG}] {DEFAULT_MZ_MF52_COEFFS_FILE}").expanduser().resolve()
DEFAULT_COMB_MF52_COEFFS_PATH = Path(Path.cwd() / f"{COMB_COEFFS_DIR}[{DEFAULT_RUN_ID}] {DEFAULT_COMB_MF52_COEFFS_FILE}").expanduser().resolve()
DEFAULT_ISO_MF52_COEFFS_PATH = Path(Path.cwd() / f"{ISO_COEFFS_DIR}[{DEFAULT_RUN_ID}] {DEFAULT_ISO_MF52_COEFFS_FILE}").expanduser().resolve()
DEFAULT_LAT_SURFACE_PATH = Path(Path.cwd() / f"{LAT_SURFACE_DIR}[{DEFAULT_RUN_ID}] {DEFAULT_LAT_SURFACE_FILE}").expanduser().resolve()
DEFAULT_LAT_DATA_PATH = Path(Path.cwd() / f"{LAT_DATA_DIR}{DEFAULT_RUN_ID}.mat").expanduser().resolve()
DEFAULT_MZ_DATA_PATHS = [Path(Path.cwd() / f"{LAT_DATA_DIR}{run_id}.mat").expanduser().resolve() for run_id in DEFAULT_MZ_RUN_IDS]
