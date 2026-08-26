from pathlib import Path

# Directories
SUSPENSION_CONFIG_DIR = "config/suspension/"
AERO_CONFIG_DIR = "config/aero/"
ENGINE_CONFIG_DIR = "config/engine/"

LAT_COEFFS_DIR = "data/coeffs/lateral/"
LAT_SURFACE_DIR = "data/surfaces/lateral/"

LAT_DATA_DIR = "data/tyres/RunData_Cornering_Matlab_SI_Round9/"

# Files
DEFAULT_CONFIG_FILE = "default.json"
SKIDPAD_CONFIG_FILE = "skidpad.json"

DEFAULT_LAT_SURFACE_FILE = "hoosier_r20_lat_surface.pkl"
DEFAULT_LAT_COEFFS_FILE = "hoosier_r20_lat_surface.mat"

# Constants
DEFAULT_RUN_ID = "B2356run8"

# Paths
SUSPENSION_DEFAULT_CONFIG_PATH = Path(Path.cwd() / f"{SUSPENSION_CONFIG_DIR}{DEFAULT_CONFIG_FILE}").expanduser().resolve()
SUSPENSION_SKIDPAD_CONFIG_PATH = Path(Path.cwd() / f"{SUSPENSION_CONFIG_DIR}{SKIDPAD_CONFIG_FILE}").expanduser().resolve()

AERO_DEFAULT_CONFIG_PATH = Path(Path.cwd() / f"{AERO_CONFIG_DIR}{DEFAULT_CONFIG_FILE}").expanduser().resolve()

DEFAULT_LAT_COEFFS_PATH = Path(Path.cwd() / f"{LAT_COEFFS_DIR}[{DEFAULT_RUN_ID}] {DEFAULT_LAT_COEFFS_FILE}").expanduser().resolve()
DEFAULT_LAT_SURFACE_PATH = Path(Path.cwd() / f"{LAT_SURFACE_DIR}[{DEFAULT_RUN_ID}] {DEFAULT_LAT_SURFACE_FILE}").expanduser().resolve()
DEFAULT_LAT_DATA_PATH = Path(Path.cwd() / f"{LAT_DATA_DIR}{DEFAULT_RUN_ID}.mat").expanduser().resolve()
