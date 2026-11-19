"""Study 1 paths, set by the command-line runner."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ["KUAIRAND_DATA_DIR"])
RESULT_DIR = Path(os.environ["S1_RESULT_DIR"])
PANEL_DIR = RESULT_DIR / "panels"
FIGURE_DIR = RESULT_DIR / "figures"

PRE_FILE = DATA_DIR / "log_standard_4_08_to_4_21_1k.csv"
POST_FILE = DATA_DIR / "log_standard_4_22_to_5_08_1k.csv"
RAND_FILE = DATA_DIR / "log_random_4_22_to_5_08_1k.csv"
VIDEO_FILE = DATA_DIR / "video_features_basic_1k.csv"
CHUNK_SIZE = 400_000
OUTCOMES = ["nov_rate_is_click", "nov_rate_long_view", "nov_rate_watch_ratio", "preference_gap_is_click"]
CONTROLS = ["log_opp", "lag_activity", "baseline_hhi_post", "baseline_activity_post"]
