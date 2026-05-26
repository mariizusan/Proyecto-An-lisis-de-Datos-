# config.py
# Central control panel for your dataset.
# Change values here — nowhere else needs to change.
#
# NOTE: selection filters (chapter count, avg words, comments per chapter)
# live in scraper1.py since they are scraper-specific constants.

import os

# ── Fandoms ───────────────────────────────────────────────────────────────────
FANDOMS = {
    "harry_potter":  "Harry Potter - J. K. Rowling",
    "bts":           "방탄소년단 | Bangtan Boys | BTS",
    "one_direction": "One Direction (Band)",
    
}

# ── Tropes ────────────────────────────────────────────────────────────────────
TROPES = {
    "hurt_comfort": {
        "ao3_tag":      "Hurt/Comfort",
        "label":        "hurt_comfort",
        "expected_arc": "U-shape",        # distress → resolution
    },
    "fluff": {
        "ao3_tag":      "Fluff",
        "label":        "fluff",
        "expected_arc": "flat-positive",  # warm throughout
    },


    "slow_burn": {
        "ao3_tag":      "Slow Burn",
        "label":        "slow_burn",
        "expected_arc": "rising",         # tension builds to climax
    },
}

# ── Quality filters ───────────────────────────────────────────────────────────
MIN_KUDOS         = 500   # proxy for broad emotional resonance
MIN_COMMENT_WORDS = 15    # filters out "great fic!" noise
MIN_WORD_COUNT = 1500

# ── Scale targets ─────────────────────────────────────────────────────────────
# 4 fandoms × 4 tropes × 15 fics = 240 fics total
FICS_PER_CELL  = 30
TOP_N_COMMENTS = 20       # top N comments per chapter, ranked by length

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Do NOT lower REQUEST_DELAY_SEC — AO3 will IP-block you.
REQUEST_DELAY_SEC = 5
MAX_RETRIES       = 3

# ── Output paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
FINAL_CSV  = os.path.join(BASE_DIR, "data", "dataset.csv")
CHECKPOINT = os.path.join(BASE_DIR, "data", "checkpoint.json")

