# supplementer.py
# Pulls from existing public fanfic datasets to supplement your scraped data.
#
# WHY SUPPLEMENT?
# Your scraper gets fic + comments (the gold standard for your project).
# But scraping is slow and AO3 may throttle you mid-run.
# Existing datasets give you additional fic TEXT for:
#   (a) pre-training your arc encoder on a larger corpus before fine-tuning
#   (b) filling cells where AO3 throttled you before you hit 15 fics
#
# IMPORTANT: existing datasets typically have NO comments.
# So supplemented fics can only be used for arc modeling (Layer 2),
# NOT for the fic→comment bridge (Layers 3–4). Label them clearly.
#
# AVAILABLE DATASETS:
# 1. FanFicFare corpus — ~277k fics, multi-platform, no comments
#    Download: https://github.com/JimmXinu/FanFicFare (you build it yourself)
# 2. AO3 2021 data dump — ~7M works, metadata only (no full text, no comments)
#    Download: https://archiveofourown.org/admin_posts/18804
# 3. "Fanfiction.net + AO3 Corpus" (Lo et al. 2022) — NLP paper corpus, ~500k
#    Available on HuggingFace: OireachtasData/fanfic (partial)

import os
import json
import csv
import re
from config import (
    FANDOMS, TROPES,
    MIN_WORD_COUNT, FICS_PER_CELL,
    OUTPUT_DIR,
)

# ── Fandom matching ───────────────────────────────────────────────────────────
# Existing datasets use inconsistent fandom naming — we fuzzy-match.
FANDOM_ALIASES = {
    "harry_potter":  ["harry potter", "hp", "potterverse", "harry potter - j. k. rowling"],
    "twilight":      ["twilight", "twilight series", "twilight - stephenie meyer"],
    "bts":           ["bts", "bangtan", "방탄소년단", "bts (band)"],
    "one_direction": ["one direction", "1d", "one direction (band)"],
}

TROPE_ALIASES = {
    "hurt_comfort": ["hurt/comfort", "hurt comfort", "h/c", "hurtcomfort"],
    "fluff":        ["fluff", "tooth-rotting fluff", "pure fluff", "domestic fluff"],
    "whump":        ["whump", "whumped", "whumped character"],
    "slow_burn":    ["slow burn", "slowburn", "slow-burn", "slow build"],
}


def match_fandom(text: str) -> str | None:
    text = text.lower().strip()
    for key, aliases in FANDOM_ALIASES.items():
        if any(alias in text for alias in aliases):
            return key
    return None


def match_trope(tags: list[str]) -> str | None:
    tags_lower = [t.lower().strip() for t in tags]
    for key, aliases in TROPE_ALIASES.items():
        if any(alias in tag for tag in tags_lower for alias in aliases):
            return key
    return None


# ── Count existing scraped fics per cell ─────────────────────────────────────
def count_existing() -> dict[tuple, int]:
    counts = {}
    for fandom_key in FANDOMS:
        for trope_key in TROPES:
            cell_dir = os.path.join(OUTPUT_DIR, fandom_key, trope_key)
            if os.path.isdir(cell_dir):
                n = len([f for f in os.listdir(cell_dir) if f.endswith(".json")])
            else:
                n = 0
            counts[(fandom_key, trope_key)] = n
    return counts


# ── Load from AO3 2021 metadata dump ─────────────────────────────────────────
def load_ao3_dump(dump_csv_path: str):
    """
    The AO3 2021 data dump is a CSV with columns:
      work_id, title, author, fandom, tags, kudos, comments, words, complete
    It has NO full text — only metadata.

    We use it to: (a) identify promising work IDs, then (b) scrape just those.
    This is much more efficient than blind searching.
    """
    if not os.path.exists(dump_csv_path):
        print(f"AO3 dump not found at {dump_csv_path} — skipping.")
        return []

    print(f"Loading AO3 dump from {dump_csv_path}...")
    candidates = []

    with open(dump_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fandom_key = match_fandom(row.get("fandom", ""))
            if fandom_key is None:
                continue

            tags = row.get("tags", "").split(",")
            trope_key = match_trope(tags)
            if trope_key is None:
                continue

            try:
                kudos    = int(row.get("kudos", 0) or 0)
                comments = int(row.get("comments", 0) or 0)
                words    = int(row.get("words", 0) or 0)
                complete = row.get("complete", "").lower() in ("true", "1", "yes", "t")
            except ValueError:
                continue

            if kudos < 500 or comments < 20 or words < MIN_WORD_COUNT or not complete:
                continue

            candidates.append({
                "work_id":    row.get("work_id", ""),
                "fandom_key": fandom_key,
                "trope_key":  trope_key,
                "source":     "ao3_dump",
                "kudos":      kudos,
                "comments":   comments,
                "words":      words,
                "has_text":   False,    # dump has no text — need to scrape
                "has_comments": False,
            })

    print(f"  Found {len(candidates)} candidates in dump.")
    return candidates


# ── Load from FanFicFare corpus ───────────────────────────────────────────────
def load_fanficfare(corpus_dir: str):
    """
    FanFicFare stores fics as individual .txt or .epub files in subdirectories.
    Structure: corpus_dir/fandom_name/story_title.txt

    Since it has no comments, records are tagged has_comments=False.
    Use only for arc encoder pre-training, not for the bridge model.
    """
    if not os.path.isdir(corpus_dir):
        print(f"FanFicFare corpus not found at {corpus_dir} — skipping.")
        return []

    print(f"Loading FanFicFare corpus from {corpus_dir}...")
    candidates = []

    for fandom_dir in os.listdir(corpus_dir):
        fandom_key = match_fandom(fandom_dir)
        if fandom_key is None:
            continue

        fandom_path = os.path.join(corpus_dir, fandom_dir)
        for fname in os.listdir(fandom_path):
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(fandom_path, fname)
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                text = f.read()

            words = len(text.split())
            if words < MIN_WORD_COUNT:
                continue

            # FanFicFare txt files often start with metadata lines
            # Attempt to extract tags from the header
            tags = re.findall(r"^Tags?:(.+)$", text[:2000], re.MULTILINE | re.IGNORECASE)
            tag_list = tags[0].split(",") if tags else []
            trope_key = match_trope(tag_list)
            if trope_key is None:
                continue

            candidates.append({
                "work_id":      fname.replace(".txt", ""),
                "fandom_key":   fandom_key,
                "trope_key":    trope_key,
                "source":       "fanficfare",
                "words":        words,
                "text":         text,
                "has_text":     True,
                "has_comments": False,   # no comments in FanFicFare
            })

    print(f"  Found {len(candidates)} candidates in FanFicFare.")
    return candidates


# ── Main supplement logic ─────────────────────────────────────────────────────
def supplement(ao3_dump_path: str = None, fanficfare_dir: str = None):
    """
    Fill cells that are under FICS_PER_CELL after scraping.
    Supplemented records are saved with source='supplement' so you can
    exclude them from bridge training (they have no comments).
    """
    existing = count_existing()
    gaps     = {cell: FICS_PER_CELL - n for cell, n in existing.items() if n < FICS_PER_CELL}

    if not gaps:
        print("All cells are full — no supplementation needed.")
        return

    print(f"\nCells needing supplementation: {len(gaps)}")
    for (f, t), n in gaps.items():
        print(f"  {f}/{t}: need {n} more")

    # Load available supplement sources
    candidates = []
    if ao3_dump_path:
        candidates += load_ao3_dump(ao3_dump_path)
    if fanficfare_dir:
        candidates += load_fanficfare(fanficfare_dir)

    if not candidates:
        print("\nNo supplement sources provided or found.")
        print("You can still run with only scraped data.")
        print("To supplement later, call:")
        print("  supplement(ao3_dump_path='path/to/dump.csv')")
        print("  supplement(fanficfare_dir='path/to/fanficfare/')")
        return

    # Fill gaps from candidates
    filled = {cell: 0 for cell in gaps}

    for candidate in candidates:
        cell = (candidate["fandom_key"], candidate["trope_key"])
        if cell not in gaps:
            continue
        if filled[cell] >= gaps[cell]:
            continue

        # Save supplement record
        cell_dir = os.path.join(OUTPUT_DIR, cell[0], cell[1])
        os.makedirs(cell_dir, exist_ok=True)

        out_path = os.path.join(cell_dir, f"supp_{candidate['work_id']}.json")
        candidate["is_supplement"] = True   # flag — exclude from bridge training
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(candidate, f, ensure_ascii=False, indent=2)

        filled[cell] += 1

    print("\nSupplementation complete.")
    for cell, n in filled.items():
        print(f"  {cell[0]}/{cell[1]}: added {n} records")


if __name__ == "__main__":
    # Example usage — point these at your actual paths
    supplement(
        ao3_dump_path  = "data/ao3_2021_dump.csv",  # download from AO3
        fanficfare_dir = "data/fanficfare_corpus/",  # build with FanFicFare tool
    )
