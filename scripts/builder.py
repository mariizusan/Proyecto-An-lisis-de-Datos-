# builder.py
# Reads all raw JSON records, cleans them, and produces:
#   (1) data/dataset.csv    — the master index (one row per fic)
#   (2) data/pairs.jsonl    — one JSON line per fic-comment pair (model-ready)
#   (3) data/summary.txt    — dataset statistics report
#
# WHY TWO OUTPUT FORMATS?
# CSV is for inspection (open in Excel/pandas, check balance).
# JSONL is for the model — each line is a self-contained training example.
# One JSONL line = one (fic_chapter_window, comment) pair.

import os
import json
import csv
from collections import defaultdict
from config import FANDOMS, TROPES, OUTPUT_DIR, FINAL_CSV, BASE_DIR


PAIRS_FILE   = os.path.join(BASE_DIR, "data", "pairs.jsonl")
SUMMARY_FILE = os.path.join(BASE_DIR, "data", "summary.txt")


# ── Load all raw records ──────────────────────────────────────────────────────
def load_all_records() -> list[dict]:
    records = []
    for fandom_key in FANDOMS:
        for trope_key in TROPES:
            cell_dir = os.path.join(OUTPUT_DIR, fandom_key, trope_key)
            if not os.path.isdir(cell_dir):
                continue
            for fname in os.listdir(cell_dir):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(cell_dir, fname), encoding="utf-8") as f:
                    try:
                        records.append(json.load(f))
                    except json.JSONDecodeError:
                        print(f"  Bad JSON: {fname} — skipping")
    return records


# ── Clean text ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Light cleaning — preserve emotional content, remove only noise.
    We do NOT strip punctuation or lowercase here:
    RoBERTa is case-sensitive and punctuation carries emotional signal
    (ellipses, em-dashes, exclamation marks all matter in fanfic).
    """
    import re
    # Remove HTML entities that survived BeautifulSoup
    text = re.sub(r"&[a-z]+;", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Build CSV index ───────────────────────────────────────────────────────────
def build_csv(records: list[dict]):
    """
    One row per fic. Used for dataset inspection and balance checking.
    """
    os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)

    fieldnames = [
        "work_id", "fandom_key", "trope_key", "title",
        "kudos", "words", "n_chapters", "n_comments",
        "is_supplement", "has_comments", "source", "url",
        "tags",
    ]

    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            chapters = r.get("chapters", [])
            comments = r.get("comments", [])
            writer.writerow({
                "work_id":       r.get("work_id", ""),
                "fandom_key":    r.get("fandom_key", ""),
                "trope_key":     r.get("trope_key", ""),
                "title":         r.get("title", ""),
                "kudos":         r.get("kudos", 0),
                "words":         r.get("words", 0),
                "n_chapters":    len(chapters),
                "n_comments":    len(comments),
                "is_supplement": r.get("is_supplement", False),
                "has_comments":  r.get("has_comments", True) and len(comments) > 0,
                "source":        r.get("source", "ao3_scrape"),
                "url":           r.get("url", ""),
                "tags":          "|".join(r.get("tags", [])),
            })

    print(f"CSV saved: {FINAL_CSV}")


# ── Build JSONL pairs ─────────────────────────────────────────────────────────
def build_pairs(records: list[dict]):
    """
    Produces the model-ready training file.

    Each line in pairs.jsonl is one training example:
    {
      "pair_id":    unique identifier
      "work_id":    source fic
      "fandom_key": fandom
      "trope_key":  trope (this is your arc label for the classifier head)
      "fic_text":   full cleaned fic text (or chapter window)
      "comment":    one comment text
      "is_supplement": whether this fic has no real comments
    }

    NOTE ON WINDOWING:
    For now we store the full text and split it into windows during training.
    This keeps the dataset simple and lets you experiment with window sizes
    without rebuilding the dataset.
    """
    os.makedirs(os.path.dirname(PAIRS_FILE), exist_ok=True)
    pair_count = 0

    with open(PAIRS_FILE, "w", encoding="utf-8") as f:
        for r in records:
            chapters = r.get("chapters", [])
            comments = r.get("comments", [])
            is_supp  = r.get("is_supplement", False)

            # Concatenate all chapter text into one fic string
            fic_text = " ".join(
                clean_text(ch.get("text", "")) for ch in chapters
            )

            if not fic_text.strip():
                continue

            if is_supp or not comments:
                # Supplement fics: one record with no comment (arc training only)
                record = {
                    "pair_id":       f"{r['work_id']}_arc",
                    "work_id":       r.get("work_id"),
                    "fandom_key":    r.get("fandom_key"),
                    "trope_key":     r.get("trope_key"),
                    "fic_text":      fic_text,
                    "comment":       None,
                    "is_supplement": True,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pair_count += 1
            else:
                # Real scraped fics: one record per comment
                for i, comment in enumerate(comments):
                    record = {
                        "pair_id":       f"{r['work_id']}_c{i}",
                        "work_id":       r.get("work_id"),
                        "fandom_key":    r.get("fandom_key"),
                        "trope_key":     r.get("trope_key"),
                        "fic_text":      fic_text,
                        "comment":       clean_text(comment.get("text", "")),
                        "is_supplement": False,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    pair_count += 1

    print(f"JSONL pairs saved: {PAIRS_FILE}  ({pair_count} pairs)")
    return pair_count


# ── Summary report ────────────────────────────────────────────────────────────
def build_summary(records: list[dict], pair_count: int):
    """
    Prints and saves a balance report so you can spot gaps before training.
    """
    cell_counts    = defaultdict(int)
    comment_counts = defaultdict(int)
    supp_counts    = defaultdict(int)

    for r in records:
        cell = (r.get("fandom_key"), r.get("trope_key"))
        cell_counts[cell] += 1
        if not r.get("is_supplement", False):
            comment_counts[cell] += len(r.get("comments", []))
        else:
            supp_counts[cell] += 1

    lines = [
        "=" * 60,
        "DATASET SUMMARY",
        "=" * 60,
        f"Total fics:        {len(records)}",
        f"Total pairs (JSONL): {pair_count}",
        "",
        f"{'Cell':<30} {'Fics':>6} {'Supp':>6} {'Comments':>10}",
        "-" * 56,
    ]

    for fandom_key in FANDOMS:
        for trope_key in TROPES:
            cell  = (fandom_key, trope_key)
            label = f"{fandom_key}/{trope_key}"
            lines.append(
                f"{label:<30} {cell_counts[cell]:>6} "
                f"{supp_counts[cell]:>6} {comment_counts[cell]:>10}"
            )

    lines += ["=" * 60]
    report = "\n".join(lines)
    print(report)

    with open(SUMMARY_FILE, "w") as f:
        f.write(report)
    print(f"Summary saved: {SUMMARY_FILE}")


# ── Entry point ───────────────────────────────────────────────────────────────
def build():
    print("Loading raw records...")
    records = load_all_records()
    print(f"  Found {len(records)} records.")

    if not records:
        print("No records found. Run scraper.py first.")
        return

    print("Building CSV index...")
    build_csv(records)

    print("Building JSONL pairs...")
    pair_count = build_pairs(records)

    print("Building summary...")
    build_summary(records, pair_count)

    print("\nDataset build complete.")
    print(f"  Index:  {FINAL_CSV}")
    print(f"  Pairs:  {PAIRS_FILE}")
    print(f"  Report: {SUMMARY_FILE}")


if __name__ == "__main__":
    build()
