# scraper_v2.py
# Rebuilt with chapter-level design.
#
# UNIT OF ANALYSIS: one chapter + comments left on that specific chapter.
#
# SELECTION CRITERIA:
#   - complete works only
#   - 1–15 chapters total
#   - average words per chapter: 2,500–5,000
#   - sorted by kudos descending (most impactful first)
#   - minimum 3 comments per chapter on average
#
# OUTPUT STRUCTURE (one JSON per fic):
# {
#   work_id, fandom_key, trope_key, title, kudos, tags, url,
#   chapters: [
#     {
#       index, chapter_id, title, text, word_count,
#       comments: [ {text, word_count} ]
#     }
#   ]
# }

import os
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlencode, quote
from config import (
    FANDOMS, TROPES,
    MIN_KUDOS, MIN_COMMENT_WORDS,
    FICS_PER_CELL, TOP_N_COMMENTS,
    REQUEST_DELAY_SEC, MAX_RETRIES,
    OUTPUT_DIR, CHECKPOINT,
)

# ── New selection constants ───────────────────────────────────────────────────
MIN_CHAPTERS          = 1
MAX_CHAPTERS          = 25
MIN_AVG_WORDS         = 1500    # multi-chapter only
MAX_AVG_WORDS         = 40000    # multi-chapter only
MAX_ONESHOT_WORDS     = 100000   # oneshot total word ceiling
MIN_COMMENTS_PER_CHAPTER = 3


# ── HTTP session ──────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "FanficResearchScraper/1.0 (academic NLP project)"
})
SESSION.cookies.update({
    "accepted_tos": "20180523",
    "view_adult":   "true",
})

BASE_URL = "https://archiveofourown.org"


# ── Core HTTP helper ──────────────────────────────────────────────────────────
def fetch(url: str) -> BeautifulSoup | None:
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY_SEC + random.uniform(0, 2))
            resp = SESSION.get(url, timeout=120)

            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")

            print(f"  HTTP {resp.status_code} on {url}")
            return None

        except requests.RequestException as e:
            print(f"  Request error (attempt {attempt+1}): {e}")
            time.sleep(10)
            continue   # retry same URL

    return None


# ── AO3 tag encoding ──────────────────────────────────────────────────────────
def ao3_encode_tag(tag: str) -> str:
    tag = tag.replace(".", "*d*")
    tag = tag.replace("/", "*s*")
    tag = tag.replace("&", "*a*")
    tag = tag.replace("#", "*h*")
    tag = tag.replace("?", "*q*")
    return quote(tag, safe="*")


# ── Search ────────────────────────────────────────────────────────────────────
def build_search_url(fandom_tag: str, trope_tag: str, page: int = 1) -> str:
    fandom_encoded = ao3_encode_tag(fandom_tag)
    params = {
        "work_search[other_tag_names]": trope_tag,
        "work_search[complete]":        "T",
        "work_search[sort_column]":     "kudos_count",
        "work_search[sort_direction]":  "desc",
        "page":                         page,
    }
    return f"{BASE_URL}/tags/{fandom_encoded}/works?{urlencode(params)}"


def search_works(fandom_tag: str, trope_tag: str, n: int) -> list[dict]:
    """
    Collect work stubs sorted by kudos.
    Uses 4× buffer because chapter/word filters will reject many.
    Retries the same page on failure rather than skipping.
    """
    results = []
    page    = 1
    target  = n * 4

    print(f"  Searching: {fandom_tag} + {trope_tag}")

    while len(results) < target:
        url  = build_search_url(fandom_tag, trope_tag, page)
        print(f"  URL: {url}")
        soup = fetch(url)

        if soup is None:
            print(f"  Page {page} failed — retrying")
            time.sleep(30)
            continue   # retry same page

        works = soup.select("li.work.blurb")
        if not works:
            break      # genuinely no more results

        for work in works:
            stub = parse_work_stub(work)
            if stub:
                results.append(stub)

        page += 1

    return results[:target]


def parse_work_stub(work_tag) -> dict | None:
    """
    Extract metadata from search result blurb.
    Pre-filters by kudos only — chapter/word filters happen after full fetch
    because chapter count isn't visible on search pages.
    """
    try:
        heading = work_tag.select_one("h4.heading a[href^='/works/']")
        if not heading:
            return None
        work_id = heading["href"].split("/works/")[1].split("/")[0]

        stats    = work_tag.select_one("dl.stats")
        kudos    = _int(stats, "dd.kudos")
        words    = _int(stats, "dd.words")
        comments = _int(stats, "dd.comments")

        # Only pre-filter by kudos — chapter/word checks need the full page
        if kudos < MIN_KUDOS:
            return None

        return {
            "work_id":  work_id,
            "title":    heading.get_text(strip=True),
            "kudos":    kudos,
            "words":    words,
            "comments": comments,
            "url":      f"{BASE_URL}/works/{work_id}",
        }

    except Exception:
        return None


def _int(parent, selector: str) -> int:
    if parent is None:
        return 0
    el = parent.select_one(selector)
    if el is None:
        return 0
    text = el.get_text(strip=True).replace(",", "").replace(".", "")
    return int(text) if text.isdigit() else 0


# ── Fetch chapter IDs ─────────────────────────────────────────────────────────
def fetch_chapter_ids(work_id: str, soup: BeautifulSoup) -> list[dict]:
    # Multi-chapter: extract from dropdown
    options = soup.select("select#selected_id option")
    if options:
        return [
            {
                "chapter_id": opt["value"],
                "title":      opt.get_text(strip=True),
            }
            for opt in options
        ]

    # Single-chapter: find chapter ID from the comments link
    # AO3 embeds it in the show_comments href: /comments/show_comments?work_id=XXX
    # But the chapter ID is in the work page URL itself
    # Try extracting from any /chapters/ link on the page
    chapter_link = soup.select_one("a[href*='/chapters/']")
    if chapter_link:
        href = chapter_link["href"]
        chapter_id = href.split("/chapters/")[1].split("?")[0].split("#")[0]
        return [{"chapter_id": chapter_id, "title": "Chapter 1"}]

    # Last fallback: use work_id itself — single chapter works
    # can be fetched directly at /works/{work_id} without a chapter ID
    return [{"chapter_id": None, "title": "Chapter 1"}]


# ── Fetch one chapter's text ──────────────────────────────────────────────────
def fetch_chapter_text(work_id: str, chapter_id: str | None) -> str | None:
    # Single-chapter fic — fetch work page directly
    if chapter_id is None:
        url = f"{BASE_URL}/works/{work_id}?view_adult=true"
    else:
        url = f"{BASE_URL}/works/{work_id}/chapters/{chapter_id}?view_adult=true"
    
    soup = fetch(url)
    if soup is None:
        return None

    text_div = soup.select_one(f"div#chapter-{chapter_id} div.userstuff") if chapter_id else None
    if not text_div:
        text_div = soup.select_one("div.userstuff[role='article']")
    if not text_div:
        text_div = soup.select_one("div.userstuff")

    return text_div.get_text(separator=" ", strip=True) if text_div else None


# ── Fetch chapter-level comments ─────────────────────────────────────────────
def fetch_chapter_comments(work_id: str, chapter_id: str | None) -> list[dict]:
    # Single-chapter: comments are on the work page directly
    if chapter_id is None:
        base_url = f"{BASE_URL}/works/{work_id}?view_adult=true&show_comments=true"
    else:
        base_url = f"{BASE_URL}/works/{work_id}/chapters/{chapter_id}?view_adult=true&show_comments=true"
    
    all_comments = []
    page = 1

    while True:
        url  = f"{base_url}&page={page}"
        soup = fetch(url)
        if soup is None:
            break

        placeholder = soup.find(id="comments_placeholder")
        if not placeholder:
            break

        comment_tags = placeholder.select("ol.thread li.comment.group")
        if not comment_tags:
            break

        found = 0
        for c in comment_tags:
            text_el = c.find("blockquote", class_="userstuff")
            if not text_el:
                continue
            text  = text_el.get_text(separator=" ", strip=True)
            words = len(text.split())
            if words < MIN_COMMENT_WORDS:
                continue
            all_comments.append({"text": text, "word_count": words})
            found += 1

        print(f"        Comments page {page}: {found} qualifying")

        next_btn = placeholder.select_one("ol.pagination li.next a")
        if not next_btn or page >= 5:
            break
        page += 1

    all_comments.sort(key=lambda c: c["word_count"], reverse=True)
    return all_comments[:TOP_N_COMMENTS]


# ── Fetch full work ───────────────────────────────────────────────────────────
def fetch_work(stub: dict, fandom_key: str, trope_key: str) -> dict | None:
    """
    Fetch a complete work: validate chapter/word filters, then collect
    each chapter's text and its specific comments.

    Returns None if the work fails any quality filter.
    """
    work_id = stub["work_id"]
    print(f"    Fetching work {work_id}: {stub['title'][:50]}")

    # Step 1: fetch work index page to get chapter list and metadata
    soup = fetch(f"{BASE_URL}/works/{work_id}?view_adult=true")
    if soup is None:
        print(f"      ✗ Failed to fetch work page")
        return None

    title_tag = soup.select_one("title")
    print(f"      Page title: {title_tag.get_text()[:60] if title_tag else '?'}")

    # Step 2: get chapter IDs
    chapter_ids = fetch_chapter_ids(work_id, soup)
    n_chapters  = len(chapter_ids)
    print(f"      Chapters: {n_chapters}")

    # Filter: chapter count must be 1–15
    if not (MIN_CHAPTERS <= n_chapters <= MAX_CHAPTERS):
        print(f"      ✗ Chapter count {n_chapters} outside 1–{MAX_CHAPTERS}")
        return None

    # Filter: average words per chapter must be 2,500–5,000
    total_words = stub["words"]
    if n_chapters > 0:
        avg_words = total_words / n_chapters
    else:
        avg_words = total_words

    print(f"      Avg words/chapter: {avg_words:.0f}")
    if not (MIN_AVG_WORDS <= avg_words <= MAX_AVG_WORDS):
        print(f"      ✗ Avg words/chapter {avg_words:.0f} outside {MIN_AVG_WORDS}–{MAX_AVG_WORDS}")
        return None
        
    # Determine if oneshot (single chapter)
    is_oneshot = n_chapters == 1
    
    # Word count filter depends on format
    if is_oneshot:
        if total_words > 60000:
            print(f"      ✗ Oneshot too long: {total_words:,} words (max 60,000)")
            return None
    else:
        if not (MIN_AVG_WORDS <= avg_words <= MAX_AVG_WORDS):
            print(f"      ✗ Avg words/chapter {avg_words:.0f} outside {MIN_AVG_WORDS}–{MAX_AVG_WORDS}")
            return None

    # Step 3: extract author-applied tags
    tags = [t.get_text(strip=True) for t in soup.select("dd.freeform.tags a")]

    # Step 4: fetch each chapter's text + comments
    chapters = []
    total_comments = 0

    for ch_info in chapter_ids:
        chapter_id = ch_info["chapter_id"]
        print(f"      Chapter {chapter_id}: fetching text...")

        text = fetch_chapter_text(work_id, chapter_id)
        if not text:
            print(f"        ✗ No text — skipping chapter")
            continue

        word_count = len(text.split())
        print(f"        {word_count} words — fetching comments...")

        comments = fetch_chapter_comments(work_id, chapter_id)
        total_comments += len(comments)

        chapters.append({
            "index":      len(chapters),
            "chapter_id": chapter_id,
            "title":      ch_info["title"],
            "text":       text,
            "word_count": word_count,
            "comments":   comments,
        })

    if not chapters:
        print(f"      ✗ No chapters fetched")
        return None

    # Filter: minimum average comments per chapter
    avg_comments = total_comments / len(chapters)
    print(f"      Avg comments/chapter: {avg_comments:.1f}")
    if avg_comments < MIN_COMMENTS_PER_CHAPTER:
        print(f"      ✗ Not enough chapter-level comments")
        return None

    print(f"      ✓ Saved {len(chapters)} chapters, {total_comments} total comments")

    return {
        "work_id":    work_id,
        "fandom_key": fandom_key,
        "trope_key":  trope_key,
        "title":      stub["title"],
        "kudos":      stub["kudos"],
        "words":      stub["words"],
        "tags":       tags,
        "url":        stub["url"],
        "chapters":   chapters,
    }


# ── Checkpoint ────────────────────────────────────────────────────────────────
def load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT):
        return set()
    with open(CHECKPOINT) as f:
        return set(json.load(f))


def save_checkpoint(done_ids: set[str]):
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump(list(done_ids), f)


# ── Main loop ─────────────────────────────────────────────────────────────────
def scrape_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done_ids = load_checkpoint()

    for fandom_key, fandom_tag in FANDOMS.items():
        for trope_key, trope_info in TROPES.items():
            trope_tag = trope_info["ao3_tag"]
            cell_dir  = os.path.join(OUTPUT_DIR, fandom_key, trope_key)
            os.makedirs(cell_dir, exist_ok=True)

            existing  = [f for f in os.listdir(cell_dir) if f.endswith(".json")]
            collected = len(existing)

            if collected >= FICS_PER_CELL:
                print(f"  [{fandom_key}/{trope_key}] complete — skipping")
                continue

            print(f"\n[{fandom_key} × {trope_key}]")
            stubs = search_works(fandom_tag, trope_tag, FICS_PER_CELL)

            for stub in tqdm(stubs, desc=f"  {fandom_key}/{trope_key}"):
                if collected >= FICS_PER_CELL:
                    break
                if stub["work_id"] in done_ids:
                    continue

                record = fetch_work(stub, fandom_key, trope_key)
                if record is None:
                    continue

                out_path = os.path.join(cell_dir, f"{stub['work_id']}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)

                done_ids.add(stub["work_id"])
                save_checkpoint(done_ids)
                collected += 1
                print(f"    ✓ Saved {stub['work_id']} ({collected}/{FICS_PER_CELL})")

    print("\nScraping complete.")


if __name__ == "__main__":
    scrape_all()
