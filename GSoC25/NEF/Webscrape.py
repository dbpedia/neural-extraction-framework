import time, csv, re, sys, email.utils as eut
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

import feedparser, requests
from bs4 import BeautifulSoup
from readability import Document
import urllib.robotparser as urobot

# ----------------- SETTINGS -----------------
HEADERS = {"User-Agent": "NEF-RecentScraper/1.0 (+contact@example.com)"}
TIMEOUT = 15
MAX_RETRIES = 1
SLEEP_BETWEEN_REQUESTS = 0.25  # seconds between article fetches

# Content thresholds
MIN_SNIPPET_CHARS = 140         # after cleaning
MAX_SNIPPET_CHARS = 900         # clamp long bodies

# Collection targets
FEEDS_PER_TOPIC   = 2
MAX_PER_FEED      = 6
TARGET_PER_TOPIC  = 10
RECENCY_HOURS     = 48          # only keep items published/updated in this window

# ----------------- TOPIC FEEDS -----------------
FEEDS = {
    "TV shows & movies": [
        "https://www.theguardian.com/film/rss",
        "https://variety.com/v/tv/feed/",
        "https://www.hollywoodreporter.com/c/tv/tv-news/feed/",
    ],
    "Science & technology": [
        "https://arstechnica.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://www.npr.org/rss/rss.php?id=1019",
    ],
    "Art": [
        "https://hyperallergic.com/feed/",
        "https://news.artnet.com/feed",
        "https://www.theguardian.com/artanddesign/rss",
    ],
    "History": [
        "https://www.historyextra.com/feed/",
        "https://www.theguardian.com/culture/series/thelongread/rss",
    ],
    "Sports": [
        "https://www.espn.com/espn/rss/news",
        "https://feeds.bbci.co.uk/sport/rss.xml",
        "https://www.npr.org/rss/rss.php?id=1055",
    ],
    "Music": [
        "https://www.theguardian.com/music/rss",
        "https://pitchfork.com/feed/feed-news/rss",
        "https://www.npr.org/rss/rss.php?id=1039",
    ],
    "Video games": [
        "https://feeds.ign.com/ign/games-all",
        "https://www.polygon.com/rss/index.xml",
        "https://www.pcgamer.com/rss/",
    ],
    "Geography": [
        "https://www.atlasobscura.com/feeds/latest",
        "https://www.bbc.com/travel/rss",
        "https://earthobservatory.nasa.gov/feeds/rss/eo.rss",
    ],
    "Politics": [
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://www.theguardian.com/politics/rss",
        "https://www.politico.com/rss/politics-news.xml",
    ],
    "Other": [
        "https://www.theguardian.com/world/rss",
    ],
}

# ----------------- HELPERS -----------------
def allowed_by_robots(url: str) -> bool:
    """Check robots.txt for article URLs; fall back to allow on errors."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urobot.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def fetch_html(url: str) -> str | None:
    """Fetch HTML (skip XML feeds/pages); retry lightly; honor robots."""
    if not allowed_by_robots(url):
        return None
    for _ in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "xml" in ctype:
                return None
            return r.text
        except Exception:
            continue
    return None

def extract_main_text(html: str) -> tuple[str, str]:
    """Prefer Readability; fallback to basic soup stripping."""
    if not html:
        return "", ""
    try:
        doc = Document(html)
        title = (doc.short_title() or "").strip()
        soup = BeautifulSoup(doc.summary(html_partial=True), "lxml")
        text = " ".join(soup.stripped_strings)
        return title, text
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string if soup.title else "").strip()
        for tag in soup(["script","style","header","footer","nav","aside"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return title, text

def clean_snippet(txt: str, limit=MAX_SNIPPET_CHARS) -> str:
    txt = re.sub(r"\s+", " ", txt or "").strip()
    return txt[:limit]

def entry_summary_fallback(entry) -> str:
    summary = entry.get("summary", "") or entry.get("description", "")
    soup = BeautifulSoup(summary, "lxml")
    return clean_snippet(" ".join(soup.stripped_strings))

def parse_entry_dt(entry) -> datetime | None:
    """Parse RFC822-like dates if available; return aware UTC datetime."""
    ts = entry.get("published") or entry.get("updated") or entry.get("dc:date")
    if not ts:
        return None
    try:
        dt = eut.parsedate_to_datetime(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def iso_or_empty(dt: datetime | None) -> str:
    return dt.astimezone(timezone.utc).isoformat() if dt else ""

# ----------------- MAIN SCRAPER -----------------
def scrape_feeds_recent(feeds_by_topic: dict) -> list[dict]:
    rows, seen_urls = [], set()
    now_utc = datetime.now(timezone.utc)
    recency_delta = timedelta(hours=RECENCY_HOURS)

    for topic, feeds in feeds_by_topic.items():
        topic_rows = 0
        print(f"\n[topic] {topic} → target {TARGET_PER_TOPIC} items from up to {min(FEEDS_PER_TOPIC, len(feeds))} feeds")

        for i, feed_url in enumerate(feeds[:FEEDS_PER_TOPIC], start=1):
            if topic_rows >= TARGET_PER_TOPIC:
                break
            print(f"  [feed {i}] {feed_url}", flush=True)

            feed = feedparser.parse(feed_url, request_headers=HEADERS)
            if feed.bozo and not feed.entries:
                print(f"    [warn] broken feed")
                continue

            kept_from_feed = 0
            for e in feed.entries:
                if kept_from_feed >= MAX_PER_FEED or topic_rows >= TARGET_PER_TOPIC:
                    break

                url = e.get("link")
                if not url or url in seen_urls:
                    continue

                # Recency filter (when date available)
                dt = parse_entry_dt(e)
                if dt is not None and (now_utc - dt) > recency_delta:
                    continue  # too old for "new knowledge" testing

                html = fetch_html(url)
                title, text = extract_main_text(html) if html else ("", "")
                if not title:
                    title = (e.get("title") or "").strip()

                snippet = clean_snippet(text)
                if len(snippet) < MIN_SNIPPET_CHARS:
                    snippet = entry_summary_fallback(e)

                if not title or len(snippet) < MIN_SNIPPET_CHARS:
                    continue

                rows.append({
                    "category": topic,
                    "feed": feed_url,
                    "title": title,
                    "url": url,
                    "published_iso": iso_or_empty(dt),
                    "snippet": snippet,
                })
                seen_urls.add(url)
                kept_from_feed += 1
                topic_rows += 1
                print(f"    [+] {kept_from_feed}/{MAX_PER_FEED} from feed | {topic_rows}/{TARGET_PER_TOPIC} in topic", flush=True)
                time.sleep(SLEEP_BETWEEN_REQUESTS)

            print(f"  [done] kept {kept_from_feed} from this feed")

        print(f"[topic done] {topic}: kept {topic_rows}/{TARGET_PER_TOPIC}")

    return rows

# ----------------- RUN -----------------
rows = scrape_feeds_recent(FEEDS)

out_path = "recent_seeds_categorized.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["category","feed","title","url","published_iso","snippet"])
    w.writeheader()
    w.writerows(rows)

print(f"\n✅ Saved {len(rows)} items → {out_path}")
