"""
This module is for normalizing hindi entity titles to Wikidata QIDs and English dbpedia titles.

Input ("दिल्ली", hi) -> 
[get_qid_from_lang_title] Asks Wikidata for QID -> 
[get_en_title_from_qid] Asks Wikidata about enwiki title -> 
canonical title: ("Delhi", en) -> 
Output: ("Delhi", "http://dbpedia.org/resource/Delhi")
"""

from typing import Dict, Optional, Tuple

import requests

QID_CACHE: Dict[Tuple[str, str], Optional[str]] = {}
EN_TITLE_CACHE: Dict[str, Optional[str]] = {}


def _requests_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "NEF-EntityNormalization (https://github.com/dbpedia/neural-extraction-framework)"
    })
    return s


def parse_genre_text(text: str):
    parts = [p.strip() for p in text.split(">>")]
    if len(parts) == 2 and parts[0]:
        return parts[0], parts[1]
    return text.strip(), "en"


def get_qid_from_lang_title(lang: str, title: str, timeout: float = 8.0):
    """Resolve a Wikipedia (lang, title) to a Wikidata QID using wbgetentities.

    Returns None if not found.
    """
    if not title:
        return None
    key = (lang, title.replace(" ", "_"))
    if key in QID_CACHE:
        return QID_CACHE[key]

    session = _requests_session()
    try:
        r = session.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "format": "json",
                "sites": f"{lang}wiki",
                "titles": title,
                "props": "",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        entities = data.get("entities", {})
        # entities is a dict mapping QID or -1 to info; pick the first QID key
        for k in entities.keys():
            if k.startswith("Q"):
                QID_CACHE[key] = k
                return k
    except Exception as e:
        print(f"Error getting QID for {title} in {lang}: {e}")
        pass
    # fallback: search API - less precise but useful for redirects/variants
    try:
        r = session.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "language": lang,
                "search": title,
                "format": "json",
                "limit": 1,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("search"):
            qid = data["search"][0]["id"]
            QID_CACHE[key] = qid
            return qid
    except Exception as e:
        print(f"Error getting QID for {title} in {lang}: {e}")
        pass

    QID_CACHE[key] = None
    return None


def get_en_title_from_qid(qid: str, timeout: float = 8.0):
    if not qid:
        return None
    if qid in EN_TITLE_CACHE:
        return EN_TITLE_CACHE[qid]

    session = _requests_session()
    try:
        r = session.get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        entity = data.get("entities", {}).get(qid, {})
        enwiki = entity.get("sitelinks", {}).get("enwiki")
        if enwiki and enwiki.get("title"):
            title = enwiki["title"].replace(" ", "_")
            EN_TITLE_CACHE[qid] = title
            return title
    except Exception as e:
        print(f"Error getting English title for {qid}: {e}")
        pass
    EN_TITLE_CACHE[qid] = None
    return None


def normalize_to_dbpedia_title_from_genre_text(genre_text: str):
    """Given mGENRE text (e.g., "दिल्ली >> hi"), return (en_title, dbr_uri).

    If normalization fails, returns (None, None).
    """
    title, lang = parse_genre_text(genre_text)
    title = title.replace(" ", "_")
    if lang == "en":
        dbr = f"http://dbpedia.org/resource/{title}"
        return title, dbr

    qid = get_qid_from_lang_title(lang, title)
    if not qid:
        return None, None
    en_title = get_en_title_from_qid(qid)
    if not en_title:
        return None, None
    dbr = f"http://dbpedia.org/resource/{en_title}"
    return en_title, dbr


def normalize_title(lang: str, title: str):
    """Normalize a (lang,title) to English Wikipedia title (underscored)."""
    if not title:
        return None
    if lang == "en":
        return title.replace(" ", "_")
    qid = get_qid_from_lang_title(lang, title)
    if not qid:
        return None
    return get_en_title_from_qid(qid)


