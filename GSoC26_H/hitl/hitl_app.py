import streamlit as st
import json
import os
import base64
import hashlib
import re
import requests
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="DBpedia Hindi · Triple Review",
    page_icon="🔗",
    layout="wide",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

CANDIDATE_PATHS = [
    _REPO_ROOT / "results" / "alignment_results_full_20k.jsonl",
    _REPO_ROOT / "data" / "alignment_results_full_20k.jsonl",
    "alignment_results_full_20k.jsonl",
    "/content/drive/MyDrive/dbpedia-hindi-gsoc/alignment_results_full_20k.jsonl",
    "/content/drive/MyDrive/alignment_results_full_20k.jsonl",
]
CANDIDATE_PATHS = [str(p) for p in CANDIDATE_PATHS]


GITHUB_REPO = "singhhnitin/neural-extraction-framework"
GITHUB_FILE_PATH = "GSoC26_H/results/hitl_corrections.jsonl"
GITHUB_BRANCH = "gsoc26h-development"

DBO_PROPERTY_PATTERN = re.compile(r'^dbo:[A-Za-z][A-Za-z0-9]*$')

DEMO_DATA = [
    {"sentence": "ताजमहल का निर्माण शाहजहाँ ने करवाया था।", "subject": "ताजमहल",
     "relation": "का निर्माण किया", "object": "शाहजहाँ ने करवाया", "dbo_uri": "dbo:builder",
     "score": 0.84, "method": "embedding"},
    {"sentence": "अमिताभ बच्चन का जन्म इलाहाबाद में हुआ था।", "subject": "अमिताभ बच्चन",
     "relation": "जन्म हुआ", "object": "इलाहाबाद", "dbo_uri": "dbo:birthPlace",
     "score": 0.91, "method": "copula_rule"},
    {"sentence": "उन्होंने राष्ट्रीय पुरस्कार जीता।", "subject": "उन्होंने",
     "relation": "जीता", "object": "राष्ट्रीय पुरस्कार", "dbo_uri": "dbo:winner",
     "score": 0.79, "method": "embedding"},
    {"sentence": "कंपनी ने नई नीति जारी की।", "subject": "कंपनी",
     "relation": "जारी की", "object": "नई नीति", "dbo_uri": None,
     "score": 0.41, "method": "hitl"},
    {"sentence": "वह संग्रहालय शहर के केंद्र में स्थित है।", "subject": "वह संग्रहालय",
     "relation": "स्थित है", "object": "शहर के केंद्र में", "dbo_uri": "dbo:location",
     "score": 0.68, "method": "embedding"},
]

CURATED_PROPERTIES = {
    "dbo:birthPlace": "Birth place", "dbo:birthDate": "Birth date",
    "dbo:deathPlace": "Death place", "dbo:deathDate": "Death date",
    "dbo:nationality": "Nationality", "dbo:occupation": "Occupation",
    "dbo:spouse": "Spouse", "dbo:child": "Child", "dbo:parent": "Parent",
    "dbo:award": "Award", "dbo:almaMater": "Alma mater", "dbo:employer": "Employer",
    "dbo:knownFor": "Known for", "dbo:religion": "Religion", "dbo:party": "Party",
    "dbo:field": "Field", "dbo:education": "Education", "dbo:ethnicity": "Ethnicity",
    "dbo:capital": "Capital", "dbo:country": "Country", "dbo:location": "Location",
    "dbo:region": "Region", "dbo:language": "Language", "dbo:leaderName": "Leader name",
    "dbo:populationTotal": "Population total", "dbo:areaTotal": "Area total",
    "dbo:elevation": "Elevation", "dbo:foundedBy": "Founded by", "dbo:foundingDate": "Founding date",
    "dbo:headquarter": "Headquarter", "dbo:leader": "Leader", "dbo:president": "President",
    "dbo:numberOfEmployees": "Number of employees", "dbo:author": "Author",
    "dbo:director": "Director", "dbo:producer": "Producer", "dbo:starring": "Starring",
    "dbo:publisher": "Publisher", "dbo:builder": "Builder", "dbo:architect": "Architect",
    "dbo:genre": "Genre", "dbo:releaseDate": "Release date", "dbo:musicComposer": "Music composer",
    "dbo:lyricist": "Lyricist", "dbo:date": "Date", "dbo:place": "Place", "dbo:winner": "Winner",
    "dbo:participant": "Participant", "dbo:doctoralAdvisor": "Doctoral advisor",
    "dbo:influenced": "Influenced", "dbo:influencedBy": "Influenced by", "dbo:team": "Team",
    "dbo:sport": "Sport", "dbo:position": "Position", "dbo:coach": "Coach",
    "dbo:successor": "Successor", "dbo:predecessor": "Predecessor", "dbo:deputy": "Deputy",
    "dbo:isPartOf": "Is part of", "dbo:related": "Related", "dbo:city": "City",
    "dbo:college": "College", "dbo:district": "District", "dbo:family": "Family",
    "dbo:movement": "Movement", "dbo:officialLanguage": "Official language",
    "dbo:origin": "Origin", "dbo:state": "State", "dbo:university": "University",
    "dbo:battle": "Battle", "dbo:commander": "Commander", "dbo:kingdom": "Kingdom",
    "dbo:ground": "Ground",
}
PROPERTY_OPTIONS = sorted(CURATED_PROPERTIES.keys())

ERROR_TYPES = [
    "Predicate normalization — surface form not standardized",
    "Predicate placeholder — no real relation identified",
    "Implicit relation — bare copula or postposition",
    "Language mixing — English and Hindi mixed in predicate",
    "Argument span error — subject or object boundary is wrong",
    "Missing triple — relation was not extracted at all",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 880px; padding-top: 2rem; padding-bottom: 4rem; }
.app-header { display: flex; align-items: center; gap: 16px; margin-bottom: 6px; }
.app-header .dbpedia-logo { height: 42px; width: auto; }
.app-header .header-title { font-family: 'Lora', serif; font-weight: 600; font-size: 28px; color: #14181A; line-height: 1.1; }
.app-header .header-subtitle { font-size: 14px; color: #5B6663; margin-top: 2px; }
.app-divider { height: 1px; background: #E2E5E1; margin: 18px 0 28px 0; }
.chip-row { display: flex; gap: 10px; flex-wrap: wrap; }
.chip { border: 1px solid #E2E5E1; border-radius: 8px; padding: 10px 14px; background: #FFFFFF; flex: 1; min-width: 140px; }
.chip-label { font-size: 10.5px; letter-spacing: 0.07em; color: #8A938F; text-transform: uppercase; margin-bottom: 5px; font-weight: 600; }
.chip-value { font-family: 'JetBrains Mono', monospace; font-size: 15px; color: #14181A; word-break: break-word; }
.sentence-box { font-family: 'JetBrains Mono', monospace; font-size: 16px; line-height: 1.7; color: #14181A; background: #FFFFFF; border: 1px solid #E2E5E1; border-left: 3px solid #0E7C7B; border-radius: 6px; padding: 16px 18px; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 600; letter-spacing: 0.02em; }
.badge-high { background: #DCF5E8; color: #15803D; }
.badge-none { background: #FEE2E2; color: #B91C1C; }
.meter { height: 6px; background: #E9ECE9; border-radius: 4px; overflow: hidden; margin-top: 10px; }
.meter-fill { height: 100%; border-radius: 4px; }
.suggestion-uri { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 500; color: #0E7C7B; margin-top: 10px; }
.suggestion-caption { font-size: 12.5px; color: #8A938F; margin-top: 8px; }
.stButton button { border-radius: 6px; font-weight: 600; padding-top: 8px; padding-bottom: 8px; }
.app-footer { text-align: center; font-size: 12.5px; color: #8A938F; margin-top: 48px; }
</style>
""", unsafe_allow_html=True)

DBPEDIA_LOGO_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/DBpedia_logo.svg"

header_html = (
    '<div class="app-header">'
    f'<img src="{DBPEDIA_LOGO_URL}" alt="DBpedia" class="dbpedia-logo">'
    '<div><div class="header-title">DBpedia Hindi Chapter</div>'
    '<div class="header-subtitle">Knowledge graph triple review · subject · relation · object</div>'
    '</div></div>'
)
st.markdown(header_html, unsafe_allow_html=True)

with st.expander("About this tool"):
    st.markdown(
        "**What this is.** Every fact extracted from a Hindi sentence is a small graph — "
        "a subject, a relation, and an object. Before a fact joins DBpedia's knowledge graph, "
        "the relation needs to match one of DBpedia's standard properties (things like "
        "`dbo:birthPlace` or `dbo:builder`). A fine-tuned model plus an LLM disambiguation "
        "step propose a match; this tool is where a person confirms, corrects, or rejects "
        "that proposal. Corrections sync automatically back to the pipeline's training data.\n\n"
        "**DBpedia** extracts structured information from Wikipedia and publishes it as "
        "linked open data. This review queue supports the DBpedia Hindi Chapter's work "
    )

with st.expander("Connect your pipeline"):
    st.markdown(
        "This app looks for a file named **`alignment_results_full_20k.jsonl`** "
        "(one JSON object per line, with `sentence`, `subject`, `relation`, `object`, "
        "`dbo_uri`, and `score` fields) in the following locations, in order:\n\n"
        "- `GSoC26_H/results/alignment_results_full_20k.jsonl` *(recommended)*\n"
        "- `GSoC26_H/data/alignment_results_full_20k.jsonl`\n"
        "- the app's own working directory\n"
        "- common Google Drive paths (for running from Colab)\n\n"
        "If none are found, the queue falls back to a small demo set."
    )

st.markdown('<div class="app-divider"></div>', unsafe_allow_html=True)

APP_PASSWORD = st.secrets.get("app_password")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    if not APP_PASSWORD:
        st.error(
            "This tool is locked and no password has been configured yet. "
            "Add `app_password` under Settings → Secrets to enable access."
        )
        st.stop()

    st.markdown("**This review tool is password-protected. Enter the password to continue.**")
    entered = st.text_input("Password", type="password", key="password_gate_input")
    if st.button("Enter"):
        if entered == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def make_triple_id(row):
    """Stable identity for a triple, independent of file order or reruns."""
    key = f"{row.get('sentence','')}|{row.get('subject','')}|{row.get('relation','')}|{row.get('object','')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def add_ids(rows):
    out = []
    for r in rows:
        r2 = dict(r)
        r2["triple_id"] = make_triple_id(r2)
        out.append(r2)
    return out


@st.cache_data
def load_data():
    for path in CANDIDATE_PATHS:
        if os.path.exists(path):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows, path
    return DEMO_DATA, None


def load_existing_corrections():
    """Pull previously synced corrections from GitHub so reloads know what's
    already been judged. Returns (list_of_decisions, set_of_triple_ids)."""
    token = st.secrets.get("github_token")
    if not token:
        return [], set()

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    try:
        resp = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=10)
    except requests.RequestException:
        return [], set()

    if resp.status_code != 200:
        return [], set()

    try:
        content = base64.b64decode(resp.json()["content"]).decode("utf-8")
    except Exception:
        return [], set()

    decisions, ids = [], set()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        decisions.append(d)
        tid = d.get("triple_id") or make_triple_id(d)
        ids.add(tid)
    return decisions, ids


def sync_to_github(new_decisions):
    """Push new decisions to GitHub via the Contents API, skipping anything
    whose triple_id is already present. Requires github_token in Streamlit
    secrets. Returns (success, message)."""
    token = st.secrets.get("github_token")
        return False, "No github_token found in app secrets. Add it under Settings → Secrets."

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    resp = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH})
    if resp.status_code == 200:
        file_data = resp.json()
        sha = file_data["sha"]
        existing_content = base64.b64decode(file_data["content"]).decode("utf-8")
    elif resp.status_code == 404:
        sha = None
        existing_content = ""
    else:
        return False, f"Failed to read existing file: HTTP {resp.status_code}"

    existing_ids = set()
    for line in existing_content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        existing_ids.add(d.get("triple_id") or make_triple_id(d))

    to_push = [d for d in new_decisions if d.get("triple_id") not in existing_ids]
    skipped = len(new_decisions) - len(to_push)

    if not to_push:
        return True, f"Nothing new to sync — {skipped} item(s) already on GitHub."

    new_lines = "\n".join(json.dumps(d, ensure_ascii=False) for d in to_push)
    if existing_content.strip():
        updated_content = existing_content.rstrip("\n") + "\n" + new_lines + "\n"
    else:
        updated_content = new_lines + "\n"

    encoded_content = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")

    payload = {
        "message": f"HITL: sync {len(to_push)} new correction(s)",
        "content": encoded_content,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(api_url, headers=headers, json=payload)
    if put_resp.status_code in (200, 201):
        msg = f"Synced {len(to_push)} correction(s) to GitHub."
        if skipped:
            msg += f" ({skipped} already present, skipped.)"
        return True, msg
    else:
        return False, f"Sync failed: HTTP {put_resp.status_code} — {put_resp.text[:200]}"


all_rows, found_path = load_data()
using_demo = found_path is None
all_rows = add_ids(all_rows)

if using_demo:
    st.info("**Demo mode** — showing 5 sample triples. See \"Connect your pipeline\" above to load real data.")
else:
    st.caption(f"Connected · {len(all_rows):,} rows loaded from `{found_path}`")

if "queue" not in st.session_state:
    existing_decisions, corrected_ids = load_existing_corrections()
    full_queue = sorted(all_rows, key=lambda r: r.get("score", 0), reverse=True)
    remaining_queue = [r for r in full_queue if r["triple_id"] not in corrected_ids]

    st.session_state.queue = remaining_queue
    st.session_state.idx = 0
    st.session_state.decisions = existing_decisions
    st.session_state.synced_count = len(existing_decisions)
    st.session_state.total_pool_size = len(full_queue)

queue = st.session_state.queue
total = len(queue)

with st.sidebar:
    st.markdown("##### Review progress")
    decisions_made = len(st.session_state.decisions)
    already_synced = st.session_state.synced_count
    pool_size = st.session_state.get("total_pool_size", total)
    st.metric("Total judged so far", decisions_made)
    st.caption(f"{already_synced} synced to GitHub · {decisions_made - already_synced} pending sync")
    st.caption(f"Queue remaining: {total} of {pool_size} total item{'s' if pool_size != 1 else ''}")
    st.progress(min(decisions_made / pool_size, 1.0) if pool_size else 0)

    st.markdown("---")
    st.markdown("##### Filter queue")
    filter_mode = st.radio(
        "Show",
        ["All", "Has suggested property", "No suggested property"],
        index=0,
        label_visibility="collapsed",
    )
    if filter_mode == "Has suggested property":
        view_queue = [r for r in queue if r.get("dbo_uri")]
    elif filter_mode == "No suggested property":
        view_queue = [r for r in queue if not r.get("dbo_uri")]
    else:
        view_queue = queue

    st.markdown("---")
    st.markdown("##### Sync to pipeline")
    unsynced = decisions_made - st.session_state.synced_count
    if unsynced > 0:
        st.caption(f"{unsynced} decision(s) not yet synced")
        if st.button("↑ Sync to GitHub now", use_container_width=True, type="primary"):
            pending = st.session_state.decisions[st.session_state.synced_count:]
            success, message = sync_to_github(pending)
            if success:
                st.session_state.synced_count = decisions_made
                st.success(message)
            else:
                st.error(message)
    else:
        st.caption("Everything synced." if decisions_made else "Review items to begin.")

    if st.session_state.decisions:
        export_str = "\n".join(json.dumps(d, ensure_ascii=False) for d in st.session_state.decisions)
        st.download_button(
            "Download corrections (.jsonl)",
            data=export_str,
            file_name="hitl_corrections.jsonl",
            mime="application/jsonl",
            use_container_width=True,
        )
    else:
        st.caption("Corrections will appear here for download once you start reviewing.")

if not view_queue:
    st.success("Nothing left in this filter view.")
    st.stop()

idx = st.session_state.idx % len(view_queue)
row = view_queue[idx]

score = row.get("score", 0)
dbo_uri = row.get("dbo_uri")
method = row.get("method", "")

if dbo_uri:
    badge_class, badge_text, meter_color = "badge-high", "Property suggested", "#15803D"
else:
    badge_class, badge_text, meter_color = "badge-none", "No property suggested", "#B91C1C"

st.caption(f"Item {idx + 1} of {len(view_queue)}")

col1, col2 = st.columns([1.6, 1])

with col1:
    st.markdown("**Sentence**")
    st.markdown(f'<div class="sentence-box">{row.get("sentence", "")}</div>', unsafe_allow_html=True)

    st.markdown("<br>**Extracted triple**", unsafe_allow_html=True)
    relation_text = row.get("relation", "")
    relation_display = f'{dbo_uri}  <span style="color:#8A938F;">({relation_text})</span>' if dbo_uri else relation_text
    chip_html = (
        '<div class="chip-row">'
        f'<div class="chip"><div class="chip-label">Subject</div><div class="chip-value">{row.get("subject","")}</div></div>'
        f'<div class="chip"><div class="chip-label">Relation</div><div class="chip-value">{relation_display}</div></div>'
        f'<div class="chip"><div class="chip-label">Object</div><div class="chip-value">{row.get("object","")}</div></div>'
        '</div>'
    )
    st.markdown(chip_html, unsafe_allow_html=True)

with col2:
    st.markdown("**Suggested mapping**")
    st.markdown(f'<span class="badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    if dbo_uri:
        st.markdown(f'<div class="suggestion-uri">{dbo_uri}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="suggestion-uri" style="color:#8A938F;">No property suggested</div>', unsafe_allow_html=True)

    pct = max(0, min(100, score * 100))
    meter_html = (
        f'<div class="meter"><div class="meter-fill" style="width:{pct}%; background:{meter_color};"></div></div>'
        f'<div class="suggestion-caption">Confidence {score:.2f} · matched via {method or "—"}</div>'
    )
    st.markdown(meter_html, unsafe_allow_html=True)

st.markdown('<div class="app-divider"></div>', unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)

def save_decision(action, **extra):
    decision = {
        "triple_id": row.get("triple_id"),
        "sentence": row.get("sentence", ""),
        "subject": row.get("subject", ""),
        "relation": row.get("relation", ""),
        "object": row.get("object", ""),
        "suggested_dbo_uri": dbo_uri,
        "suggested_score": score,
        "action": action,
        "timestamp": datetime.utcnow().isoformat(),
        **extra,
    }
    st.session_state.decisions.append(decision)
    st.session_state.idx += 1
    st.session_state.show_modify = False
    st.session_state.show_reject = False
    st.rerun()

with b1:
    if st.button("✓  Accept", use_container_width=True, type="primary", disabled=not dbo_uri):
        save_decision("accept", final_dbo_uri=dbo_uri)
with b2:
    if st.button("✎  Modify", use_container_width=True):
        st.session_state.show_modify = True
        st.session_state.show_reject = False
with b3:
    if st.button("✕  Reject", use_container_width=True):
        st.session_state.show_reject = True
        st.session_state.show_modify = False

if st.session_state.get("show_modify"):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Pick the correct property**")
    new_prop = st.selectbox(
        "Correct dbo: property",
        options=["— type a custom property below —"] + PROPERTY_OPTIONS,
        key=f"modify_select_{idx}",
        label_visibility="collapsed",
    )
    custom_prop = ""
    if new_prop == "— type a custom property below —":
        custom_prop = st.text_input("Custom property", placeholder="dbo:somePropertyName", key=f"custom_{idx}")
        if custom_prop and not DBO_PROPERTY_PATTERN.match(custom_prop.strip()):
            st.caption("⚠ Must look like `dbo:PropertyName` — starts with `dbo:`, letters and numbers only, no spaces.")

    if st.button("Save correction", key=f"save_mod_{idx}"):
        if new_prop == "— type a custom property below —":
            candidate = custom_prop.strip()
            if not DBO_PROPERTY_PATTERN.match(candidate):
                st.error("Please enter a valid `dbo:PropertyName` before saving.")
            else:
                save_decision("modify", final_dbo_uri=candidate)
        else:
            save_decision("modify", final_dbo_uri=new_prop)

if st.session_state.get("show_reject"):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Why is this wrong?**")
    error_type = st.radio("Error type", ERROR_TYPES, key=f"error_{idx}", label_visibility="collapsed")
    note = st.text_input("Note (optional)", key=f"note_{idx}")
    if st.button("Save rejection", key=f"save_rej_{idx}"):
        save_decision("reject", error_type=error_type, note=note, final_dbo_uri=None)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Skip without deciding →"):
    st.session_state.idx += 1
    st.session_state.show_modify = False
    st.session_state.show_reject = False
    st.rerun()

st.markdown(
    '<div class="app-footer">Built for the DBpedia Hindi Chapter · Google Summer of Code 2026</div>',
    unsafe_allow_html=True,
)
