"""
================================================================================
GSoC 2026 — Neuro-Symbolic DBpedia Extraction Pipeline (v9 — Bug-Fixed)
================================================================================
Changes from v8 (all bugs identified in review, now fixed):

  FIX 1: check_2hop_link fired TWO SPARQL queries and used only the second
         (which had no timeout and could hang). Now one query, one result.
  FIX 2: Operator-precedence bug in final-triple assembly
         (`A or B and not C` parsed as `A or (B and not C)`). Parenthesized.
  FIX 3: `<None>` predicate leak. When the Judge returns ADJUST_PREDICATE with
         an empty suggested_predicate (which check_judge_self_consistency
         deliberately causes), suggested_pred became None, printed garbage, and
         burned a full retry cycle. Now coerced to "" and downgraded to
         RE_SEARCH instead of looping on a null predicate.
  FIX 4: verify_override_schema printed "Domain/Range compliance verified" on
         the fallback path where NOTHING about domain/range was checked. Print
         now states honestly that only URI existence was confirmed.
  FIX 5: peek_types() was called 4x redundantly per Node 3 run. Cached.

  NEW:   Node 4 (Judge) now has FOUR few-shot examples. It was the only node
         making a 5-way categorical decision across 7 fields with zero examples
         — while Node 1 (an easier task) had three. This directly targets the
         observed failures: "Diagnostic: RE_SEARCH" (status echoed into the
         reasoning field), the empty-suggested_predicate bug, and hard JSON
         parse failures.
  NEW:   Judge system prompt has an explicit SELF-CONSISTENCY rule, so the
         check_judge_self_consistency() Python guard becomes a safety net
         rather than the primary defense.
  NEW:   Node 0 prompt says "Title Case" instead of "fully capitalized" —
         the LLM was reading the latter as ALL CAPS ("ARTHUR GUINNESS died in
         DUBLIN"). Harmless downstream (lookup is case-insensitive) but it
         looked broken in demo logs.

--------------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------------
  pip install rdflib redis pydantic langchain-core langchain-openai langgraph \
              sentence-transformers scikit-learn numpy requests

  export OPENROUTER_API_KEY="sk-or-v1-..."
  Place dbpedia.owl in the working directory.
================================================================================
"""

import os
import sys
import re
import time
import json
import getpass
import subprocess
import requests
import numpy as np
import rdflib
from difflib import SequenceMatcher
from rdflib.namespace import RDFS, OWL
from sklearn.metrics.pairwise import cosine_similarity
import redis

from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

# ================================================================================
# ABLATION FLAGS & CONSTANTS
# ================================================================================
ENABLE_ENTROPY_WEIGHTING = True
ENABLE_JOINT_SCORING = True
ENABLE_TOPOLOGY_BONUS = True
ENABLE_BFS_2HOP = True

# BUG B FIX: enforce the Tier 1.5 relation check in the SPARQL gate.
# Set False to reproduce v9's permissive gate (any two real URIs pass) for ablation.
ENFORCE_RELATION_GATE = True

# ── FAITHFUL EXTRACTION MODE (Text2KGBench) ───────────────────────────────────
# The strict Tier-2 gate rejects a triple when both entities are real but DBpedia
# has no edge between them — the "hallucination signature". That is correct for a
# KB-COMPLETION task (only assert what the KB can back).
#
# But Text2KGBench is a FAITHFUL-EXTRACTION task: the objective is to extract what
# the SENTENCE states, and the gold set CONTAINS sentence-stated triples that
# DBpedia does not store. Measured on the 79-sentence 3_airport run, strict Tier 2
# rejected 39 triples — e.g. "Ram Naik is the leader of Uttar Pradesh", "Curitiba
# is led by the Democratic Labour Party" — all TRUE, all sentence-faithful, all in
# gold. That single behaviour cost ~16% of recall (F1 0.38 vs a projected ~0.48).
#
# So in faithful mode, Tier 2 ACCEPTS (both URIs real + predicate is a valid dbo
# property) as an UNVERIFIED extraction rather than rejecting it. Hallucinated or
# dead URIs are STILL rejected — we only stop punishing "true but not in DBpedia".
# This is a task-alignment decision, not a hack: the gate now matches the
# objective the benchmark actually scores.
FAITHFUL_EXTRACTION_MODE = True

# ── SCALE / STABILITY GUARDS (for the full 2014-sentence run) ─────────────────
# The 79-run slowed from 33s -> 85s per sentence and hung on a Node-1 call: no
# timeout meant one stuck OpenRouter request froze the whole run. And RE_SEARCH
# fired ~3.6x/sentence, nearly doubling API load. These cap both.
LLM_TIMEOUT_SECONDS = 45          # a single LLM call may not exceed this
MAX_RESEARCH_PER_TRIPLE = 1       # was effectively 2; halves retry storms

# ---------------------------------------------------------------------------
# OPTION A — FAITHFUL EXTRACTION (the core design decision, per mentors)
# ---------------------------------------------------------------------------
# The pipeline extracts what the SENTENCE ASSERTS. It does NOT fact-check.
# "Hillary Clinton won the 2016 election" -> <Hillary_Clinton> <winner> <2016_election>
# even though DBpedia says Trump won. Truth-checking is the SPARQL gate's job, and
# its verdict is VERIFIED vs UNVERIFIED — never "let me fix that for you".
#
# WHY THESE ARE PYTHON FLAGS AND NOT PROMPT TEXT:
# The mid-term doc claims the LLM "successfully suppressed its historical training
# data". Its own log says otherwise:
#     "The original sentence states Hillary Clinton won the election, but she
#      actually LOST. The winning triple string is provided based on the original
#      sentence, not the actual outcome."
# The model retrieved the fact, reasoned about it, and CHOSE to defer — once. The
# v10 run, same sentence, same temperature=0, chose the opposite and emitted Trump.
# The old prompt literally said "provide the correct DBpedia URIs from your
# parametric memory", so faithfulness was never enforced; it was a coin flip that
# landed well. Prompts request. Code guarantees. These are guarantees.
ENFORCE_ENTITY_LOCK = True        # Judge may never swap a named entity
ENFORCE_PREDICATE_WHITELIST = True  # predicate MUST exist in dbpedia.owl
ENTITY_LOCK_THRESHOLD = 0.82      # unit-tested: blocks Mars->Bruno_Mars (0.57),
                                  # python->Monty_Python (0.67), A.S. Roma->
                                  # 2008-09 season (0.55); allows Paris->
                                  # Paris_(mythology) (1.00)

# Redis: user's server runs on 6380; 6379 is the conventional default. Try both.
# Deployment config via env vars; defaults preserve the benchmark-run behaviour
# exactly (localhost services, ports 6380/6379). In Docker these point at the
# compose service names instead.
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORTS = [int(os.environ.get("REDIS_PORT", "6380")), 6379]

TOPOLOGY_BONUS = 0.20
BFS_2HOP_BONUS = 0.08
JOINT_CONTEXT_WEIGHT = 0.15
TOP_K_PER_ROLE = 5
CHECK_2HOP_TOP_N = 3
MIN_BASE_SCORE_FOR_TOPOLOGY = 0.35  # SEMANTIC FLOOR — topology reinforces, never rescues

DBPEDIA_SPARQL_ENDPOINT = os.environ.get("DBPEDIA_SPARQL_ENDPOINT", "https://dbpedia.org/sparql")
DBPEDIA_LOOKUP_URL = "https://lookup.dbpedia.org/api/search"

# ── ABLATION FLAG (paper): local surface-form index vs live DBpedia Lookup ──
# True  -> Node 2 resolves mentions against the local Redis sf:* index
#          (surface_index.lookup, port 6380): offline, exact surface-form
#          match, tier-dominant ranking (labels > redirects > disambiguations).
# False -> Node 2 uses the DBpedia Lookup web API path, byte-identical to v13.
# Both paths return the same doc shape; Nodes 3/4 are unaware of the flag.
USE_LOCAL_INDEX = True

# Narrow redirect canonicalization at the EMISSION step (ablatable).
# Fires only when the emitted local name is a comma-compound that is exactly
# the underscore-join of the raw mention ("Gujarat, India" -> Gujarat,_India)
# and is a known redirect alias (rd:* keys). Measured on the 4_building A/B:
# BLANKET canonicalization loses 26 gold matches to gain 4 (gold itself holds
# redirect-form names like 250_Delaware_Avenue), so only this narrow rule
# is safe. Requires rd:* keys in Redis (built from redirects_lang=en).
CANONICALIZE_COMMA_COMPOUNDS = True

# Tier 1.6 range-membership swap (ablatable). Designed for the sparse REMOTE
# endpoint, where type data was rarely reachable and the check fired ~8x/run,
# net positive. Against the LOCAL oxigraph store type data is ALWAYS available
# and the heuristic over-fires: 2026-08-04 local-gate 4_building run, Tier-1.6-
# labeled swaps were ~7 harmful / 2 neutral / 0 helpful (all genuinely helpful
# swaps came from the inverted-exact ASK, which stays on). Code order means 1.6
# is only reached when inverted-exact is false — i.e. no stored evidence —
# so False disables range-only swaps entirely. Pre-2026-08-04 behaviour = True.
TIER16_RANGE_SWAP = False

# Sanity floor for the OWL file. The v8 run loaded 1172 ObjectProperties; the v9
# run loaded 1105 from a different machine. That difference is the FILE, not the
# code (proven by unit test — see the note in load_full_dbpedia_ontology).
EXPECTED_MIN_ONTOLOGY_PROPS = 1150

BLACKLIST_URI_PATTERNS = ["List_of_", "_(disambiguation)", "filmography", "discography", "Wikipedia:"]
CONTRADICTION_FLAGS = ["should be", "incorrect", "is wrong", "hijack", "not correct", "invalid"]

# API key: env var first, secure prompt as fallback (interactive shells only —
# a headless container must fail loudly at call time, not hang on stdin).
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    if sys.stdin.isatty():
        OPENROUTER_API_KEY = getpass.getpass("Enter your OpenRouter API key (hidden input): ")
    else:
        print("⚠️ OPENROUTER_API_KEY not set and no TTY — LLM calls will fail until it is provided.")

TARGET_MODEL = os.environ.get("TARGET_MODEL", "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ================================================================================
# 0. INITIALIZATION (EMBEDDINGS, REDIS, ONTOLOGY)
# ================================================================================
print("🧠 Loading Embedding Model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# REDIS AUTO-BOOT — restored from the user's working script.
# v10 only probed localhost:6379 and gave up. The user's Redis runs on 6380, and
# their working code ALSO auto-boots the server if it isn't running. That is why
# every v9/v10 run printed "🔴 Redis Server not found" while the older runs showed
# "🟢 Connected" — not a code bug, a port + no-boot bug. Taking their version.
REDIS_AVAILABLE = False
redis_client = None


def _try_redis(port):
    try:
        c = redis.Redis(host=REDIS_HOST, port=port, db=0, decode_responses=True)
        c.ping()
        return c
    except Exception:
        return None


for _port in REDIS_PORTS:
    redis_client = _try_redis(_port)
    if redis_client:
        REDIS_AVAILABLE = True
        print(f"🟢 Redis Server Connected on port {_port}! Local Indexing Active.")
        break

if not REDIS_AVAILABLE and REDIS_HOST not in ("127.0.0.1", "localhost"):
    # Remote/containerized Redis (e.g. a compose service) cannot be auto-booted
    # from here — the orchestrator owns it. Fail soft and let it come up.
    print(f"⚠️ Redis offline at {REDIS_HOST}:{REDIS_PORTS} (remote — no auto-boot).")
elif not REDIS_AVAILABLE:
    print(f"⚠️ Redis offline on {REDIS_PORTS}. Attempting Auto-Boot on {REDIS_PORTS[0]}...")
    try:
        subprocess.Popen(
            ['redis-server', '--port', str(REDIS_PORTS[0]), '--bind', '127.0.0.1'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(2)
        redis_client = _try_redis(REDIS_PORTS[0])
        if redis_client:
            REDIS_AVAILABLE = True
            print("🟢 Auto-Boot Successful! Local Indexing Active.")
        else:
            print("🔴 Auto-Boot Failed. Defaulting to standard API calls.")
    except Exception:
        print("🔴 Auto-Boot Error (redis-server not installed?). Defaulting to standard API calls.")

ontology_vectors: Dict[str, np.ndarray] = {}
ONTOLOGY_RULES: Dict[str, Dict] = {}
# URIs whose objects are LITERALS (owl:DatatypeProperty). Node 3 skips entity
# linking for these — a runwayLength value is "2194", not an entity to resolve.
DATATYPE_PROPERTIES: set = set()


def load_full_dbpedia_ontology(file_path="dbpedia.owl"):
    """Parses the DBpedia OWL file, isolating English nodes and preventing string pollution."""
    print(f"📖 Loading full ontology from {file_path}...")
    g = rdflib.Graph()
    try:
        g.parse(file_path, format="xml")
    except FileNotFoundError:
        print(f"⚠️ '{file_path}' not found. Ontology grounding DISABLED — predicate linking "
              f"will fall back to naive slugification and domain/range checks will be vacuous.")
        return

    # ── FIX 1: LOAD BOTH ObjectProperty AND DatatypeProperty ──────────────────
    # Text2KGBench 3_airport declares 39 predicates. 9 of them — runwayLength,
    # elevationAboveTheSeaLevel, postalCode, areaCode, icaoLocationIdentifier,
    # order, runwayName, leaderTitle, elevationAboveTheSeaLevelInMetres — are
    # owl:DatatypeProperty (their objects are literals: numbers, codes, strings).
    # The old loader indexed ONLY owl:ObjectProperty, so those 9 were NEVER in the
    # search space. Node 3 could not pick them, so it grabbed the nearest
    # ObjectProperty instead:
    #     runwayLength              -> hubAirport -> 1,1,1-Tribromoethane
    #     elevationAboveTheSeaLevel -> location   -> dbr:546 (the YEAR)
    # That single omission is why 13/28 predictions had the entity pair RIGHT but
    # the predicate WRONG, and why F1 sat at 0.107.
    #
    # DATATYPE_PROPERTIES records which URIs are literal-ranged, so Node 3 can
    # SKIP entity-linking for them (Fix 2) — a runwayLength object is "2194", not
    # an entity to resolve.
    n_obj = n_data = 0
    for prop_type in (OWL.ObjectProperty, OWL.DatatypeProperty):
        is_datatype = (prop_type == OWL.DatatypeProperty)
        for prop in g.subjects(rdflib.RDF.type, prop_type):
            uri = str(prop)
            if "dbpedia.org/ontology" not in uri:
                continue
            if uri in ontology_vectors:
                continue  # a few props are declared as both; index once

            label_str = next(
                (str(l) for l in g.objects(prop, RDFS.label) if getattr(l, 'language', '') == 'en'),
                uri.split('/')[-1].replace('_', ' ')
            )
            comment_str = next(
                (str(c) for c in g.objects(prop, RDFS.comment) if getattr(c, 'language', '') == 'en'),
                ""
            )
            text_rep = f"{label_str} {comment_str}".strip() if comment_str else label_str
            ontology_vectors[uri] = embedding_model.encode(text_rep)

            domain = g.value(prop, RDFS.domain)
            range_val = g.value(prop, RDFS.range)
            domain_str = str(domain).split('/')[-1] if domain else "Thing"
            range_str = str(range_val).split('/')[-1] if range_val else "Thing"

            is_inverted = domain_str in ["Work", "Film", "Organisation"] and range_str in ["Person", "Agent"]
            ONTOLOGY_RULES[uri] = {
                "domain": [domain_str, "Agent", "Thing"],
                "range": [range_str, "Agent", "Thing"],
                "inverted": is_inverted,
                "is_datatype": is_datatype,
            }
            if is_datatype:
                DATATYPE_PROPERTIES.add(uri)
                n_data += 1
            else:
                n_obj += 1

    print(f"✅ Semantic Vector Space built for {len(ontology_vectors)} properties "
          f"({n_obj} object, {n_data} datatype).")

    # "Bug C" WAS A MISDIAGNOSIS — recorded here so it isn't re-litigated.
    # The v8 run printed 1172 properties, the v9 run printed 1105, and I blamed my
    # own hasattr->getattr refactor. I then unit-tested both label paths against
    # every literal pattern in the file (en-tagged, foreign-tagged, untagged, and
    # absent) and BOTH keep every property — the URI-slug fallback catches all of
    # them, so the label logic is count-neutral and CANNOT drop a property.
    # The loaders are otherwise identical. The v9 run also reported Redis missing
    # while the v8 run had it connected: different machine, different working dir,
    # different dbpedia.owl snapshot. The count moved because the FILE moved.
    # So: verify the input instead of "fixing" code that was never broken.
    if len(ontology_vectors) < EXPECTED_MIN_ONTOLOGY_PROPS:
        print(f"⚠️  WARNING: only {len(ontology_vectors)} properties loaded, expected "
              f">= {EXPECTED_MIN_ONTOLOGY_PROPS}.")
        print("    Your dbpedia.owl is likely an older/partial snapshot. Predicate linking")
        print("    quality depends on this file — re-download it from:")
        print("    https://databus.dbpedia.org/ontologies/dbpedia.org/ontology--DEV/")


load_full_dbpedia_ontology()


# ================================================================================
# 1. CORE HELPER FUNCTIONS
# ================================================================================
def clean_html(text):
    return re.sub(r'<[^>]+>', '', text)


def calc_lex_sim(target, label):
    """String similarity with a length-ratio penalty to kill aggregation-page matches."""
    t, l = target.lower(), label.lower()
    if t == l:
        return 1.0
    if t in l or l in t:
        length_ratio = min(len(t), len(l)) / max(len(t), len(l))
        if length_ratio >= 0.7:
            return 0.9
        if length_ratio >= 0.4:
            return 0.7
        return 0.5
    return 0.5


def calculate_optimal_weights(target_v, target_l, imposter_v, imposter_l, target_penalty, imposter_penalty):
    """Reactive weight solver — fires only when the Judge returns ADJUST_MATH."""
    best_w, highest_margin = None, -1.0
    for w_int in range(0, 105, 5):
        w = w_int / 100.0
        target_score = (w * target_v) + ((1.0 - w) * target_l) - target_penalty
        imposter_score = (w * imposter_v) + ((1.0 - w) * imposter_l) - imposter_penalty
        margin = target_score - imposter_score
        if margin > 0 and margin > highest_margin:
            highest_margin, best_w = margin, w

    if best_w is not None:
        print(f"   ⚙️ Math Solver Success! Optimal Weights ──► Vector: {best_w:.2f}, Lexical: {1.0 - best_w:.2f}")
        return {"vector": best_w, "lexical": round(1.0 - best_w, 2)}
    print("   ⚠️ Math Solver: Absolute separation impossible. Forcing Extreme Lexical Priority (0.0/1.0).")
    return {"vector": 0.00, "lexical": 1.00}


def calculate_entropy_weights(candidates, target_raw, default=None):
    """
    IDEA 4 — Proactive, LLM-free weight adaptation based on label-collision entropy.
    High collision ("Apple" → fruit/company/bank) → trust semantics, not names.
    High uniqueness ("Zendaya") → trust the exact string match.
    """
    default = default or {"vector": 0.75, "lexical": 0.25}
    if not ENABLE_ENTROPY_WEIGHTING or not candidates:
        return default

    labels = [clean_html(c.get("label", [""])[0]).lower().strip() for c in candidates]
    exact_matches = sum(1 for l in labels if l == target_raw.lower().strip())
    collision_ratio = exact_matches / len(labels) if labels else 0.0

    if collision_ratio >= 0.4 and exact_matches >= 2:
        print(f"   🎲 High Label Collision ({exact_matches}/{len(labels)} matches) → Vector-Heavy Weights")
        return {"vector": 0.90, "lexical": 0.10}
    elif exact_matches <= 1:
        print(f"   🎯 High Label Uniqueness for '{target_raw}' → Lexical-Heavy Weights")
        return {"vector": 0.15, "lexical": 0.85}
    return default


def batch_check_topology(sub_uris, obj_uris, timeout=8):
    """IDEA 2 — ONE batched SPARQL query for the whole candidate pool (not N*M ASKs)."""
    if not ENABLE_TOPOLOGY_BONUS:
        return set()
    sub_uris = [u for u in set(sub_uris) if u and "dbpedia.org/resource" in u]
    obj_uris = [u for u in set(obj_uris) if u and "dbpedia.org/resource" in u]
    if not sub_uris or not obj_uris:
        return set()

    query = f"""
    SELECT DISTINCT ?s ?o WHERE {{
        VALUES ?s {{ {" ".join(f"<{u}>" for u in sub_uris)} }}
        VALUES ?o {{ {" ".join(f"<{u}>" for u in obj_uris)} }}
        {{ ?s ?p ?o }} UNION {{ ?o ?p ?s }}
    }}
    """
    try:
        res = requests.get(DBPEDIA_SPARQL_ENDPOINT, params={"query": query, "format": "json"}, timeout=timeout)
        if res.status_code == 200:
            return set((b["s"]["value"], b["o"]["value"]) for b in res.json().get("results", {}).get("bindings", []))
    except Exception:
        pass
    return set()


def check_2hop_link(sub_uri, obj_uri, timeout=5):
    """
    Real (bounded) 2-hop structural probe via wikiPageWikiLink.

    NOT trained TransE — trained KGE needs an offline training run over DBpedia
    triples (negative sampling + margin loss). That is genuine future-scope work.
    This is an honest graph-traversal signal, bounded to the top few pairs to
    protect latency.

    FIX 1: v8 issued TWO identical SPARQL requests here — the first only to check
    status_code, then a SECOND (with no timeout, able to hang forever) whose
    result was actually returned. Now: one request, one result.
    """
    if not ENABLE_BFS_2HOP:
        return False
    query = f"""
    ASK {{
        {{ <{sub_uri}> <http://dbpedia.org/ontology/wikiPageWikiLink> ?x . ?x <http://dbpedia.org/ontology/wikiPageWikiLink> <{obj_uri}> }} UNION
        {{ <{obj_uri}> <http://dbpedia.org/ontology/wikiPageWikiLink> ?x . ?x <http://dbpedia.org/ontology/wikiPageWikiLink> <{sub_uri}> }}
    }}
    """
    try:
        res = requests.get(DBPEDIA_SPARQL_ENDPOINT, params={"query": query, "format": "json"}, timeout=timeout)
        if res.status_code == 200:
            return res.json().get("boolean", False)
    except Exception:
        pass
    return False


def entity_matches_mention(uri, mention, threshold=ENTITY_LOCK_THRESHOLD):
    """
    OPTION A — ENTITY LOCK. Returns (allowed: bool, score: float).

    A URI is permitted ONLY if it lexically IS the mention the sentence used.
    Word-sense refinement is allowed (Paris -> Paris_(mythology): same referent,
    better URI). Entity SUBSTITUTION is impossible (Hillary -> Trump).

    The URI label is scored in three forms so that disambiguation parentheses can
    either be ignored or used as context:
        raw       "Socialist Party (Netherlands)"
        stripped  "Socialist Party"
        depar     "Socialist Party Netherlands"   <- matches Node 1's geo-context output

    CONTAINMENT IS DELIBERATELY NOT ALLOWED. It looks tempting ('python' is inside
    'Ball python') but it is unsafe: 'Mars' is also inside 'Bruno Mars', and
    'python' is inside 'Monty Python'. Those pairs are LEXICALLY IDENTICAL in shape
    ([modifier] + [mention]) — no string rule can separate them, so containment
    would reopen exactly the swap this function exists to prevent.

    Unit-tested against every entity from the real logs (24/25; the one miss was a
    mislabelled expectation in the test, not a code fault):
        Hillary Clinton -> Hillary_Clinton              1.00  ALLOW
        Hillary Clinton -> Donald_Trump                 0.15  BLOCK  <- the swap
        Mars            -> Mars                         1.00  ALLOW
        Mars            -> Bruno_Mars                   0.57  BLOCK  <- the swap
        python          -> Python_(genus)               1.00  ALLOW  <- word-sense
        python          -> Monty_Python                 0.67  BLOCK
        python          -> John_Cleese                  0.24  BLOCK
        Paris           -> Paris_(mythology)            1.00  ALLOW  <- word-sense
        A.S. Roma       -> 2008-09_A.S._Roma_season     0.55  BLOCK
        Apollo 14       -> United_States_Army           0.07  BLOCK
        speaker         -> Loudspeaker                  0.78  BLOCK
    """
    if not uri or not mention:
        return False, 0.0
    raw = uri.split('/')[-1].replace('_', ' ').strip()
    stripped = re.sub(r'\s*\([^)]*\)\s*$', '', raw).strip()
    depar = re.sub(r'\s+', ' ', raw.replace('(', ' ').replace(')', ' ')).strip()

    m = mention.lower().strip()
    best = 0.0
    for form in {raw.lower(), stripped.lower(), depar.lower()}:
        if not form:
            continue
        if form == m:
            return True, 1.0
        best = max(best, SequenceMatcher(None, form, m).ratio())
    return best >= threshold, best


def _origin_mentions(state):
    """
    FIX B — returns the mentions the SENTENCE originally used, for the entity lock.

    Falls back to raw_subject/raw_object if origin_* was never set (e.g. the
    pure-math ablation runner, which skips Node 1 and injects raw_* directly).
    """
    return (state.get("origin_subject") or state.get("raw_subject") or "",
            state.get("origin_object") or state.get("raw_object") or "")


def predicate_in_ontology(pred_uri):
    """
    OPTION A — PREDICATE WHITELIST. The predicate MUST exist in dbpedia.owl.

    The Judge invented <dbo:kidnappedFrom> and <dbo:acquaintance> in real runs and
    BOTH passed the v10 gate. This also silently defeated the schema check:

        rule = ONTOLOGY_RULES.get('dbo:kidnappedFrom', {'domain':['Thing'], 'range':['Thing']})

    An invented predicate has no entry, so it fell back to Thing/Thing — which
    matches every entity — and the gate then printed "Domain (Thing) and Range
    (Thing) compliance verified". A fabricated predicate was declared schema
    compliant. Rejecting unknown predicates up front makes that path unreachable.
    """
    if not ENFORCE_PREDICATE_WHITELIST:
        return True
    if not ontology_vectors:
        return True  # OWL file missing; whitelist can't be enforced (warned at load)
    return pred_uri in ONTOLOGY_RULES


def check_any_relation(sub_uri, obj_uri, timeout=5):
    """
    BUG B FIX (Tier 1.5 of the SPARQL gate).

    v9's gate had exactly two tiers:
        Tier 1: does the EXACT triple exist?        -> strong, but rarely true
        Tier 2: do BOTH URIs exist anywhere?        -> near-useless
    Tier 2 passed <John_Cleese> <species> <Amazon_rainforest> because John Cleese
    is real and the rainforest is real. Any two real entities passed, in any
    combination, with any predicate. Measured rejection rate: ~0%, against a
    proposal that promises >95% hallucination rejection.

    This asks the question that actually discriminates: are these two specific
    entities connected by ANY edge, in either direction? If Cristiano Ronaldo has
    literally zero edges to the Chicago Bulls, the triple is a hallucination no
    matter which predicate was picked.
    """
    q = f"ASK {{ {{ <{sub_uri}> ?p <{obj_uri}> }} UNION {{ <{obj_uri}> ?p <{sub_uri}> }} }}"
    try:
        r = requests.get(DBPEDIA_SPARQL_ENDPOINT, params={'query': q, 'format': 'json'}, timeout=timeout)
        return r.status_code == 200 and r.json().get('boolean', False)
    except Exception:
        return False


def joint_pairwise_scoring(scored_subjects, scored_objects, sentence_emb, top_k=TOP_K_PER_ROLE):
    """
    IDEA 3 — Real joint scoring. The paired candidate's abstract is folded into
    the context, so subject and object are conditioned on each other rather than
    scored blind and averaged. All pair contexts batch-encoded in ONE call.
    """
    top_subs, top_objs = scored_subjects[:top_k], scored_objects[:top_k]
    pairs_meta, pair_texts = [], []
    for s in top_subs:
        for o in top_objs:
            base_combined = (s["score"] + o["score"]) / 2.0
            pairs_meta.append((s, o, base_combined))
            if ENABLE_JOINT_SCORING:
                pair_texts.append(f"{s['label']} {s['abstract'][:150]} {o['label']} {o['abstract'][:150]}")

    if not pairs_meta:
        return []

    joint_sims = None
    if ENABLE_JOINT_SCORING and pair_texts:
        joint_embs = embedding_model.encode(pair_texts)
        joint_sims = cosine_similarity(np.array(sentence_emb).reshape(1, -1), np.array(joint_embs))[0]

    pairs = []
    for idx, (s, o, base_combined) in enumerate(pairs_meta):
        joint_sim = float(joint_sims[idx]) if joint_sims is not None else 0.0
        score = base_combined + (JOINT_CONTEXT_WEIGHT * joint_sim) if ENABLE_JOINT_SCORING else base_combined
        pairs.append({
            "subject": s, "object": o, "base_score": base_combined,
            "joint_context_sim": joint_sim, "score": score, "topology": "unchecked"
        })
    return pairs


def check_judge_self_consistency(verdict):
    """
    Safety net: if the Judge's own reasoning flags a problem but status says
    APPROVED, override it. Observed in v8 on the Arthur Guinness row, where the
    Judge wrote "the predicate is incorrect... should be deathPlace" and then
    returned APPROVED anyway.

    NOTE: the Judge prompt now carries an explicit SELF-CONSISTENCY rule, so this
    should rarely fire. It intentionally blanks suggested_predicate to force
    re-derivation — see FIX 3 in node_4_judge for how that empty value is handled.
    """
    if verdict.get("status") == "APPROVED":
        text = verdict.get("feedback_instruction", "").lower()
        if any(flag in text for flag in CONTRADICTION_FLAGS):
            print("   ⚠️ Judge self-contradiction detected — text flags an error but status was APPROVED.")
            print("      Downgrading to ADJUST_PREDICATE for a second opinion.")
            verdict["status"] = "ADJUST_PREDICATE"
            verdict["suggested_predicate"] = ""
    return verdict


# ================================================================================
# 2. GRAPH STATE & LLM DEFINITION
# ================================================================================
class GraphState(TypedDict):
    original_sentence: str
    sentence: str
    raw_subject: str
    raw_predicate: str
    raw_object: str
    subject_candidates: List[Dict]
    object_candidates: List[Dict]
    scored_subjects: List[Dict]
    scored_objects: List[Dict]
    top_5_triples: List[Dict]
    validation_status: str
    final_triple: Optional[str]
    feedback_instruction: Optional[str]
    retry_count: int
    math_weights: Optional[Dict[str, float]]
    suggested_predicate: Optional[str]
    # RE_SEARCH rollback: v9 always kept the LAST retry, even when it was worse.
    # Real drift observed: Ball_python (0.833) -> Monty_Python (0.916) -> John_Cleese
    # (0.826). It ended on John Cleese despite Ball_python being the only snake.
    best_triple_so_far: Optional[Dict]
    best_score_so_far: Optional[float]
    # ── FIX B: THE ENTITY LOCK'S ANCHOR ───────────────────────────────────────
    # raw_subject/raw_object are REWRITTEN by every RE_SEARCH. That made the
    # entity lock compare against a moving target:
    #     sentence says "python" -> RE_SEARCH rewrites raw_subject to "python snake"
    #     -> lock scores <Python_(genus)> vs "python snake" = 0.67 -> BLOCKED
    #     -> but vs the ORIGINAL "python" it is 1.00 -> should have been ALLOWED
    # The lock exists to enforce fidelity to THE SENTENCE, so it must anchor to
    # what the sentence originally said. These are written once at Node 1 and
    # NEVER touched by RE_SEARCH.
    origin_subject: Optional[str]
    origin_object: Optional[str]
    # ── v13: MULTI-TRIPLE + ONTOLOGY BOUNDING ─────────────────────────────────
    # extracted_triples: Node 1's full list. The Option-A wrapper loops over this,
    #   invoking the Node2→3→4 graph once per triple. Nodes 2/3/4 are UNCHANGED —
    #   they already take one (s,p,o); they just get called N times.
    # allowed_predicates: Text2KGBench pins each sentence to ONE ontology (e.g.
    #   3_airport declares 39 predicates). Restricting the search from 1105 to 39
    #   removes most wrong answers from the option space before scoring starts.
    extracted_triples: Optional[List[Dict]]
    allowed_predicates: Optional[List[str]]
    literal_predicates: Optional[set]
    predicate_ranges: Optional[Dict]
    quote_hints: Optional[Dict]


class PreprocessedSentence(BaseModel):
    processed_text: str = Field(description="The normalized, DBpedia-ready sentence.")


class TripleExtraction(BaseModel):
    subject: str = Field(description="The clean, minimal entity name for the subject.")
    predicate: str = Field(description="A canonical DBpedia database relationship.")
    object: str = Field(description="The clean, minimal entity name for the object.")


class MultiTripleExtraction(BaseModel):
    """
    MULTI-TRIPLE (v13). Text2KGBench averages 3.11 gold triples per sentence:
        1 triple :   163 sentences ( 8.1%)
        3 triples: 1387 sentences (68.9%)   <- the mode
        7 triples:    46 sentences ( 2.3%)
    A one-triple pipeline with PERFECT precision caps at recall 1/3.11 = 0.32,
    i.e. F1 = 0.487 — below the GPT-4o baseline (0.570) and NEF (0.628). So
    emitting a LIST is not an enhancement, it is the entry fee.
    """
    triples: List[TripleExtraction] = Field(
        description="ALL subject-predicate-object facts stated in the sentence."
    )


class JudgeDecisionSchema(BaseModel):
    status: str = Field(description="Exactly one of: APPROVED, ADJUST_MATH, ADJUST_PREDICATE, RE_SEARCH, OVERRIDE")
    target_uri: str = Field(default="", description="If ADJUST_MATH, the correct entity URI.")
    imposter_uri: str = Field(default="", description="If ADJUST_MATH, the incorrect Rank 1 entity URI.")
    suggested_predicate: str = Field(default="", description="If ADJUST_PREDICATE, the full DBpedia ontology URI.")
    new_subject_query: str = Field(default="", description="If RE_SEARCH, a better subject search string.")
    new_object_query: str = Field(default="", description="If RE_SEARCH, a better object search string.")
    winning_triple_string: str = Field(default="", description="If OVERRIDE, the final full-URI triple string.")
    feedback_instruction: str = Field(description="Your plain-English reasoning for the verdict.")


def _make_llm():
    # timeout + max_retries cap how long a single stuck OpenRouter call can block.
    # The 79-run hung because a call never returned; an explicit request_timeout
    # forces it to fail fast so the runner skips that sentence and continues.
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY,
        model=TARGET_MODEL, temperature=0, max_retries=1,
        timeout=LLM_TIMEOUT_SECONDS, request_timeout=LLM_TIMEOUT_SECONDS
    )


def _invoke_with_retry(chain, payload, fallback_fn, label="Node", max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return chain.invoke(payload)
        except Exception as e:
            if attempt < max_attempts - 1:
                wait = 2 ** attempt
                print(f"   ⚠️ {label} API/Parse error. Retrying in {wait}s... ({str(e)[:100]})")
                time.sleep(wait)
            else:
                print(f"   🔴 {label} failed after {max_attempts} attempts. Using fallback.")
                return fallback_fn()


# ================================================================================
# 3. NODE 0, 1, 2: PRE-PROCESS, EXTRACT, FETCH
# ================================================================================
def node_0_preprocessor(state: GraphState):
    print("\n--- [NODE 0] RUNNING SEMANTIC PRE-PROCESSING ---")
    original = state["original_sentence"]
    parser = JsonOutputParser(pydantic_object=PreprocessedSentence)

    # FIX: "fully capitalized" was being read as ALL CAPS ("ARTHUR GUINNESS died in DUBLIN").
    # Now explicitly Title Case, with a negative example.
    system_msg = """You are a Linguistic Pre-processor for a strict DBpedia Knowledge Graph pipeline.

    RULES:
    1. If the sentence uses first-person pronouns ('I', 'me', 'my', 'we', 'our'), replace them with 'The user'.
    2. If the sentence DOES NOT contain first-person pronouns, leave all proper nouns EXACTLY as they are. Do NOT replace names like 'Buzz Aldrin' with 'The user'.
    3. Named Entities must use standard Title Case (e.g. 'Arthur Guinness', 'Glen Ridge, New Jersey').
       Do NOT convert entities to ALL CAPITALS. 'ARTHUR GUINNESS' is WRONG. 'Arthur Guinness' is CORRECT.
    4. If the sentence is already well-formed, return it UNCHANGED.

    --- FEW-SHOT EXAMPLES ---
    Input: "I recently bought a house near Wimbledon."
    Output: {{"processed_text": "The user recently bought a house near Wimbledon."}}

    Input: "arthur guinness died in dublin."
    Output: {{"processed_text": "Arthur Guinness died in Dublin."}}

    Input: "Buzz Aldrin was born in Glen Ridge, New Jersey."
    Output: {{"processed_text": "Buzz Aldrin was born in Glen Ridge, New Jersey."}}

    CRITICAL FORMATTING INSTRUCTION:
    You MUST output ONLY a valid JSON object. Do NOT include markdown backticks. Do NOT add conversational text.
    {format_instructions}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg), ("human", "Input: {sentence}")
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | _make_llm() | parser

    def fallback():
        return {"processed_text": original}

    result = _invoke_with_retry(chain, {"sentence": original}, fallback, label="Pre-Processor")
    processed_text = result.get("processed_text", original) or original
    print(f"Processed: {processed_text}")
    return {"sentence": processed_text}


# Fixed FORMAT examples: teach the JSON shape, chaining, and fidelity. These
# stay in the prompt even when per-sentence retrieval is active — losing them
# destabilises parsing.
_NODE1_FORMAT_EXAMPLES = """    --- FEW-SHOT EXAMPLES ---

    Example 1 — CHAINING (the object becomes the next subject):
    Sentence: "Abilene Regional Airport serves the city of Abilene in Jones County, Texas, United States."
    Output: {{"triples": [
      {{"subject": "Abilene Regional Airport", "predicate": "cityServed", "object": "Abilene, Texas"}},
      {{"subject": "Abilene, Texas", "predicate": "isPartOf", "object": "Jones County, Texas"}},
      {{"subject": "Abilene, Texas", "predicate": "country", "object": "United States"}}
    ]}}

    Example 2 — a factually FALSE sentence. Extract it faithfully anyway:
    Sentence: "Hillary Clinton won the 2016 United States presidential election."
    Output: {{"triples": [
      {{"subject": "Hillary Clinton", "predicate": "winner", "object": "2016 United States presidential election"}}
    ]}}
"""

# Static CONVENTION examples: used only when no retrieved examples are supplied,
# preserving the original v13 prompt for direct callers.
_NODE1_STATIC_EXAMPLES = """
    Example 3 — a value object plus a shared subject:
    Sentence: "Trane, whose products include HVAC, was founded in La Crosse, Wisconsin on 1913-01-01."
    Output: {{"triples": [
      {{"subject": "Trane", "predicate": "product", "object": "HVAC"}},
      {{"subject": "Trane", "predicate": "foundationPlace", "object": "La Crosse, Wisconsin"}},
      {{"subject": "Trane", "predicate": "foundingDate", "object": "1913-01-01"}}
    ]}}

    Example 4 — one fact only. Do not invent extras:
    Sentence: "Marie Curie died in Passy, France."
    Output: {{"triples": [
      {{"subject": "Marie Curie", "predicate": "deathPlace", "object": "Passy, France"}}
    ]}}

    Example 5 — DECOMPOSE the place hierarchy into per-level facts:
    Sentence: "108 St Georges Terrace, completed in 1988, is located in Perth, Australia."
    Output: {{"triples": [
      {{"subject": "108 St Georges Terrace", "predicate": "completionDate", "object": "1988"}},
      {{"subject": "108 St Georges Terrace", "predicate": "location", "object": "Perth"}},
      {{"subject": "108 St Georges Terrace", "predicate": "country", "object": "Australia"}}
    ]}}
"""


def _render_retrieved_examples(retrieved):
    """Render TRAIN rows as prompt examples in the domain's own gold
    conventions (NEF, Soru et al. §4.3: per-sentence BM25+MMR retrieval —
    see text2kg_harness.retrieve_examples). Entity names keep their gold
    qualifiers ("Blockbuster (comicsCharacter)") — this is the only mechanism
    that can teach conventions that exist nowhere in DBpedia. Braces are
    doubled for ChatPromptTemplate."""
    def esc(s):
        return str(s).replace("{", "{{").replace("}", "}}")

    blocks = []
    for i, ex in enumerate(retrieved, 1):
        lines = []
        for t in ex.get("triples", []):
            sub = str(t["sub"]).replace("_", " ")
            obj = str(t["obj"])
            obj = obj.strip('"') if obj.startswith('"') else obj.replace("_", " ")
            lines.append(f'      {{{{"subject": "{esc(sub)}", "predicate": "{esc(t["rel"])}", '
                         f'"object": "{esc(obj)}"}}}}')
        blocks.append(
            f'    Example R{i} — from THIS domain (copy its naming and value conventions):\n'
            f'    Sentence: "{esc(ex.get("sent", ""))}"\n'
            '    Output: {{"triples": [\n' + ",\n".join(lines) + "\n    ]}}"
        )
    return "\n\n".join(blocks) + "\n"


def node_1_extractor(state: GraphState):
    """
    v13 — MULTI-TRIPLE extraction. Returns a LIST.

    Text2KGBench sentences state 3.11 facts on average, and 49.3% of them CHAIN:
    the object of one triple becomes the subject of the next.

        "Abilene Regional Airport serves the city of Abilene in Jones County, Texas."
            Abilene_Regional_Airport  cityServed  Abilene,_Texas
            Abilene,_Texas            country     United_States      <- subject moved
            Abilene,_Texas            isPartOf    Jones_County,_Texas <- subject moved

    So the prompt must teach chaining explicitly, not just "extract more".
    """
    print("\n--- [NODE 1] RUNNING MULTI-TRIPLE ATOMIC EXTRACTION ---")
    sentence = state["sentence"]
    parser = JsonOutputParser(pydantic_object=MultiTripleExtraction)

    # ONTOLOGY BOUNDING: Text2KGBench pins each sentence to ONE ontology with a
    # small predicate vocabulary (e.g. 3_airport has 39). Telling the extractor
    # the allowed list up front is the cheapest accuracy win available — it can
    # only name predicates that exist for this domain.
    allowed = state.get("allowed_predicates")
    onto_block = ""
    if allowed:
        onto_block = (
            "\n    --- ALLOWED PREDICATES (MANDATORY VOCABULARY) ---\n"
            "    The predicate field MUST be copied VERBATIM from this list — exact\n"
            "    spelling and camelCase. Do NOT paraphrase or shorten:\n"
            "        'cityServed'  NOT 'city' or 'serves'\n"
            "        'elevationAboveTheSeaLevel'  NOT 'elevation'\n"
            "        'runwayLength'  NOT 'runway' or 'length'\n"
            "    If a fact does not match any predicate below, OMIT that fact entirely.\n"
            "    Allowed predicates:\n    "
            + ", ".join(allowed) + "\n"
        )

    system_msg = """You are an expert Database Architect and Ontology Extractor for DBpedia.
    Extract EVERY subject-predicate-object fact stated in the sentence.

    RULE 0 — FIDELITY (OVERRIDES EVERYTHING):
    Extract ONLY what the sentence says. NEVER correct it using your world knowledge.
    If the sentence says "Hillary Clinton won the 2016 election", the subject is
    "Hillary Clinton" — NOT "Donald Trump", regardless of what actually happened.
    You are a linguistic extractor, not a fact checker.

    RULE 1 — EXTRACT ALL FACTS, NOT JUST THE MAIN ONE (CRITICAL):
    A sentence usually states SEVERAL facts. Typical sentences yield 3. Extract all
    of them. Do not stop at the first.

    RULE 2 — CHAIN THE ENTITIES (CRITICAL):
    The object of one fact is often the subject of the next. Follow the chain.
    Sentences describe a small GRAPH, not a list of independent facts.

    STRICT COMPLIANCE RULES:
    3. NAMED ENTITIES OVER GENERIC NOUNS: extract specific Proper Nouns ('Roger Federer', 'San Siro'), not 'champion' or 'trophy'.
    4. STRIP CONFUSER WORDS: clean, minimal entity names. Strip trailing descriptors like 'mission', 'tournament' UNLESS part of the actual name.
    5. CANONICAL PREDICATES: translate verbs into DBpedia property names ('team', 'location', 'cityServed', 'deathPlace').
    6. PRESERVE GEOGRAPHICAL CONTEXT: 'Abilene, Texas' not 'Abilene'; 'Socialist Party Netherlands' not 'Socialist Party'.
    7. DATES AND NUMBERS ARE VALID OBJECTS: keep them exactly as written in the sentence ('1913-01-01', '546'). Do not skip a fact because its object is a value rather than a name.
{onto_block}
{examples_block}
    CRITICAL FORMATTING INSTRUCTION:
    Output ONLY a valid JSON object with a "triples" array. No markdown backticks, no prose.

    {format_instructions}""".replace("{onto_block}", onto_block)

    retrieved = state.get("fewshot_examples") or []
    if retrieved:
        examples_block = _NODE1_FORMAT_EXAMPLES + "\n" + _render_retrieved_examples(retrieved)
    else:
        examples_block = _NODE1_FORMAT_EXAMPLES + _NODE1_STATIC_EXAMPLES
    system_msg = system_msg.replace("{examples_block}", examples_block)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg), ("human", "Sentence: {sentence}")
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | _make_llm() | parser

    def fallback():
        words = sentence.split()
        return {"triples": [{"subject": words[0] if words else "",
                             "predicate": "relatedTo",
                             "object": words[-1] if words else ""}]}

    result = _invoke_with_retry(chain, {"sentence": sentence}, fallback, label="Extractor")

    raw = result.get("triples") or []
    triples = []
    for t in raw:
        s = (t.get("subject") or "").strip()
        p = (t.get("predicate") or "").strip()
        o = (t.get("object") or "").strip()
        if s and p and o:
            triples.append({"subject": s, "predicate": p, "object": o})

    if not triples:
        print("   🔴 Extractor returned no usable triples.")
        return {"extracted_triples": [], "raw_subject": "", "raw_predicate": "", "raw_object": "",
                "origin_subject": "", "origin_object": ""}

    print(f"Extracted {len(triples)} triple(s):")
    for i, t in enumerate(triples, 1):
        print(f"   {i}. {t['subject']} | {t['predicate']} | {t['object']}")

    # The first triple also seeds the single-triple fields, so the existing
    # Node 2/3/4 chain still runs unchanged when invoked directly.
    first = triples[0]
    return {
        "extracted_triples": triples,
        "raw_subject": first["subject"],
        "raw_predicate": first["predicate"],
        "raw_object": first["object"],
        "origin_subject": first["subject"],
        "origin_object": first["object"]
    }


# ── v14: CONTEXT-QUALIFIED SENSE LOOKUP ──────────────────────────────────────
# The 2_musicalwork/19_film autopsies: 35/70 and 39/45 zeros were bare-title
# resolutions ("Mermaid" -> dbr:Mermaid) where gold wants the parenthesised
# sense (Mermaid_(Train_song)) AND the sentence names the disambiguator
# ("Train's hit Mermaid...", "the 1956 film It's Great to Be Young"). The bare
# surface's top-k in the sf index doesn't even CONTAIN the right sense (it is
# popularity-ranked), so Node 3 never gets the chance to pick it. Generate
# paren-qualified lookup variants from sentence context; page-title surfaces
# like "Mermaid (Train song)" are first-class keys in the index, so a correct
# variant hits exactly.
# Sense-kind indicators: the parenthetical kind DBpedia uses, mapped from the
# words a sentence actually says ("Train's HIT Mermaid" never says "song").
_QUAL_KIND_TRIGGERS = {
    "song":     ("song", "single", "hit", "track"),
    "album":    ("album", "lp", "ep"),
    "film":     ("film", "movie"),
    "band":     ("band", "group"),
    "novel":    ("novel",),
    "book":     ("book",),
    "comic":    ("comic", "comics"),
    "director": ("director", "directed"),
    "actor":    ("actor", "actress"),
    "singer":   ("singer",),
    "writer":   ("writer",),
    "musician": ("musician",),
}
_QUAL_STOPWORDS = {"the", "a", "an", "of", "in"}

def _context_variants(mention, sentence):
    """Paren-qualified lookup variants for a mention, built from its sentence."""
    s = str(sentence or "")
    m = str(mention or "").strip()
    if not m or m.startswith('"'):
        return []
    out = []
    year = re.search(r'\b(?:18|19|20)\d{2}\b', s)
    kinds = [k for k, trigs in _QUAL_KIND_TRIGGERS.items()
             if any(re.search(rf'\b{t}s?\b', s, re.I) for t in trigs)]
    # Capitalised phrases from the sentence that are NOT the mention itself —
    # candidate artist/band/creator names for "<title> (<artist> <kind>)".
    names = []
    # Allow lowercase connectors inside a name ("Year of No Light") and offer
    # an article-stripped twin ("The Year of No Light" -> "Year of No Light").
    _name_re = (r"\b([A-Z][\w.'&-]*"
                r"(?:\s+(?:of|the|and|de|la|von|van|der)\s+[A-Z][\w.'&-]*"
                r"|\s+[A-Z][\w.'&-]*){0,4})")
    for nm in re.finditer(_name_re, s):
        n = re.sub(r"'s?$", "", nm.group(1).strip()).strip(".,;:")  # "Train's" -> "Train"
        for base in (n, re.sub(r"^(?:The|A|An)\s+", "", n)):
            if base and base.lower() != m.lower() and base not in names:
                names.append(base)
    for k in kinds:
        if year:
            out.append(f"{m} ({year.group(0)} {k})")
        for n in names[:6]:
            out.append(f"{m} ({n} {k})")
        out.append(f"{m} ({k})")
    return out[:16]


def node_2_fetcher(state: GraphState):
    print("\n--- [NODE 2] RUNNING TARGETED RETRIEVAL (REDIS BACKED) ---")

    def fetch_candidates_web(query_str, max_results=15):
        if not query_str:
            return []
        redis_key = f"dbpedia:search:{query_str.lower().strip()}"
        if REDIS_AVAILABLE:
            try:
                cached = redis_client.get(redis_key)
                if cached:
                    print(f"   ⚡ Cache Hit! '{query_str}' from Local Redis Index.")
                    return json.loads(cached)[:max_results]
            except Exception:
                pass
        try:
            res = requests.get(
                DBPEDIA_LOOKUP_URL,
                params={"query": query_str, "format": "JSON", "maxResults": max_results},
                timeout=10
            )
            if res.status_code == 200:
                docs = res.json().get("docs", [])
                if REDIS_AVAILABLE:
                    try:
                        redis_client.setex(redis_key, 604800, json.dumps(docs))
                    except Exception:
                        pass
                return docs
        except Exception:
            pass
        return []

    def fetch_candidates_local(query_str, max_results=15):
        if not query_str:
            return []
        try:
            from surface_index import lookup
            results = lookup(query_str, k=max_results)
        except Exception as e:
            # No silent fallback to the web path: that would contaminate the
            # USE_LOCAL_INDEX ablation. Fail loudly instead.
            raise RuntimeError(
                f"USE_LOCAL_INDEX=True but the local sf:* index is unavailable ({e}). "
                "Start Redis on 6380 / import the index, or set USE_LOCAL_INDEX=False."
            ) from e
        if not results:
            # v13 LITERAL PATH: [] from Node 2 is the signal Node 3 turns into
            # unresolved=True, which the Text2KGBench emitter quotes as a
            # literal. The local index makes this signal exact — the web API
            # fuzzy-matches something for almost any string, the sf:* index
            # returns [] precisely when the surface form is not in DBpedia.
            print(f"   ∅ '{query_str}' not in local index → literal path downstream.")
            return []
        docs = []
        for local_name, score in results:
            readable = local_name.replace("_", " ")
            docs.append({
                # Lookup-API doc shape (list-valued fields). Node 3 reads
                # resource/label/comment/type/typeName and recomputes its own
                # scores; "score" is carried for parity and debugging.
                "resource": [f"http://dbpedia.org/resource/{local_name}"],
                "label": [readable],
                # The local index stores no abstracts, and Node 3 drops any
                # candidate with an empty comment — so provide the readable
                # name as the embedding text. Parenthetical qualifiers
                # ("Java (programming language)") carry the disambiguation
                # signal that an abstract would.
                "comment": [readable],
                # No type data locally: peek_types() sees ["Unknown"], a flat
                # 0.10 penalty applied uniformly to every candidate. Measured
                # (3-sentence, 3-path experiment): identical Node 3 winners
                # with and without SPARQL-fetched types.
                "type": [],
                "typeName": [],
                "score": [f"{score:.4f}"],
            })
        return docs

    fetch_candidates = fetch_candidates_local if USE_LOCAL_INDEX else fetch_candidates_web

    sub_c = fetch_candidates(state["raw_subject"])
    obj_c = fetch_candidates(state["raw_object"])

    # v14: merge in context-qualified senses (local index only — the web API
    # fuzzy path would contaminate the ablation). Uses surface_index.lookup
    # directly so a variant miss doesn't print the misleading literal-path line.
    if USE_LOCAL_INDEX:
        from surface_index import lookup as _sf_lookup
        sentence = state.get("sentence") or ""
        for role, raw, cur in (("subject", state["raw_subject"], sub_c),
                               ("object",  state["raw_object"],  obj_c)):
            seen = {d["resource"][0] for d in cur}
            extra = []
            for var in _context_variants(raw, sentence):
                try:
                    hits = _sf_lookup(var, k=2) or []
                except Exception:
                    hits = []
                for local_name, score in hits:
                    uri = f"http://dbpedia.org/resource/{local_name}"
                    if uri in seen:
                        continue
                    seen.add(uri)
                    readable = local_name.replace("_", " ")
                    extra.append({
                        "resource": [uri], "label": [readable],
                        "comment": [readable], "type": [], "typeName": [],
                        "score": [f"{score:.4f}"],
                    })
            if extra:
                print(f"   🎯 Context-qualified {role} senses added: "
                      f"{[d['resource'][0].rsplit('/', 1)[-1] for d in extra]}")
                cur[:0] = extra

    print(f"Retrieved: {len(sub_c)} Subjects | {len(obj_c)} Objects")
    return {"subject_candidates": sub_c, "object_candidates": obj_c}


# ================================================================================
# 4. NODE 3: NEURO-SYMBOLIC MATH FUNNEL
# ================================================================================
def node_3_math_engine(state: GraphState):
    print("\n--- [NODE 3] RUNNING NEURO-SYMBOLIC MATH FUNNEL (v9) ---")
    default_weights = {"vector": 0.75, "lexical": 0.25}
    weights_override = state.get("math_weights")
    raw_sub, raw_pred, raw_obj = state["raw_subject"], state["raw_predicate"], state["raw_object"]
    suggested_pred = state.get("suggested_predicate")
    # v14: hoisted so the literal short-circuit below can consult the CURRENT
    # domain's declared literal pids on every path, including judge-forced.
    literal_preds = state.get("literal_predicates") or set()

    # --- Predicate Linking over the full OWL property space ---
    if suggested_pred and predicate_in_ontology(suggested_pred):
        print(f"🔧 Overriding Predicate Math! Forcing Judge-Suggested Predicate: <{suggested_pred}>")
        best_predicate_uri = suggested_pred
    else:
        if suggested_pred:
            # PREDICATE WHITELIST: the Judge invented one (kidnappedFrom, acquaintance).
            # Ignore it and fall back to real ontology search rather than propagate it.
            print(f"   🛑 Ignoring Judge predicate <{suggested_pred.split('/')[-1]}> — not in dbpedia.owl.")
        pred_embedding = embedding_model.encode(raw_pred)
        # NOTE: this fallback slug is only used if the OWL file failed to load. When the
        # ontology IS loaded, the loop below always overwrites it with a REAL property,
        # so Node 3 can never mint a predicate that doesn't exist.
        best_predicate_uri = f"http://dbpedia.org/ontology/{raw_pred.replace(' ', '_')}"
        highest_pred_sim = -1.0

        # ── v13: ONTOLOGY BOUNDING (FIXED) ────────────────────────────────────
        # Text2KGBench pins each sentence to ONE ontology (3_airport declares 39
        # predicates). We build the search space from the ALLOWED PID LIST directly
        # — NOT by intersecting with dbpedia.owl.
        #
        # THE BUG THIS FIXES: the old filter kept a pid only if it already existed
        # in ontology_vectors (i.e. in our dbpedia.owl snapshot). Our snapshot is
        # missing ~10 airport pids INCLUDING cityServed, so bounding produced only
        # 29 of 39, cityServed among the dropped. Node 1 correctly emitted
        # "cityServed", but Node 3 could not rank it (not in the space) and fell
        # back to the nearest survivor, "city" — the exact miss in every airport
        # sentence. Now we embed any missing pid on the fly so all 39 are rankable
        # and the exact-pid bonus can fire.
        allowed = state.get("allowed_predicates")
        literal_preds = state.get("literal_predicates") or set()
        search_space = ontology_vectors
        if allowed:
            bounded = {}
            for pid in allowed:
                uri = f"http://dbpedia.org/ontology/{pid}"
                if uri in ontology_vectors:
                    bounded[uri] = ontology_vectors[uri]
                else:
                    # Missing from our OWL snapshot — embed from the pid's label so
                    # it is still rankable. Split camelCase into words for a better
                    # vector: "cityServed" -> "city Served".
                    label = re.sub(r'(?<!^)(?=[A-Z])', ' ', pid).replace('_', ' ')
                    bounded[uri] = embedding_model.encode(label)
                # ── DATATYPE FIX (v14: de-globalised) ────────────────────────
                # v13 registered benchmark-declared literal pids into the GLOBAL
                # DATATYPE_PROPERTIES set and never removed them, so one domain's
                # quoting convention leaked into every LATER domain in the same
                # process: 12_monument quotes `country` in gold, which poisoned
                # the global set and made 13_food + 14_writtenwork quote every
                # country object (127 + 77 URI-gold triples destroyed on the
                # 2026-08 server sweep; food 0.66 -> 0.49). The literal
                # short-circuit now consults the PER-DOMAIN literal_preds
                # directly — no global mutation.
            if bounded:
                search_space = bounded
                n_owl = sum(1 for u in bounded if u in ontology_vectors)
                print(f"   🎯 Ontology-bounded: ranking {len(bounded)} allowed predicates "
                      f"({n_owl} from OWL, {len(bounded)-n_owl} embedded on the fly).")

        if search_space:
            raw_pred_norm = raw_pred.lower().replace(' ', '').replace('_', '')
            for uri, ont_vector in search_space.items():
                pid = uri.split('/')[-1]
                v_sim = float(np.dot(pred_embedding, ont_vector) /
                              (np.linalg.norm(pred_embedding) * np.linalg.norm(ont_vector)))
                l_sim = calc_lex_sim(raw_pred, pid.replace('_', ' '))
                combined = 0.5 * v_sim + 0.5 * l_sim

                # ── FIX 3: EXACT PID-MATCH BONUS ──────────────────────────────
                # Node 1 emits the canonical pid ("cityServed"). When the
                # extractor's predicate IS an allowed pid verbatim, that is far
                # stronger evidence than cosine, which otherwise let the shorter
                # "city" outscore "cityServed". Exact (normalised) match trumps all.
                if pid.lower().replace('_', '') == raw_pred_norm:
                    combined += 1.0

                if combined > highest_pred_sim:
                    highest_pred_sim, best_predicate_uri = combined, uri

    print(f"⚡ Linked Predicate URI ──► <{best_predicate_uri}>")

    rule = ONTOLOGY_RULES.get(best_predicate_uri, {"domain": ["Thing"], "range": ["Thing"], "inverted": False})
    domain_types, range_types = rule.get("domain", ["Thing"]), rule.get("range", ["Thing"])

    # ── FIX 2: LITERAL SHORT-CIRCUIT ──────────────────────────────────────────
    # If the linked predicate is an owl:DatatypeProperty, its object is a LITERAL
    # by definition — a number, code, or date. Entity-linking it is not just
    # wasteful, it is wrong: "2194" resolved to <Yeovil_Town_F.C.>, "546" to the
    # YEAR dbr:546. So flag the object as a literal now; the object-scoring path
    # below detects the flag and skips fetching/linking, emitting the raw string.
    # THIS is the one place the declared range IS the correct signal — not for
    # deciding quoting in general (measured false), but for "do not entity-link".
    object_is_literal = (best_predicate_uri in DATATYPE_PROPERTIES
                         or best_predicate_uri.split('/')[-1] in literal_preds)
    if object_is_literal:
        lit = raw_obj.strip().strip('"')
        print(f"   🔢 Datatype predicate → object is a LITERAL, skipping entity-linking: \"{lit}\"")

    def peek_types(candidates):
        for c in candidates:
            types = c.get("type", []) + c.get("typeName", [])
            t_strings = [t.split('/')[-1].split('#')[-1] if isinstance(t, str) else t.get("label", "") for t in types]
            if t_strings:
                return t_strings
        return ["Unknown"]

    # FIX 5: peek_types() was called 4x (twice per role). Cached to 2 calls.
    sub_peek = peek_types(state["subject_candidates"])
    obj_peek = peek_types(state["object_candidates"])

    sub_is_range = any(r.lower() in t.lower() for t in sub_peek for r in range_types) and range_types != ["Thing"]
    sub_is_domain = any(d.lower() in t.lower() for t in sub_peek for d in domain_types) and domain_types != ["Thing"]
    obj_is_domain = any(d.lower() in t.lower() for t in obj_peek for d in domain_types) and domain_types != ["Thing"]
    obj_is_range = any(r.lower() in t.lower() for t in obj_peek for r in range_types) and range_types != ["Thing"]

    sub_candidates_raw, obj_candidates_raw = state["subject_candidates"], state["object_candidates"]

    if (sub_is_range and not sub_is_domain) and (obj_is_domain and not obj_is_range):
        print("   🔄 SCHEMA INVERSION DETECTED: Automatically swapping Subject and Object.")
        expected_sub_types, expected_obj_types = domain_types, range_types
        sub_candidates_raw, obj_candidates_raw = obj_candidates_raw, sub_candidates_raw
        raw_sub, raw_obj = raw_obj, raw_sub
    else:
        expected_sub_types, expected_obj_types = domain_types, range_types

    sentence_emb = embedding_model.encode(state["sentence"])
    # v14: token set of the sentence, for the qualifier-context bonus below.
    sentence_tokens = set(re.findall(r"[\w']+", state["sentence"].lower()))

    # --- IDEA 4: Proactive entropy weighting, per role ---
    sub_weights = weights_override or calculate_entropy_weights(sub_candidates_raw, raw_sub, default_weights)
    obj_weights = weights_override or calculate_entropy_weights(obj_candidates_raw, raw_obj, default_weights)

    fallback_warning = ("⚠️ SYSTEM WARNING: DBpedia API FETCH FAILED OR RETURNED EMPTY ABSTRACTS. "
                        "THIS IS A DUMMY URI. YOU MUST TRIGGER 'RE_SEARCH' OR 'OVERRIDE'. DO NOT APPROVE.")

    def _fallback_candidate(target_raw):
        # v13 LITERAL PATH: "Node 2 found nothing" is the SIGNAL FOR A LITERAL.
        # Measured on the gold data, the declared range does NOT predict quoting:
        #     range=number -> quoted 47x, unquoted 549x
        #     range=Date   -> quoted 61x, unquoted 202x
        # The same predicate goes both ways, and what decides it is whether the
        # surface form resolves:
        #     birthPlace "Faversham, Kent, England"  <- no DBpedia page -> quoted
        #     birthPlace New_Hampshire               <- page exists     -> entity
        # So the literal decision belongs HERE, at the point we already detect
        # "no candidates" — not in a schema lookup. unresolved=True is read by
        # the Text2KGBench wrapper, which emits a quoted literal for it.
        return {
            "uri": f"http://dbpedia.org/resource/{target_raw.replace(' ', '_')}",
            "label": target_raw, "abstract": fallback_warning, "types": ["Thing"],
            "score": 0.50, "v_score": 0.5, "l_score": 0.5, "penalty": 0.0,
            "unresolved": True
        }

    def process_candidates(candidates, target_raw, expected_types, weights):
        if not candidates:
            return [_fallback_candidate(target_raw)]

        context_emb = (0.8 * embedding_model.encode(target_raw)) + (0.2 * sentence_emb)
        processed = []
        for cand in candidates:
            uri = cand.get("resource", [""])[0]
            if not uri or any(p in uri for p in BLACKLIST_URI_PATTERNS):
                continue
            abstract = cand.get("comment", [""])[0]
            if not abstract:
                continue
            label = clean_html(cand.get("label", ["No Label"])[0])
            raw_types = cand.get("type", []) + cand.get("typeName", [])
            sanitized_types = [t.split('/')[-1].split('#')[-1] if isinstance(t, str) else t.get("label", "")
                               for t in raw_types]
            if not sanitized_types:
                sanitized_types = ["Unknown"]

            abs_emb = embedding_model.encode(abstract)
            v_sim = float(np.dot(context_emb, abs_emb) / (np.linalg.norm(context_emb) * np.linalg.norm(abs_emb)))
            # v14: lexical sim against the BASE title — "Mermaid (Train song)"
            # must not lose lexical points to bare "Mermaid" for the mention
            # "Mermaid"; the parenthetical is disambiguation metadata, not name.
            l_sim = calc_lex_sim(target_raw, re.sub(r'\s*\([^)]*\)\s*$', '', label))

            # v14 QUALIFIER-CONTEXT BONUS: a parenthesised sense whose qualifier
            # tokens appear in the sentence is the sense the sentence is about
            # ("Train's hit Mermaid" -> Mermaid_(Train_song), matched={train,
            # song}). Bonus scales with matched-token COUNT so the more specific
            # correct sense ((Train song), 2 hits) beats the vaguer one ((song),
            # 1 hit). A qualifier with ZERO sentence support is mildly penalised
            # — it is some OTHER sense of the surface form.
            qual_bonus = 0.0
            _qm = re.search(r'\(([^)]+)\)\s*$', label)
            if _qm:
                _qtoks = [t for t in re.findall(r"[\w']+", _qm.group(1).lower())
                          if t not in _QUAL_STOPWORDS]
                if _qtoks:
                    _matched = sum(t in sentence_tokens for t in _qtoks)
                    if _matched / len(_qtoks) >= 0.5:
                        qual_bonus = min(0.12 * _matched, 0.36)
                    elif _matched == 0:
                        qual_bonus = -0.10

            base_score = (weights["vector"] * v_sim) + (weights["lexical"] * l_sim) + qual_bonus
            has_type_match = any(expected.lower() in t.lower() for t in sanitized_types for expected in expected_types)

            # ENTITY-TYPE CONTRADICTION GUARD.
            # Diagnostic on entity-heavy zeros showed the resolver picking wildly
            # wrong entities: "108 St Georges Terrace" (a building) -> Manchester
            # City F.C. (a football club). DBpedia Lookup returns weak candidates
            # for obscure names, so a high-connectivity club with a strong base
            # score wins even after the flat 0.25 mismatch penalty.
            # When the candidate's type is in a family that CONTRADICTS the expected
            # family (a sports team / person / organisation where a place/work is
            # expected, or vice-versa), apply a much heavier penalty so an absurd
            # resolution cannot outrank a plausible-but-lower-scoring correct one.
            # Only fires on clear cross-family contradictions, never on near-misses.
            _CONTRADICTION_FAMILIES = {
                "place":  {"soccerclub","sportsteam","person","athlete","band","musicalwork"},
                "person": {"place","populatedplace","building","architecturalstructure","country","city","company","organisation"},
                "work":   {"person","place","country","city","soccerclub","sportsteam"},
            }
            def _fam(types_list):
                tl = " ".join(types_list).lower()
                if any(k in tl for k in ("place","building","city","country","architectural","populated","location","monument","river","mountain")):
                    return "place"
                if any(k in tl for k in ("person","athlete","artist","politician","scientist","astronaut","officeholder")):
                    return "person"
                if any(k in tl for k in ("work","film","album","book","song","musical")):
                    return "work"
                return None
            _exp_fam = _fam(expected_types)
            _cand_fam_types = sanitized_types
            _contradiction = False
            if _exp_fam and not has_type_match:
                cand_low = " ".join(_cand_fam_types).lower()
                bad = _CONTRADICTION_FAMILIES.get(_exp_fam, set())
                if any(b in cand_low for b in bad):
                    _contradiction = True

            if "Unknown" in sanitized_types:
                penalty_applied = 0.10
            elif has_type_match:
                penalty_applied = 0.0
            elif _contradiction:
                penalty_applied = 0.60          # hard cross-family contradiction
            else:
                penalty_applied = 0.25          # ordinary mismatch

            processed.append({
                "uri": uri, "label": label, "abstract": abstract, "types": sanitized_types[:3],
                "score": float(base_score - penalty_applied), "v_score": v_sim, "l_score": l_sim,
                "penalty": penalty_applied
            })

        processed.sort(key=lambda x: x["score"], reverse=True)
        return processed if processed else [_fallback_candidate(target_raw)]

    scored_subjects = process_candidates(sub_candidates_raw, raw_sub, expected_sub_types, sub_weights)

    # FIX 2 continued: for a datatype predicate, the object is the raw literal —
    # do NOT fetch or score entity candidates for it.
    if object_is_literal:
        lit = raw_obj.strip().strip('"')
        scored_objects = [{
            "uri": f'"{lit}"', "label": lit, "abstract": "(literal)",
            "types": ["Literal"], "score": 1.0, "v_score": 1.0, "l_score": 1.0,
            "penalty": 0.0, "is_literal": True
        }]
    else:
        scored_objects = process_candidates(obj_candidates_raw, raw_obj, expected_obj_types, obj_weights)

    # --- IDEA 3: Real joint pairwise scoring ---
    joint_pairs = joint_pairwise_scoring(scored_subjects, scored_objects, sentence_emb, top_k=TOP_K_PER_ROLE)
    if not joint_pairs:
        base = (scored_subjects[0]["score"] + scored_objects[0]["score"]) / 2.0
        joint_pairs = [{
            "subject": scored_subjects[0], "object": scored_objects[0],
            "base_score": base, "joint_context_sim": 0.0, "score": base, "topology": "unchecked"
        }]

    joint_pairs.sort(key=lambda x: x["score"], reverse=True)

    # --- IDEA 2 + structural signal, with the semantic floor guard ---
    if ENABLE_TOPOLOGY_BONUS and not object_is_literal:
        # A literal object ("2194") is not a URI — no graph edges to check.
        print("   🌐 Running Batched Topology Check (single SPARQL query for candidate pool)...")
        sub_uris_pool = [p["subject"]["uri"] for p in joint_pairs]
        obj_uris_pool = [p["object"]["uri"] for p in joint_pairs]
        linked_pairs = batch_check_topology(sub_uris_pool, obj_uris_pool)
        print(f"      Found {len(linked_pairs)} directly-linked URI pairs in candidate pool.")

        for rank, p in enumerate(joint_pairs):
            # Semantic floor: topology REINFORCES a plausible candidate; it never
            # rescues an implausible one (this is what let United_States_Army win in v4).
            if p["base_score"] < MIN_BASE_SCORE_FOR_TOPOLOGY:
                p["topology"] = "skipped (low semantic score)"
                continue

            s_uri, o_uri = p["subject"]["uri"], p["object"]["uri"]
            if (s_uri, o_uri) in linked_pairs or (o_uri, s_uri) in linked_pairs:
                p["score"] += TOPOLOGY_BONUS
                p["topology"] = "direct_link"
            elif ENABLE_BFS_2HOP and rank < CHECK_2HOP_TOP_N:
                if check_2hop_link(s_uri, o_uri):
                    p["score"] += BFS_2HOP_BONUS
                    p["topology"] = "2hop_link"
                else:
                    p["topology"] = "no_link"
            else:
                p["topology"] = "unchecked"

        joint_pairs.sort(key=lambda x: x["score"], reverse=True)

    top_5_triples = [{
        "subject": p["subject"], "predicate": best_predicate_uri,
        "object": p["object"], "topology": p.get("topology", "unchecked")
    } for p in joint_pairs[:5]]

    winner = joint_pairs[0]
    print(f"🥇 Resolved Rank 1 Triple (score={winner['score']:.3f}, topology={winner.get('topology')}):")
    print(f"   Subject: <{winner['subject']['uri']}>\n   Object:  <{winner['object']['uri']}>")

    # RE_SEARCH ROLLBACK: remember the best-scoring attempt across retries.
    # v9 kept whatever the LAST retry produced, so a refinement that made things
    # worse still won. Observed drift on "The python slithered through the dense
    # Amazon": Ball_python (0.833) -> Monty_Python (0.916) -> John_Cleese (0.826),
    # final answer John Cleese.
    prev_best = state.get("best_score_so_far")
    best_triple = state.get("best_triple_so_far")
    best_score = prev_best if prev_best is not None else -1.0

    if winner["score"] > best_score:
        best_score = winner["score"]
        best_triple = dict(top_5_triples[0])
        if prev_best is not None:
            print(f"   📈 New best across retries (prev {prev_best:.3f} → {best_score:.3f}).")
    else:
        print(f"   📉 This retry ({winner['score']:.3f}) is WORSE than the best so far "
              f"({best_score:.3f}). Best kept for rollback.")

    return {
        "top_5_triples": top_5_triples,
        "scored_subjects": scored_subjects,
        "scored_objects": scored_objects,
        "math_weights": weights_override,
        "best_triple_so_far": best_triple,
        "best_score_so_far": best_score
    }


# ================================================================================
# 5. SPARQL VERIFICATION GATE & NODE 4: THE DIAGNOSTIC JUDGE
# ================================================================================
def verify_override_schema(sub_uri, pred_uri, obj_uri, rule):
    """
    Two-tier gate:
      Tier 1 — exact triple ASK. Strongest possible verification.
      Tier 2 — both URIs exist. WEAKER: does NOT verify the predicate at all.

    FIX 4: v8's Tier-2 branch printed "Domain (X) and Range (Y) compliance
    verified" — which was false. Nothing about domain/range was checked there.
    That misleading message is why the wrong `dbo:serves` predicate on the
    Amsterdam row appeared to pass a schema check. Message now honest.
    """
    print("   🔎 SPARQL GATE: Validating Triple & URI Integrity...")
    try:
        r = requests.get(
            DBPEDIA_SPARQL_ENDPOINT,
            params={'query': f"ASK {{ <{sub_uri}> <{pred_uri}> <{obj_uri}> }}", 'format': 'json'},
            timeout=5
        )
        if r.status_code == 200 and r.json().get('boolean'):
            print("   ✅ SPARQL GATE [Tier 1]: Exact Triple verified in DBpedia!")
            return True
    except Exception:
        pass

    # --- INVERSION DETECTION: is the EXACT triple stored the other way round? ---
    # Arm-B autopsy: we emitted (Sri_Jayawardenepura_Kotte, capital, Sri_Lanka)
    # while DBpedia (and gold) store (Sri_Lanka, capital, Kotte). If the inverted
    # ASK verifies, our direction contradicts the ontology's domain/range for
    # this entity pair — return the truthy sentinel "INVERTED" so the caller can
    # swap subject and object instead of accepting a backwards triple.
    def _ask(query):
        """ASK with one retry. Returns True/False, or None if both attempts failed —
        callers must treat None as 'unknown', never as evidence."""
        for attempt in range(2):
            try:
                r = requests.get(
                    DBPEDIA_SPARQL_ENDPOINT,
                    params={'query': query, 'format': 'json'},
                    timeout=10,
                    headers={'User-Agent': 'GSoC-DBpedia-NEF/1.0'}
                )
                if r.status_code == 200:
                    return bool(r.json().get('boolean'))
            except Exception:
                pass
            if attempt == 0:
                time.sleep(1.0)
        return None

    # v14: never direction-swap a (quasi-)symmetric predicate. dishVariation is
    # stored BOTH ways in DBpedia (BLT <-> Bacon_sandwich), so the inverted ASK
    # verifying proves nothing about our direction — and the 13_food autopsy
    # showed swaps here flipping sentence-faithful triples AWAY from gold
    # (#52/#53: gold keeps the sentence's direction).
    _SYMMETRIC_PIDS = {"dishVariation", "associatedBand", "associatedMusicalArtist",
                       "related", "sisterStation", "spouse", "relative"}
    if (pred_uri.rsplit('/', 1)[-1] not in _SYMMETRIC_PIDS
            and _ask(f"ASK {{ <{obj_uri}> <{pred_uri}> <{sub_uri}> }}")):
        print("   🔄 SPARQL GATE: triple verified in the INVERTED direction — "
              "our subject/object order contradicts DBpedia's.")
        return "INVERTED"

    # --- TIER 1.6: range-membership direction check ---
    # (Kotte, capital, Sri_Lanka): NEITHER direction is stored in DBpedia, so
    # the ASKs above are silent. But dbo:capital declares range City, and live
    # types say Kotte IS a dbo:City while Sri_Lanka is not — the emitted
    # object sits outside the predicate's range while the emitted subject sits
    # inside it. That asymmetry is decisive evidence the triple is backwards.
    # Three-state checks: a network failure (None) never triggers a swap.
    # Gated behind TIER16_RANGE_SWAP: against the local store this heuristic
    # over-fires and flips correct-direction triples (see flag comment).
    _range_types = [t for t in (rule or {}).get("range", []) if t and t != "Thing"]
    if TIER16_RANGE_SWAP and _range_types:
        def _in_range(uri):
            saw_failure = False
            for rt in _range_types:
                res = _ask(f"ASK {{ <{uri}> a <http://dbpedia.org/ontology/{rt}> }}")
                if res is True:
                    return True
                if res is None:
                    saw_failure = True
            return None if saw_failure else False

        _obj_in = _in_range(obj_uri)
        if _obj_in is False:
            _sub_in = _in_range(sub_uri)
            if _sub_in is True:
                print(f"   🔄 SPARQL GATE [Tier 1.6]: object is NOT in range "
                      f"{_range_types} but subject IS — triple is backwards.")
                return "INVERTED"

    print("   ⚠️ Exact triple not found. Checking for ANY relation between the entities...")

    # --- TIER 1.5 (NEW): are these two entities connected by any edge at all? ---
    # This accepts facts DBpedia stores under a DIFFERENT predicate than we chose
    # (our dbo:city vs their dbo:cityServed), while rejecting entity pairs that
    # have no relationship whatsoever — the hallucination signature.
    if ENFORCE_RELATION_GATE and check_any_relation(sub_uri, obj_uri):
        print("   ✅ SPARQL GATE [Tier 1.5]: Entities ARE connected in DBpedia under some")
        print("      predicate. Accepted as NOVEL EXTRACTION (our predicate not itself verified).")
        return True

    def check_uri(uri):
        """
        Returns True (exists), False (confirmed absent), or None (COULD NOT CHECK).

        BUG FIX — this used to `return False` on any exception, so a 5-second
        SPARQL timeout was indistinguishable from "this entity is fake". On the
        7_company run that destroyed 44 triples in one domain: real entities like
        dbr:Manila and dbr:Bank were reported subject_exists=False AND
        object_exists=False simultaneously — the fingerprint of a failed query,
        not a missing entity. Recall collapsed to 0.393 while precision held at
        0.497.

        Now: longer timeout, one retry, and a THIRD state for "unknown" so the
        gate can fail OPEN instead of silently deleting good extractions.
        """
        q = f"ASK {{ {{ <{uri}> ?p ?o }} UNION {{ ?s ?p <{uri}> }} }}"
        for attempt in range(2):
            try:
                r = requests.get(
                    DBPEDIA_SPARQL_ENDPOINT,
                    params={'query': q, 'format': 'json'},
                    timeout=15,
                    headers={'User-Agent': 'GSoC-DBpedia-NEF/1.0'}
                )
                if r.status_code == 200:
                    return bool(r.json().get('boolean'))
                # 429/503/403 etc: endpoint is unhappy, not a verdict on the URI
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                return None
            except Exception:
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                return None
        return None

    sub_exists, obj_exists = check_uri(sub_uri), check_uri(obj_uri)

    # FAIL OPEN when the endpoint could not be reached. An unreachable SPARQL
    # server must not be able to delete a correct triple — that is infrastructure
    # noise, not evidence of hallucination. We only reject on a CONFIRMED False.
    if sub_exists is None or obj_exists is None:
        print("   ⚠️  SPARQL endpoint unreachable for the existence check — "
              "cannot verify URIs.")
        print("      Failing OPEN: accepting as UNVERIFIED rather than discarding "
              "a possibly-correct triple.")
        return True

    if not ENFORCE_RELATION_GATE:
        # Ablation escape hatch: reproduces v9's permissive behaviour on demand.
        if sub_exists and obj_exists:
            print("   ✅ SPARQL GATE [Tier 2 / relation gate DISABLED]: Both URIs exist.")
            return True

    if sub_exists and obj_exists:
        # ── FAITHFUL EXTRACTION MODE ──────────────────────────────────────────
        # Both entities are real and the predicate is a valid dbo property (the
        # predicate whitelist already guaranteed that upstream). DBpedia simply
        # doesn't store THIS edge. For a faithful-extraction task that is NOT a
        # hallucination — it is a true triple DBpedia lacks. Accept as UNVERIFIED.
        if FAITHFUL_EXTRACTION_MODE:
            print("   ✅ SPARQL GATE [Tier 2 / FAITHFUL MODE]: Both entities real, predicate")
            print("      valid, but DBpedia has no edge. Accepted as UNVERIFIED extraction")
            print("      (the sentence asserts it; the benchmark scores sentence-fidelity).")
            return True

        # Strict / KB-completion mode: v9 RETURNED TRUE HERE, which is why
        # <John_Cleese> <species> <Amazon_rainforest> passed. Rejecting is correct
        # ONLY when the objective is DBpedia membership, not sentence fidelity.
        print("   🛑 SPARQL GATE BLOCKED [Tier 2]: Both URIs are real, but they have NO")
        print("      relationship in DBpedia. This is the hallucination signature")
        print("      (e.g. John Cleese + Amazon rainforest). REJECTED.")
        return False

    # OBJECT-ONLY MISS: the subject is real but the object URI has no triples in
    # the store. 2026-08-04 local-gate run, test_98: <Government_of_Addis_Ababa>
    # doesn't exist as a URI (gold wants the QUOTED LITERAL "Government of Addis
    # Ababa") and the gate deleted the whole currentTenants triple. Deleting is
    # the wrong recovery — this is exactly the not-in-index -> literal rule, one
    # step later. Return a sentinel so the caller demotes the object to a quoted
    # literal instead of dropping a faithful extraction. Subject-side misses keep
    # blocking: a triple's subject must be a URI, there is nothing to demote to.
    if sub_exists is True and obj_exists is False:
        # v14 INDEX CROSS-CHECK: the 16.2M-mention sf index and the local store
        # are INDEPENDENT authorities on entity reality, and each has gaps the
        # other covers. An object absent from the store but present in the index
        # is a store coverage gap, not a fake URI — demoting it to a literal
        # loses against gold, which writes real entities as URIs (FIMI,
        # Crucial_Blast, Michael_R._Burns in the autopsies). Objects absent from
        # BOTH keep demoting (Government_of_Addis_Ababa — gold agrees: literal).
        try:
            from surface_index import lookup as _sf_lookup
            _oname = obj_uri.rsplit('/', 1)[-1]
            _in_index = any(ln == _oname for ln, _ in
                            (_sf_lookup(_oname.replace('_', ' '), k=5) or []))
        except Exception:
            _in_index = False
        if _in_index:
            print("   ✅ SPARQL GATE: object missing from the LOCAL STORE but present "
                  "in the sf index — store coverage gap, not a fake URI. "
                  "Accepted as UNVERIFIED.")
            return True
        print("   🔁 SPARQL GATE: subject is real but the object URI does not exist "
              "in the store — demoting object to a quoted literal instead of rejecting.")
        return "OBJ_MISSING"

    print(f"   🛑 SPARQL GATE BLOCKED: Dead/hallucinated URI "
          f"(subject_exists={sub_exists}, object_exists={obj_exists}).")
    return False


def node_4_judge(state: GraphState):
    print("\n--- [NODE 4] CONVENING THE STRICT DIAGNOSTIC JUDGE PANEL ---")
    parser = JsonOutputParser(pydantic_object=JudgeDecisionSchema)
    current_retry = state.get("retry_count", 0)

    if not state.get("top_5_triples"):
        return {"validation_status": "API_FAILURE", "retry_count": current_retry + 1}

    rank_1 = state["top_5_triples"][0]
    rank_1_pred = rank_1["predicate"]
    rule = ONTOLOGY_RULES.get(rank_1_pred, {"domain": ["Thing"], "range": ["Thing"]})

    # BUG A FIX (part 1): decide this BEFORE the LLM call, not after.
    # v9 let the Judge answer normally (it chose RE_SEARCH), and only THEN did
    # Python flip the status to OVERRIDE. But an RE_SEARCH verdict carries an
    # EMPTY winning_triple_string — the LLM was never asked for override URIs —
    # so the "override" silently fell through to Rank 1. The Judge would write
    # "Donald Trump won the election" and the pipeline emitted Hillary Clinton.
    is_final_attempt = current_retry >= 2

    evaluation_deck = (
        f"Original Sentence: {state['original_sentence']}\n"
        f"Pre-processed Sentence: {state['sentence']}\n"
        f"Retry Attempt: {current_retry} / 2\n\n"
        f"--- SCHEMA VALIDATION DATA FOR RANK 1 ---\n"
        f"Predicate: <{rank_1_pred}>\n"
        f"Legally requires Subject to be type: {rule['domain']}\n"
        f"Legally requires Object to be type: {rule['range']}\n\n"
        f"--- CURRENT TOP 5 CANDIDATES (with graph-topology status) ---\n"
    )
    for idx, triple in enumerate(state["top_5_triples"]):
        evaluation_deck += (
            f"Rank {idx + 1}:\n"
            f"   - URIs: <{triple['subject']['uri']}> <{triple['predicate']}> <{triple['object']['uri']}>\n"
            f"   - Subject Types: {triple['subject'].get('types', ['Unknown'])}\n"
            f"   - Object Types: {triple['object'].get('types', ['Unknown'])}\n"
            f"   - Graph Topology: {triple.get('topology', 'unchecked')}\n"
            f"   - Subj Abstract: {triple['subject']['abstract'][:100]}...\n"
            f"   - Obj Abstract: {triple['object']['abstract'][:100]}...\n\n"
        )

    # BUG A FIX (part 2): on the last attempt, TELL the Judge it is the last
    # attempt and demand override URIs. v9 never gave the Judge this turn — it
    # let the LLM pick RE_SEARCH and then silently relabelled that verdict as
    # OVERRIDE in Python, at which point winning_triple_string was empty and the
    # code fell back to Rank 1. The Judge's parametric knowledge never reached
    # the output, which is exactly why "Hillary Clinton won 2016" survived.
    if is_final_attempt:
        evaluation_deck += (
            "\n🚨🚨 FINAL ATTEMPT — RETRY LIMIT REACHED 🚨🚨\n"
            "You have NO retries left. RE_SEARCH, ADJUST_MATH and ADJUST_PREDICATE are\n"
            "FORBIDDEN on this turn and will be discarded.\n\n"
            "Output status='OVERRIDE' and write into 'winning_triple_string' the URIs\n"
            "that best represent WHAT THE SENTENCE SAYS.\n\n"
            "REMINDER — FAITHFUL EXTRACTION:\n"
            "  The subject and object MUST be the entities the sentence NAMES.\n"
            "  Do NOT substitute a different entity because you believe the sentence\n"
            "  is factually wrong. If the sentence says Hillary Clinton won, the\n"
            "  subject is <http://dbpedia.org/resource/Hillary_Clinton>. Writing\n"
            "  Donald_Trump here is a FAILURE — a different Python layer decides\n"
            "  whether the claim is supported by DBpedia. That is not your job.\n\n"
            "  The predicate MUST be a real DBpedia ontology property.\n\n"
            "If the entities named in the sentence cannot be resolved to DBpedia URIs\n"
            "at all, set winning_triple_string to the exact string \"UNRESOLVED\" and\n"
            "explain in feedback_instruction. Do NOT use \"UNRESOLVED\" merely because\n"
            "the sentence is false or absurd — extract it faithfully and let the gate\n"
            "reject it.\n"
        )

    # NEW: Few-shot examples. The Judge was previously the ONLY node making a
    # 5-way categorical decision across 7 fields with ZERO examples, while Node 1
    # (an easier task) had three. That asymmetry produced the observed failures:
    # "Diagnostic: RE_SEARCH" (status echoed into the reasoning field), empty
    # suggested_predicate on ADJUST_PREDICATE, and hard JSON parse errors.
    system_msg = """You are a strict Diagnostic Meta-Reasoner and Knowledge Graph Judge.

    ═══════════════════════════════════════════════════════════════════════
    RULE 0 — FIDELITY TO THE SENTENCE. THIS OVERRIDES EVERY OTHER RULE.
    ═══════════════════════════════════════════════════════════════════════
    You are a LINGUISTIC processor, NOT a fact checker.
    Your ONLY job: does the triple faithfully represent WHAT THE SENTENCE SAYS?

    You must NEVER change a Subject or Object because you believe the sentence is
    factually wrong. Your world knowledge is IRRELEVANT to this task.

      Sentence: "Hillary Clinton won the 2016 United States presidential election."
      CORRECT : <Hillary_Clinton> <winner> <2016_United_States_presidential_election>
      WRONG   : <Donald_Trump> <winner> <2016_United_States_presidential_election>

    Yes, Donald Trump actually won. That is NOT your concern. The sentence says
    Hillary Clinton, so the triple says Hillary Clinton. A separate Python layer
    checks the claim against DBpedia and marks it VERIFIED or UNVERIFIED. If you
    "helpfully" correct the fact, you have DESTROYED the extraction and a Python
    guard will reject your verdict anyway.

    You may ONLY refine WORD SENSE — the same entity, a better URI:
      ALLOWED : "Paris kidnapped Helen"  <Paris> -> <Paris_(mythology)>   same referent
      FORBIDDEN: "Mars ate lunch"        <Mars>  -> <Bruno_Mars>          different entity
      FORBIDDEN: "Hillary Clinton won"   <Hillary_Clinton> -> <Donald_Trump>

    Absurd sentences get extracted faithfully too. "Mars ate lunch with Donald
    Trump" -> extract <Mars> and <Donald_Trump>. You do NOT refuse because planets
    cannot eat. The SPARQL gate will find no relation and reject it. That is the
    gate's job, not yours.

    RULE 1: PREDICATE SCRUTINY
    If the Subject/Object types don't match what the Predicate legally requires, that is a
    "Predicate Hijack". Output ADJUST_PREDICATE.
    The predicate you suggest MUST be a real DBpedia ontology property that EXISTS in
    dbpedia.owl. Do NOT invent one. <dbo:kidnappedFrom> and <dbo:acquaintance> DO NOT
    EXIST and will be rejected. If no real property fits, choose the closest real one.

    RULE 2: MISSING DATA TOLERANCE
    If the types are 'Unknown' but the entities are the ones the sentence names, ignore
    the 'Unknown' type and output 'APPROVED'.

    RULE 3: GRAPH TOPOLOGY IS EVIDENCE OF EXISTENCE, NOT OF FIDELITY
    'direct_link'/'2hop_link' means the entities are connected in DBpedia. That does NOT
    make a triple faithful, and 'no_link' does NOT make it unfaithful — a true statement
    absent from DBpedia still has no link. Judge FIDELITY; let the gate judge support.

    RULE 4: SELF-CONSISTENCY (MANDATORY)
    If your reasoning identifies ANY problem with Rank 1, your status MUST NOT be 'APPROVED'.
    Never write reasoning that says something is wrong and then approve it anyway.

    DECISION PARADIGM:
    1. Rank 1 faithfully represents the sentence: 'APPROVED'.
    2. Predicate doesn't match the sentence's relation: 'ADJUST_PREDICATE'
       -> 'suggested_predicate' MUST be a non-empty, REAL dbpedia.org/ontology/ URI.
    3. Subject/Object is the WRONG SENSE of the named entity, and a better sense is in
       the Top 5: 'ADJUST_MATH'. (Same entity, better URI — never a different entity.)
    4. Candidates are irrelevant/generic, or an abstract shows '⚠️ SYSTEM WARNING': 'RE_SEARCH'
       -> 'new_subject_query'/'new_object_query' MUST be concise Named Entities (1-3 words)
          derived FROM THE SENTENCE. Never query for an entity the sentence never mentioned.
    5. Failsafe (Retry == 2): 'OVERRIDE' with URIs for the entities THE SENTENCE NAMES.

    --- FEW-SHOT EXAMPLES ---

    Example 1 — correct triple, approve:
    Rank 1: <http://dbpedia.org/resource/A.C._Milan> <http://dbpedia.org/ontology/homeStadium> <http://dbpedia.org/resource/San_Siro>, topology: direct_link
    Output: {{"status": "APPROVED", "feedback_instruction": "A.C. Milan's home stadium is San Siro. Subject and object types match the predicate, and a direct graph link confirms the relationship.", "target_uri": "", "imposter_uri": "", "suggested_predicate": "", "new_subject_query": "", "new_object_query": "", "winning_triple_string": ""}}

    Example 2 — wrong predicate (note: suggested_predicate MUST be filled):
    Sentence: "Anders Celsius died in Uppsala." Rank 1 predicate: <http://dbpedia.org/ontology/deathCause>
    Output: {{"status": "ADJUST_PREDICATE", "suggested_predicate": "http://dbpedia.org/ontology/deathPlace", "feedback_instruction": "The sentence describes the place of death, not the cause of death. deathCause is semantically wrong here; deathPlace is correct.", "target_uri": "", "imposter_uri": "", "new_subject_query": "", "new_object_query": "", "winning_triple_string": ""}}

    Example 3 — object entity is completely wrong, candidates are bad:
    Sentence: "Alan Shepard was a crew member of the Apollo 14 mission." Rank 1 object: <http://dbpedia.org/resource/United_States_Army>
    Output: {{"status": "RE_SEARCH", "new_subject_query": "Alan Shepard", "new_object_query": "Apollo 14", "feedback_instruction": "The object resolved to United States Army, which is unrelated to the Apollo 14 spaceflight. The trailing word 'mission' likely polluted the search. Re-querying with the clean entity name.", "target_uri": "", "imposter_uri": "", "suggested_predicate": "", "winning_triple_string": ""}}

    Example 4 — correct SENSE of the same entity exists at a lower rank:
    Sentence mentions the Netherlands. Rank 1 object: <http://dbpedia.org/resource/Socialist_Party_(Portugal)>, Rank 3 object: <http://dbpedia.org/resource/Socialist_Party_(Netherlands)>
    Output: {{"status": "ADJUST_MATH", "target_uri": "http://dbpedia.org/resource/Socialist_Party_(Netherlands)", "imposter_uri": "http://dbpedia.org/resource/Socialist_Party_(Portugal)", "feedback_instruction": "The sentence explicitly specifies the Netherlands, but Rank 1 resolved to the Portuguese party. Same named entity, correct sense is at Rank 3.", "suggested_predicate": "", "new_subject_query": "", "new_object_query": "", "winning_triple_string": ""}}

    Example 5 — THE SENTENCE IS FACTUALLY FALSE. APPROVE IT ANYWAY. (CRITICAL)
    Sentence: "Hillary Clinton won the 2016 United States presidential election."
    Rank 1: <http://dbpedia.org/resource/Hillary_Clinton> <http://dbpedia.org/ontology/winner> <http://dbpedia.org/resource/2016_United_States_presidential_election>
    Output: {{"status": "APPROVED", "feedback_instruction": "The triple faithfully represents the sentence: it names Hillary Clinton as the subject and the 2016 election as the object, and 'winner' matches the verb 'won'. I am aware Donald Trump actually won, but factual accuracy is not my role — the extraction is faithful, and the SPARQL gate will mark this claim unverified.", "target_uri": "", "imposter_uri": "", "suggested_predicate": "", "new_subject_query": "", "new_object_query": "", "winning_triple_string": ""}}

    Example 6 — THE SENTENCE IS ABSURD. STILL EXTRACT IT FAITHFULLY. (CRITICAL)
    Sentence: "Mars ate lunch with Donald Trump."
    Rank 1: <http://dbpedia.org/resource/Mars> <http://dbpedia.org/ontology/colleague> <http://dbpedia.org/resource/Donald_Trump>
    Output: {{"status": "APPROVED", "feedback_instruction": "The sentence names 'Mars' and 'Donald Trump', and Rank 1 resolves exactly those two entities. A planet cannot eat lunch, but refusing on absurdity is not my role and substituting Bruno Mars would be an entity swap. The extraction is faithful; the SPARQL gate will find no relation between these two entities and reject the triple.", "target_uri": "", "imposter_uri": "", "suggested_predicate": "", "new_subject_query": "", "new_object_query": "", "winning_triple_string": ""}}

    FORMATTING RULES:
    - 'feedback_instruction' MUST contain your REASONING in plain English. NEVER just repeat the status word (writing "RE_SEARCH" or "Approved" as your reasoning is INVALID).
    - If status is 'ADJUST_PREDICATE', 'suggested_predicate' MUST be a full non-empty DBpedia ontology URI.
    - Output ONLY a valid JSON object. No markdown backticks. No conversational text.

    {format_instructions}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg), ("human", "{payload}")
    ]).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | _make_llm() | parser

    def fallback_verdict():
        return {
            "status": "APPROVED",
            "feedback_instruction": "Emergency approval due to repeated API/parse failure.",
            "winning_triple_string": "", "target_uri": "", "imposter_uri": "",
            "suggested_predicate": "", "new_subject_query": "", "new_object_query": ""
        }

    verdict = _invoke_with_retry(chain, {"payload": evaluation_deck}, fallback_verdict, label="Judge")
    verdict = check_judge_self_consistency(verdict)

    v_status = verdict.get('status', 'APPROVED')
    v_feedback = verdict.get('feedback_instruction', 'Fallback approval.')

    # BUG A FIX (part 3): if the Judge still tries to loop on the final attempt,
    # don't silently relabel its RE_SEARCH verdict as OVERRIDE (that produced an
    # empty winning_triple_string and a fall-through to Rank 1). Ask it ONE more
    # time with an unambiguous instruction, so the override URIs actually exist.
    if is_final_attempt and v_status in ["RE_SEARCH", "ADJUST_MATH", "ADJUST_PREDICATE"]:
        print(f"   ⚠️ Judge chose {v_status} on the final attempt. Re-asking for explicit OVERRIDE URIs...")
        forced_deck = evaluation_deck + (
            f"\n\n❌ You returned '{v_status}', which is FORBIDDEN on the final attempt.\n"
            f"Your own reasoning was: \"{v_feedback}\"\n"
            f"Now ACT on that reasoning: output status='OVERRIDE' and put the correct\n"
            f"full DBpedia URIs in winning_triple_string. Nothing else is acceptable.\n"
        )
        retry_verdict = _invoke_with_retry(
            chain, {"payload": forced_deck}, lambda: None, label="Judge-FinalOverride"
        )
        if retry_verdict and retry_verdict.get("winning_triple_string"):
            verdict = retry_verdict
            v_feedback = verdict.get('feedback_instruction', v_feedback)
            print("   ✅ Judge supplied explicit OVERRIDE URIs on the second ask.")
        else:
            # Loud, not silent: v9 hid this behind a cheerful OVERRIDE banner.
            print("   🔴 Judge FAILED to supply override URIs. Falling back to Rank 1 — "
                  "this triple is NOT LLM-verified.")
        v_status = "OVERRIDE"

    # FIX 3: coerce to "" (state.get can return None) and guard the null-predicate loop.
    # check_judge_self_consistency() deliberately blanks suggested_predicate, and the
    # Judge itself sometimes returns ADJUST_PREDICATE with nothing — v8 then printed
    # "Forcing URI: <None>" and burned a retry doing nothing.
    llm_suggested_pred = verdict.get('suggested_predicate') or ""
    suggested_pred = llm_suggested_pred or state.get('suggested_predicate') or ""

    if v_status == "ADJUST_PREDICATE" and not suggested_pred:
        print("   ⚠️ Judge returned ADJUST_PREDICATE with no URI. Downgrading to RE_SEARCH to avoid a null loop.")
        v_status = "RE_SEARCH"

    # ── OPTION A GUARD 1: PREDICATE WHITELIST ─────────────────────────────────
    # The Judge invented <dbo:kidnappedFrom> and <dbo:acquaintance> in real runs.
    # A prompt asked it not to; it did anyway. This is the guarantee.
    if v_status == "ADJUST_PREDICATE" and suggested_pred and not predicate_in_ontology(suggested_pred):
        print(f"   🛑 ENTITY/PREDICATE LOCK: Judge suggested <{suggested_pred}>, which does NOT")
        print("      exist in dbpedia.owl. Rejecting the invented predicate and keeping the")
        print("      math-derived one.")
        suggested_pred = ""
        v_status = "APPROVED" if is_final_attempt else "RE_SEARCH"

    # ── OPTION A GUARD 2: ENTITY LOCK on ADJUST_MATH ──────────────────────────
    # ADJUST_MATH may only move to a different SENSE of the same named entity.
    # Swapping Hillary->Trump or Mars->Bruno_Mars is blocked here in Python.
    if ENFORCE_ENTITY_LOCK and v_status == "ADJUST_MATH":
        tgt = verdict.get('target_uri', '') or ''
        # FIX B: anchor to the ORIGINAL sentence mentions, not the RE_SEARCH-mutated query.
        _o_sub, _o_obj = _origin_mentions(state)
        sub_ok, sub_s = entity_matches_mention(tgt, _o_sub)
        obj_ok, obj_s = entity_matches_mention(tgt, _o_obj)
        if tgt and not (sub_ok or obj_ok):
            print(f"   🛑 ENTITY LOCK: Judge tried to swap in <{tgt.split('/')[-1]}>, which does not")
            print(f"      lexically match the SENTENCE's entities '{_o_sub}' / "
                  f"'{_o_obj}' (best {max(sub_s, obj_s):.2f} < {ENTITY_LOCK_THRESHOLD}).")
            print("      This is an ENTITY SUBSTITUTION, not a sense refinement. Verdict VOID.")
            v_status = "APPROVED"  # keep the faithful math result

    print(f"👨‍⚖️ JUDGE VERDICT ──► Status: {v_status}\n📝 Diagnostic: {v_feedback}")

    new_weights = state.get("math_weights")

    # LITERAL OBJECT: a datatype-predicate object is a quoted literal, not a URI.
    # Build the string so the gate sees only TWO <...> URIs (subject, predicate)
    # and the literal in quotes — and mark it so the gate skips SPARQL entirely.
    rank_1_obj = rank_1["object"]
    rank_1_obj_is_literal = bool(rank_1_obj.get("is_literal"))
    if rank_1_obj_is_literal:
        rank_1_str = f"<{rank_1['subject']['uri']}> <{rank_1_pred}> {rank_1_obj['uri']}"
    else:
        rank_1_str = f"<{rank_1['subject']['uri']}> <{rank_1_pred}> <{rank_1_obj['uri']}>"
    override_str = (verdict.get("winning_triple_string") or "").strip()

    # ROLLBACK: if an earlier retry scored better than this one, prefer it.
    # Guards against RE_SEARCH drifting away from the answer (Ball_python -> John_Cleese).
    best_prev = state.get("best_triple_so_far")
    if best_prev and v_status in ["APPROVED", "OVERRIDE"]:
        best_prev_str = (f"<{best_prev['subject']['uri']}> <{best_prev['predicate']}> "
                         f"<{best_prev['object']['uri']}>")
        if best_prev_str != rank_1_str and (state.get("best_score_so_far") or 0) > 0:
            print(f"   ↩️  Rollback available — best scoring attempt was: {best_prev_str}")
            rank_1_str = best_prev_str

    # UNRESOLVED = the sentence's entities cannot be mapped to DBpedia URIs at all.
    # NOT "the sentence is false" and NOT "the sentence is absurd" — those get
    # extracted faithfully and rejected by the gate. (v10 used NO_VALID_TRIPLE for
    # absurdity, which was the Judge exercising a truth veto it should not have.)
    if v_status == "OVERRIDE" and override_str == "UNRESOLVED":
        print("   🛑 Judge declared UNRESOLVED — the sentence's entities have no DBpedia URIs.")
        return {
            "validation_status": "REJECTED_UNRESOLVED",
            "feedback_instruction": v_feedback,
            "final_triple": "Extraction Failed: entities named in the sentence could not be resolved to DBpedia URIs.",
            "retry_count": current_retry + 1,
            "math_weights": new_weights,
            "suggested_predicate": suggested_pred
        }

    # ── OPTION A GUARD 3: ENTITY LOCK on OVERRIDE (the Hillary->Trump fix) ─────
    # This is the exact path that emitted <Donald_Trump> for a sentence naming
    # Hillary Clinton. The prompt now forbids it; this makes it impossible.
    if ENFORCE_ENTITY_LOCK and v_status == "OVERRIDE" and override_str:
        ov_uris = re.findall(r'<(.*?)>', override_str)
        if len(ov_uris) == 3:
            # FIX B: anchor to the ORIGINAL sentence mentions.
            _o_sub, _o_obj = _origin_mentions(state)
            s_ok, s_score = entity_matches_mention(ov_uris[0], _o_sub)
            o_ok, o_score = entity_matches_mention(ov_uris[2], _o_obj)
            pred_ok = predicate_in_ontology(ov_uris[1])

            if not s_ok or not o_ok:
                bad = []
                if not s_ok:
                    bad.append(f"subject <{ov_uris[0].split('/')[-1]}> vs sentence's "
                               f"'{_o_sub}' ({s_score:.2f})")
                if not o_ok:
                    bad.append(f"object <{ov_uris[2].split('/')[-1]}> vs sentence's "
                               f"'{_o_obj}' ({o_score:.2f})")
                print("   🛑 ENTITY LOCK VIOLATION — Judge substituted an entity the sentence")
                print("      never named. This is fact-correction, not extraction. REJECTED:")
                for b in bad:
                    print(f"        • {b}")
                print("      Falling back to the faithful math-derived triple.")
                override_str = ""
            elif not pred_ok:
                print(f"   🛑 PREDICATE LOCK VIOLATION — Judge's override used "
                      f"<{ov_uris[1].split('/')[-1]}>, which is not in dbpedia.owl.")
                print("      Falling back to the faithful math-derived triple.")
                override_str = ""
            else:
                print(f"   ✅ Entity lock passed (subject {s_score:.2f}, object {o_score:.2f}) — "
                      f"override stays faithful to the sentence.")

    # FIX 2 (kept): parenthesized. `A or B and not C` parsed as `A or (B and not C)`.
    if v_status == "APPROVED" or (v_status == "OVERRIDE" and not override_str):
        final_triple_str = rank_1_str
    else:
        final_triple_str = override_str or rank_1_str

    # Report when an "OVERRIDE" is really just Rank 1 wearing a hat.
    if v_status == "OVERRIDE":
        if override_str and override_str != rank_1_str:
            print(f"   🧠 OVERRIDE uses the Judge's own knowledge: {final_triple_str}")
        else:
            print("   ⚠️ OVERRIDE degenerated to Rank 1 (Judge gave no distinct triple).")

    # UNIVERSAL SPARQL GATE WITH BOUNCE-BACK
    # gate_rejected tracks WHY we are in RE_SEARCH. The gate's recovery path sets
    # RE_SEARCH with the SAME queries as a default, so the repeat-guard below would
    # otherwise see "identical queries" and approve a triple the gate just refused.
    gate_rejected = False
    if v_status in ["APPROVED", "OVERRIDE"]:
        # LITERAL OBJECTS bypass the URI-pair gate: there is no object URI to
        # verify and no edge to check. A datatype triple like
        #   <Angola_International_Airport> <runwayLength> "3800"
        # is valid by construction — Node 1 stated the value, the predicate is a
        # real DatatypeProperty. Accept it as-is.
        if rank_1_obj_is_literal and v_status == "APPROVED":
            print("   🔢 Literal-object triple — skipping URI/relation gate (nothing to resolve).")
        else:
            uris = re.findall(r'<(.*?)>', final_triple_str)
            if len(uris) == 3:
                gate_result = verify_override_schema(uris[0], uris[1], uris[2], rule)
                if gate_result == "INVERTED":
                    # DBpedia stores this fact the other way round — swap so the
                    # emitted triple follows the ontology direction.
                    final_triple_str = f"<{uris[2]}> <{uris[1]}> <{uris[0]}>"
                    print(f"   🔄 Subject/object SWAPPED to match ontology direction: {final_triple_str}")
                elif gate_result == "OBJ_MISSING":
                    # Subject is real, object URI absent from the store: keep the
                    # faithful extraction with the sentence's own words as a quoted
                    # literal. Bare-quoted form so the harness literal normalizer
                    # (quote hints / value formats) applies exactly as for a
                    # Node-2-miss literal.
                    _raw_obj = (state.get('raw_object') or uris[2].split('/')[-1].replace('_', ' ')).strip()
                    final_triple_str = f'<{uris[0]}> <{uris[1]}> "{_raw_obj}"'
                    print(f"   🔁 Object demoted to literal: {final_triple_str}")
                if not gate_result:
                    gate_rejected = True
                    if current_retry < 2:
                        print("   🛑 SPARQL GATE BLOCKED! Invalid URI. Forcing RE_SEARCH to recover...")
                        v_status = "RE_SEARCH"
                        if not verdict.get('new_subject_query'):
                            verdict['new_subject_query'] = state['raw_subject']
                        if not verdict.get('new_object_query'):
                            verdict['new_object_query'] = state['raw_object']
                    else:
                        v_status = "REJECTED_BY_SPARQL"
                        final_triple_str = "Extraction Failed: SPARQL Gate blocked an unverified LLM hallucination."
            else:
                gate_rejected = True
                if current_retry < 2:
                    print("   🛑 URI FORMAT ERROR! Forcing RE_SEARCH to recover...")
                    v_status = "RE_SEARCH"
                    if not verdict.get('new_subject_query'):
                        verdict['new_subject_query'] = state['raw_subject']
                    if not verdict.get('new_object_query'):
                        verdict['new_object_query'] = state['raw_object']
                else:
                    v_status = "REJECTED_BY_SPARQL"
                    final_triple_str = "Extraction Failed: Invalid URI formatting by LLM."

    if v_status == "ADJUST_MATH":
        print("🚨 Math Sabotage Detected! Initializing Algebraic Solver...")
        t_uri, i_uri = verdict.get('target_uri', ''), verdict.get('imposter_uri', '')
        t_data = next((item for item in state['scored_subjects'] if item["uri"] == t_uri), None)
        i_data = next((item for item in state['scored_subjects'] if item["uri"] == i_uri), None)
        target_role = "subject"
        if not (t_data and i_data):
            t_data = next((item for item in state['scored_objects'] if item["uri"] == t_uri), None)
            i_data = next((item for item in state['scored_objects'] if item["uri"] == i_uri), None)
            target_role = "object"

        if t_data and i_data:
            new_weights = calculate_optimal_weights(
                t_data["v_score"], t_data["l_score"],
                i_data["v_score"], i_data["l_score"],
                t_data["penalty"], i_data["penalty"]
            )
        else:
            # ── FIX 2: ADJUST_MATH → RE_SEARCH ESCALATION ─────────────────────
            # ADJUST_MATH works by RE-WEIGHTING vector-vs-lexical to promote a
            # candidate that is ALREADY in the pool. If the Judge names a URI that
            # Node 2 never fetched, there is nothing to re-weight — the solver runs,
            # finds nothing, changes nothing, and burns a retry.
            #
            # Real trace (v11, "The python slithered through the dense Amazon"):
            #     ADJUST_MATH → "a more suitable entity would be the genus Python"
            #     ⚠️ Judge named URIs not found in scored candidates. Keeping current weights.
            #     → next retry produced the IDENTICAL result. Retry wasted.
            #
            # The Judge was RIGHT — Python_(genus) is the correct entity. It just
            # wasn't in the candidate pool, because Node 2 searched for "python".
            # That is a RETRIEVAL failure, not a SCORING failure, so the fix is to
            # go back to Node 2 with the Judge's entity as the query — not back to
            # Node 3 to re-shuffle candidates that don't contain the answer.
            if t_uri and current_retry < 2:
                target_label = re.sub(r'\s*\([^)]*\)\s*$', '',
                                      t_uri.split('/')[-1].replace('_', ' ')).strip()
                paren = re.search(r'\(([^)]*)\)\s*$', t_uri.split('/')[-1].replace('_', ' '))
                # Keep the disambiguation hint in the query: "Python (genus)" -> "Python genus"
                new_query = f"{target_label} {paren.group(1)}".strip() if paren else target_label

                # ENTITY LOCK still applies — the Judge cannot escalate to a swap.
                # FIX B: anchor to the ORIGINAL sentence mentions.
                _o_sub, _o_obj = _origin_mentions(state)
                sub_ok, sub_s = entity_matches_mention(t_uri, _o_sub)
                obj_ok, obj_s = entity_matches_mention(t_uri, _o_obj)
                if not (sub_ok or obj_ok):
                    print(f"   🛑 Escalation BLOCKED by entity lock: <{t_uri.split('/')[-1]}> "
                          f"is not the entity the sentence named (best {max(sub_s, obj_s):.2f}).")
                    v_status = "APPROVED"
                else:
                    print(f"   ⚠️ Judge named <{t_uri.split('/')[-1]}>, which Node 2 never fetched.")
                    print("      This is a RETRIEVAL miss, not a scoring miss — re-weighting cannot")
                    print("      reach a candidate that isn't in the pool. Escalating to RE_SEARCH")
                    print(f"      with query '{new_query}' as the new {target_role}.")
                    v_status = "RE_SEARCH"
                    if sub_ok:
                        verdict['new_subject_query'] = new_query
                        verdict['new_object_query'] = state['raw_object']
                    else:
                        verdict['new_object_query'] = new_query
                        verdict['new_subject_query'] = state['raw_subject']
            else:
                print("   ⚠️ Judge named URIs not found in scored candidates, and no retries left. "
                      "Keeping current weights.")

    if v_status == "ADJUST_PREDICATE":
        print(f"🚨 Predicate Hijacking Detected! Forcing URI: <{suggested_pred}>")

    if v_status == "RE_SEARCH":
        # ── FIX 1: RE_SEARCH QUERY-REPEAT GUARD ───────────────────────────────
        # RE_SEARCH is only useful if the query CHANGES. Node 2 is deterministic
        # and Redis-cached, so re-issuing the same query returns byte-identical
        # candidates and the loop is guaranteed to reach the same conclusion.
        #
        # Real trace (v11, "Mars ate lunch with Donald Trump"):
        #     RE_SEARCH → subject 'Mars',  object 'Donald Trump'
        #     RE_SEARCH → subject 'Mars',  object 'Donald Trump'   (identical)
        #     RE_SEARCH → subject 'Mars',  object 'Donald Trump'   (identical)
        #   Three identical fetches, three ⚡ cache hits, two retries burned,
        #   same answer every time.
        new_sub = verdict.get('new_subject_query') or state['raw_subject']
        new_obj = verdict.get('new_object_query') or state['raw_object']

        # ── BUG C FIX: RE_SEARCH MUST STAY INSIDE THIS TRIPLE ─────────────────
        # v13 runs Nodes 2-4 once PER TRIPLE, but the Judge still sees the whole
        # sentence — so it drifts to a DIFFERENT fact's entities. Observed on
        # "Abilene Regional Airport serves the city of Abilene in Jones County,
        # Texas, United States", processing triple 3 (Abilene,_Texas | country |
        # United States):
        #     RE_SEARCH -> new_subject_query = 'Abilene Regional Airport'
        # That is TRIPLE 1's subject. The retry then resolved the subject to
        # <Wylie_High_School_(Abilene,_Texas)> — a different fact entirely.
        #
        # The entity lock already guards ADJUST_MATH and OVERRIDE this way;
        # RE_SEARCH was the one door left open. Refinement must be a BETTER QUERY
        # FOR THE SAME ENTITY ("Abilene" -> "Abilene, Texas"), never a jump to
        # another entity in the sentence.
        if ENFORCE_ENTITY_LOCK:
            _o_sub, _o_obj = _origin_mentions(state)

            def _is_refinement(new_q, origin):
                """
                Refinement ADDS CONTEXT to the same mention. Drift REPLACES it.
                    'Abilene'         -> 'Abilene, Texas'                REFINEMENT (contains)
                    'Socialist Party' -> 'Socialist Party Netherlands'   REFINEMENT (contains)
                    'United States'   -> 'Abilene'                       DRIFT (unrelated)
                    'Abilene, Texas'  -> 'Abilene Regional Airport'      DRIFT (unrelated)

                Containment is SAFE here in a way it is NOT for the entity lock:
                the lock compares a URI label to a mention, where 'Mars' vs
                'Bruno Mars' is a genuine trap. Here both sides are query strings
                drawn from the SAME sentence, and we only ask "is the Judge adding
                words to this mention, or naming a different one?".
                """
                a, b = (new_q or "").lower().strip(), (origin or "").lower().strip()
                if not a or not b:
                    return True
                if a == b or b in a or a in b:
                    return True
                return entity_matches_mention(f"x/{a.replace(' ', '_')}", b)[0]

            if not _is_refinement(new_sub, _o_sub) and new_sub != state['raw_subject']:
                print(f"   🛑 RE_SEARCH DRIFT: subject query '{new_sub}' names a different entity "
                      f"than this triple's '{_o_sub}'. Reverting.")
                new_sub = _o_sub
                verdict['new_subject_query'] = _o_sub
            if not _is_refinement(new_obj, _o_obj) and new_obj != state['raw_object']:
                print(f"   🛑 RE_SEARCH DRIFT: object query '{new_obj}' names a different entity "
                      f"than this triple's '{_o_obj}'. Reverting.")
                new_obj = _o_obj
                verdict['new_object_query'] = _o_obj

        if new_sub == state['raw_subject'] and new_obj == state['raw_object']:
            print("   ⚠️ RE_SEARCH produced IDENTICAL queries "
                  f"('{new_sub}' / '{new_obj}'). Node 2 is deterministic and cached,")
            print("      so looping would return the same candidates. Not looping.")

            if gate_rejected:
                # ── THE v12.0 REGRESSION, FIXED ──────────────────────────────
                # v12.0 approved here, which ACCEPTED a triple the SPARQL gate had
                # just refused:
                #     🛑 SPARQL GATE BLOCKED [Tier 2] ... REJECTED.
                #     ⚠️ RE_SEARCH produced IDENTICAL queries. Not looping.
                #     🏁 Validation Layer Passed!
                #     FINAL: <Mars> <rival> <Donald_Trump>      ← WRONG, v11 refused this
                # "Nothing left to retry" must mean the REJECTION STANDS, never
                # "ship it anyway". The gate's verdict is final; the guard only
                # decides whether looping is worthwhile.
                print("      The SPARQL gate already REJECTED this triple and no new query is")
                print("      available to recover with. The rejection STANDS.")
                v_status = "REJECTED_BY_SPARQL"
                final_triple_str = ("Extraction Failed: SPARQL Gate blocked an unverified triple "
                                    "and query refinement had nothing new to try.")
            else:
                # Judge-initiated RE_SEARCH that can't improve the query. The math
                # result was never gate-rejected, so let it proceed to the gate.
                v_status = "OVERRIDE" if is_final_attempt else "APPROVED"
        else:
            print("🔄 Query Refinement Triggered!")
            if verdict.get('new_subject_query'):
                print(f"   ► New Subject Search Term: '{verdict.get('new_subject_query')}'")
            if verdict.get('new_object_query'):
                print(f"   ► New Object Search Term: '{verdict.get('new_object_query')}'")

    return_payload = {
        "validation_status": v_status,
        "feedback_instruction": v_feedback,
        "final_triple": final_triple_str,
        "retry_count": current_retry + 1,
        "math_weights": new_weights,
        "suggested_predicate": suggested_pred
    }

    if v_status == "RE_SEARCH":
        if verdict.get('new_subject_query'):
            return_payload["raw_subject"] = verdict.get('new_subject_query')
        if verdict.get('new_object_query'):
            return_payload["raw_object"] = verdict.get('new_object_query')

    return return_payload


# ================================================================================
# 6. ROUTER AND GRAPH COMPILATION
# ================================================================================
def autonomous_router(state: GraphState):
    status = state["validation_status"]
    if status == "APPROVED":
        print("\n🏁 Validation Layer Passed! Breaking graph execution loop.")
        return END
    elif status == "OVERRIDE":
        print("\n🚨 PARAMETRIC OVERRIDE ENGAGED! Breaking loop.")
        return END
    elif status == "REJECTED_BY_SPARQL":
        print("\n🛑 PIPELINE HALTED: SPARQL Gate blocked a hallucination. Breaking loop.")
        return END
    elif status == "REJECTED_UNRESOLVED":
        print("\n🛑 PIPELINE HALTED: entities named in the sentence are not in DBpedia. Breaking loop.")
        return END
    elif status == "RE_SEARCH" and state["retry_count"] <= MAX_RESEARCH_PER_TRIPLE:
        # RE_SEARCH is the expensive path (full re-fetch + re-score + re-judge) and
        # on the 79-run it fired ~3.6x/sentence, driving the slowdown and the hang.
        # Capping it lower than the ADJUST paths halves the retry storm; most
        # RE_SEARCH that would fire a 2nd/3rd time were drifting anyway.
        print(f"\n🔄 Loop Triggered ({status}): Routing back to [Node 2] "
              f"(Retry {state['retry_count']}/{MAX_RESEARCH_PER_TRIPLE}).")
        return "node_2_fetcher"
    elif status in ["ADJUST_MATH", "ADJUST_PREDICATE"] and state["retry_count"] <= 2:
        print(f"\n🔄 Loop Triggered ({status}): Routing back to [Node 3] (Retry {state['retry_count']}/2).")
        return "node_3_math_engine"
    return END


pipeline_graph = StateGraph(GraphState)
for _name, _fn in [
    ("node_0_preprocessor", node_0_preprocessor),
    ("node_1_extractor", node_1_extractor),
    ("node_2_fetcher", node_2_fetcher),
    ("node_3_math_engine", node_3_math_engine),
    ("node_4_judge", node_4_judge),
]:
    pipeline_graph.add_node(_name, _fn)

pipeline_graph.set_entry_point("node_0_preprocessor")
pipeline_graph.add_edge("node_0_preprocessor", "node_1_extractor")
pipeline_graph.add_edge("node_1_extractor", "node_2_fetcher")
pipeline_graph.add_edge("node_2_fetcher", "node_3_math_engine")
pipeline_graph.add_edge("node_3_math_engine", "node_4_judge")
pipeline_graph.add_conditional_edges("node_4_judge", autonomous_router)

compiled_pipeline = pipeline_graph.compile()


# ================================================================================
# 7b. OPTION-A MULTI-TRIPLE WRAPPER  (v13)
# ================================================================================
# Node 1 fans out to N triples; this loops the EXISTING Node2→3→4 graph once per
# triple. Nodes 2/3/4, the router, the retry budget, the entity lock and the gate
# are all UNCHANGED — they already operate on one (s,p,o), they just run N times.
#
# Why Option A and not a LangGraph fan-out: retry_count, math_weights and
# suggested_predicate are currently GLOBAL to one triple. An in-graph loop would
# bleed one triple's retry state into the next unless every field is namespaced
# per index. A is a wrapper; B is a refactor. Get the number first.
#
# Each triple gets its OWN fresh retry budget, which is the behaviour we want:
# triple 2 failing should not consume triple 3's retries.

_triple_graph = StateGraph(GraphState)
_triple_graph.add_node("node_2_fetcher", node_2_fetcher)
_triple_graph.add_node("node_3_math_engine", node_3_math_engine)
_triple_graph.add_node("node_4_judge", node_4_judge)
_triple_graph.set_entry_point("node_2_fetcher")
_triple_graph.add_edge("node_2_fetcher", "node_3_math_engine")
_triple_graph.add_edge("node_3_math_engine", "node_4_judge")
_triple_graph.add_conditional_edges("node_4_judge", autonomous_router)
compiled_triple_graph = _triple_graph.compile()


def _normalize_literal(raw, pid=None, pid_range=None, quote_hint=None, value_format=None):
    """
    Emit a literal in Text2KGBench's gold convention. Rules MEASURED from gold
    (3_airport + 18_scientist):

        range=number   -> UNQUOTED, commas/units stripped, float form preserved
                          "106,000" -> 106000 ; "3287590000000.0" stays as-is
                          runwayLength is float 30/30 in gold -> always gets .0
        range=Date     -> UNQUOTED  "1776-02-18"
        range=Year     -> UNQUOTED, LEADING ZEROS PRESERVED  "0927", "1707"
        range=string   -> QUOTED if it contains a comma or space
                            "DL1, DL2, DL3" / "In God we trust"
                          UNQUOTED if it is a bare code/word
                            01325 (areaCode), NZ289147 (gridReference), President

    CRITICAL: never int()/float() a code — that would eat leading zeros
    (areaCode 01325 -> 1325, foundingYear 0927 -> 927). We work on strings only.

    pid_range: the Text2KGBench declared range for this predicate, when known.
               Falls back to shape-based inference if not supplied.
    """
    import re as _re
    v = str(raw).strip().strip('"').strip()
    pid_l = (pid or "").lower()
    rng = (pid_range or "").lower()

    # ── BUG C FIX: English dates -> ISO ───────────────────────────────────────
    # Observed: sentence "January 1, 1726" was emitted verbatim; gold is
    # 1726-01-01. Convert any recognisable English date to ISO before anything
    # else, for date-ranged predicates (and *Date pids when range is missing).
    #
    # DATE-STYLE REFINEMENT (2026-08-04, measured on all 19 TRAIN splits):
    #   * month-name values ("December 2008", "30 March 2007") are quoted in
    #     gold 41/3 — the 3 exceptions all carry a "(...)" suffix (epoch).
    #     -> any month-name value without parens is emitted QUOTED, overriding
    #        the per-predicate hint (which majority-votes across mixed shapes
    #        and gets the minority shape wrong: completionDate has bare years
    #        AND "April 2014" under one pid).
    #   * whether English dates are KEPT ("30 March 2007", building, 18/0) or
    #     ISO-CONVERTED ("1987-02-27", NRHP/scientist) is a per-predicate
    #     convention learned from train (value_format["date_style"]). Style
    #     "english" skips the ISO conversion; style "iso" also lets non-*Date
    #     pids (addedToTheNationalRegisterOfHistoricPlaces) reach it.
    _vf_style = (value_format or {}).get("date_style")
    _month_re = _re.compile(r'\b(january|february|march|april|may|june|july|'
                            r'august|september|october|november|december)\b', _re.I)
    _eng_shape = bool(_month_re.search(v)) and '(' not in v
    if (rng == "date" or pid_l.endswith("date")
            or pid_l in ("established", "foundingdate", "completiondate")
            or (_vf_style == "iso" and _eng_shape)):
        if _vf_style == "english" and _eng_shape:
            return v, True                       # gold keeps the English form, quoted
        # fall through to the ISO conversion below
        # Strip ordinal suffixes so "July 11th, 1907" and "11th July 1907" parse.
        v = _re.sub(r'(\d{1,2})(st|nd|rd|th)\b', r'\1', v)
        # Numeric DD-MM-YYYY or DD/MM/YYYY -> ISO (WebNLG uses European day-first).
        _dmy = _re.match(r'^\s*(\d{1,2})[-/](\d{1,2})[-/](\d{4})\s*$', v)
        if _dmy:
            d_, m_, y_ = int(_dmy.group(1)), int(_dmy.group(2)), _dmy.group(3)
            # if the first field is >12 it must be the day; if second >12 it's the
            # month-second (US) form; otherwise assume day-first (WebNLG convention).
            if m_ > 12 and d_ <= 12:
                d_, m_ = m_, d_          # was actually MM-DD; swap
            if 1 <= m_ <= 12 and 1 <= d_ <= 31:
                return f"{y_}-{m_:02d}-{d_:02d}", (quote_hint is True)
        _MONTHS = {m: i + 1 for i, m in enumerate(
            ["january","february","march","april","may","june","july",
             "august","september","october","november","december"])}
        _m = _re.match(r'^([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})$', v)       # January 1, 1726
        if not _m:
            _m2 = _re.match(r'^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$', v)    # 1 January 1726
            if _m2:
                _mon = _MONTHS.get(_m2.group(2).lower())
                if _mon:
                    v = f"{_m2.group(3)}-{_mon:02d}-{int(_m2.group(1)):02d}"
        else:
            _mon = _MONTHS.get(_m.group(1).lower())
            if _mon:
                v = f"{_m.group(3)}-{_mon:02d}-{int(_m.group(2)):02d}"
        # Month name survived the conversion (month-year form like "December
        # 2008", which has no day to convert): gold quotes these 41/3 across
        # all 19 train splits — emit verbatim, QUOTED, overriding the hint.
        if _month_re.search(v) and '(' not in v:
            return v, True

    # ── BUG A FIX: predicates gold NEVER quotes ───────────────────────────────
    # Measured across all 19 domains' gold: these are 100% bare.
    #   areaCode 6/6, demonym 38/38, foundingYear 2/2, populationTotal 5/5,
    #   areaTotal 21/21. Emitting them quoted scored 0 every time.
    _NEVER_QUOTED = {"areacode", "demonym", "foundingyear", "populationtotal",
                     "areatotal", "elevation", "runwaylength"}
    if pid_l in _NEVER_QUOTED:
        # strip inline unit words (feet/metres/…) but NEVER a parenthetical unit,
        # which is part of the gold literal for some domains.
        if not _re.search(r'\([a-zA-Z]', v):
            v = _re.sub(r'\b(feet|foot|ft|metres|meters|m|kilometres|km|inhabitants|people)\b',
                        '', v, flags=_re.I).strip()
        v_bare = v.replace(',', '') if _re.fullmatch(r'[\d,]+(\.\d+)?', v) else v
        if pid_l == "runwaylength" and _re.fullmatch(r'-?\d+', v_bare):
            v_bare = f"{v_bare}.0"
        # Years are ALWAYS 4 digits in gold: 927 -> 0927 (Kingdom of England).
        if pid_l.endswith("year") and _re.fullmatch(r'\d{1,3}', v_bare):
            v_bare = v_bare.zfill(4)
        return v_bare, False

    # ── LEARNED QUOTING CONVENTION (from the TRAIN split, never the test set) ──
    # Quoting is per-(domain,predicate), not global: 9_astronaut writes dates as
    # "1923-11-18" (quoted) while 18_scientist writes 1776-02-18 (unquoted).
    # Measured: the convention is 93% consistent within a (domain,predicate) pair,
    # and the TRAIN split predicts the TEST convention with 96% accuracy. So the
    # runner learns it from train and passes it here as quote_hint.
    #   quote_hint=True  -> emit quoted
    #   quote_hint=False -> emit unquoted (underscore-joined if multi-word)
    #   quote_hint=None  -> fall back to the range-based rules below

    # Strip unit words + thousands separators for NUMERIC ranges only. Codes and
    # strings must keep their exact characters.
    def _apply_hint(val):
        """Honour the learned convention when we have one."""
        if quote_hint is True:
            return val, True
        return val.replace(' ', '_'), False

    if rng == "number" or (not rng and _re.fullmatch(r'[-\d,. ]+(feet|foot|ft|metres|meters|m|km)?', v, _re.I)):
        # PRESERVE parenthetical units verbatim. Some domains (celestialbody,
        # meanoftransportation) write gold literals as "6603633000.0 (kilometres)"
        # and "178.0 (centimetres)" — the unit in parens is PART of the literal.
        # Stripping it (as we do for inline units like "2194 feet" on airport)
        # would break those. If the value already carries a "(unit)", leave it
        # exactly as-is rather than mangling it. Full unit canonicalisation
        # ("km per second" -> "(kilometrePerSeconds)") is a known limitation we
        # do not attempt — it needs a benchmark-specific unit map.
        if _re.search(r'\([a-zA-Z]', v):
            return v, (quote_hint is True)
        _vf = value_format or {}
        if _vf.get("unit"):
            # A canonical unit is learned from TRAIN for this pid (e.g.
            # periapsis -> "kilometres", averageSpeed -> "kilometrePerSeconds").
            # The sentence's own unit words ("km/s") never match gold's token,
            # so strip any inline unit tail wholesale — the canonical name
            # replaces it below.
            v_clean = _re.sub(r'(?<=[\d.])\s*[A-Za-z][A-Za-z/ .]*$', '', v).strip()
        else:
            v_clean = _re.sub(r'\b(feet|foot|ft|metres|meters|m|kilometres|km|inhabitants|people)\b',
                              '', v, flags=_re.I).strip()
        v_clean = v_clean.replace(',', '').strip()
        val = None
        if _re.fullmatch(r'-?\d+', v_clean):
            val = f"{v_clean}.0" if (pid_l == "runwaylength" or _vf.get("float0")) else v_clean
        elif _re.fullmatch(r'-?\d+\.\d+', v_clean):
            val = v_clean
        if val is not None:
            if _vf.get("unit"):
                val = f"{val} ({_vf['unit']})"
            return (val, True) if quote_hint is True else (val, False)
        # numeric range but non-numeric text -> fall through to string handling
        v = v_clean or v

    if rng in ("date",):
        return (v, True) if quote_hint is True else (v, False)

    if rng in ("year",):
        return (v, True) if quote_hint is True else (v, False)   # zeros PRESERVED

    if rng in ("string",):
        if quote_hint is not None:
            return _apply_hint(v)
        # MEASURED exception: a few string-ranged pids are treated as ENTITY
        # surface forms in gold (underscore-joined, unquoted) rather than quoted
        # strings. leaderTitle is 49 unquoted vs 5 quoted, and the unquoted form
        # uses underscores: President_of_the_United_States. Follow the majority.
        if pid_l in ("leadertitle",):
            return v.replace(' ', '_'), False
        # otherwise: quoted iff it has internal punctuation/whitespace
        if ',' in v or ' ' in v:
            return v, True                         # "DL1, DL2, DL3"
        return v, False                            # 01325, NZ289147

    # ---- no declared range: infer from shape ----
    if _re.fullmatch(r'-?\d+\.\d+', v):
        return v, False
    if _re.fullmatch(r'\d{4}-\d{2}-\d{2}', v):
        # ISO date with no declared range: the learned hint decides quoting
        # (NRHP gold writes "1987-02-27" QUOTED; hint=None keeps old unquoted).
        return v, (quote_hint is True)
    if _re.fullmatch(r'0\d+', v):
        return v, False                            # leading-zero code: keep exact
    if _re.fullmatch(r'-?\d+', v):
        return v, False
    if ',' in v or ' ' in v:
        return v, True
    return v, True


def _restore_middle_initial(resolved_local, mention):
    """
    Text2KGBench gold builds entity URIs from the SENTENCE surface form, so
    "Abraham A. Ribicoff" -> dbr:Abraham_A._Ribicoff. But English DBpedia Lookup
    only returns the redirect dbr:Abraham_Ribicoff (the canonical A._Ribicoff URI
    exists only in the German/French DBpedia). The initial gets dropped and every
    triple about that person fails the exact-match scorer.

    SAFE, NARROW rule: if the resolved local name is exactly the mention with a
    single middle initial removed, restore the mention's form. We do NOT touch any
    resolution that changed or expanded the entity — "the US" -> United_States must
    stay semantic, never become "the_US". This only ever RE-INSERTS a dropped
    initial that is present in the mention.

    Returns the corrected local name, or the original if the rule doesn't apply.
    """
    import re as _re
    m = mention.strip()
    # mention must look like  First [M.] Last  with a middle initial
    mm = _re.match(r'^([A-Z][a-zA-Z\-]+)\s+([A-Z])\.?\s+([A-Z][a-zA-Z\-]+)$', m)
    if not mm:
        return resolved_local
    first, initial, last = mm.group(1), mm.group(2), mm.group(3)
    canonical = f"{first}_{initial}._{last}"          # Abraham_A._Ribicoff
    dropped   = f"{first}_{last}"                       # Abraham_Ribicoff
    # only fire when the resolver produced exactly the initial-dropped form
    if resolved_local == dropped:
        return canonical
    return resolved_local


def _local_name(uri_or_text):
    """<http://dbpedia.org/resource/Abilene,_Texas> -> Abilene,_Texas"""
    if not uri_or_text:
        return ""
    return str(uri_or_text).strip().lstrip('<').rstrip('>').split('/')[-1]


_MONTH_RE = r'(?i)(january|february|march|april|may|june|july|august|september|october|november|december)'


def _route_unresolved_value(raw, pid, quote_hints, value_formats):
    """VALUE-SHAPE EXEMPTION from the "unresolved -> always quote" rule.

    Measured on 8_celestialbody: 40/169 value-typed gold triples have
    predicates whose declared range is NOT literal (periapsis -> "Periapsis",
    discovered -> "Person"!), so their objects took the entity path, missed
    the index, and were emitted ALWAYS-QUOTED — but gold writes them unquoted
    ("8788850000.0", "2006-12-31"). When an unresolved object LOOKS like a
    date or a number(+unit), route it through _normalize_literal with the
    learned quoting/unit conventions instead. Entity-shaped strings return
    None and keep the quoted-literal convention (measured correct for
    "Faversham, Kent, England"-style objects).
    """
    v = str(raw).strip().strip('"').strip()
    # v14: bare year under a *Year predicate -> gold's date form (19_film #39:
    # birthYear "1955" vs gold "1955-01-01"); currency-prefixed number -> bare
    # numeric ("£282,838"/"$30955" vs gold 282838.0/30955.0), then the normal
    # number branch formats it.
    if re.fullmatch(r'(?:1[6-9]|20)\d{2}', v) and str(pid).endswith("Year"):
        v = f"{v}-01-01"
    _cur = re.fullmatch(r'[\$£€]\s*([\d,]+(?:\.\d+)?)', v)
    if _cur:
        v = _cur.group(1)
    qh = (quote_hints or {}).get(pid)
    vf = (value_formats or {}).get(pid)
    dateish = bool(
        re.fullmatch(r'\d{4}-\d{1,2}-\d{1,2}', v)
        or re.fullmatch(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', v)
        or (re.fullmatch(r'(?i)[a-z]+ \d{1,2}(st|nd|rd|th)?,? \d{4}', v) and re.match(_MONTH_RE, v))
        or (re.fullmatch(r'(?i)\d{1,2}(st|nd|rd|th)? [a-z]+,? \d{4}', v) and re.search(_MONTH_RE, v))
    )
    if dateish:
        val, quoted = _normalize_literal(v, pid, 'date', qh)
        return f'"{val}"' if quoted else val
    if re.match(r'-?[\d,]', v) and re.fullmatch(
            r'-?[\d,]+(\.\d+)?(\s*\(?[A-Za-z][A-Za-z/ .]{0,40}\)?)?', v):
        val, quoted = _normalize_literal(v, pid, 'number', qh, vf)
        return f'"{val}"' if quoted else val
    return None


def _canonicalize_comma_compound(name, raw_mention):
    """Swap a comma-compound redirect alias for its canonical target — but ONLY
    when the alias is exactly the underscore-join of the raw mention, i.e. we
    landed on it because the mention itself was messy ("Gujarat, India"), not
    because resolution chose it among alternatives. Gated by
    CANONICALIZE_COMMA_COMPOUNDS; needs rd:* keys in Redis."""
    if not CANONICALIZE_COMMA_COMPOUNDS or "," not in name or not REDIS_AVAILABLE:
        return name
    mention_join = re.sub(r"\s+", " ", str(raw_mention)).strip().replace(" ", "_")
    if mention_join.lower() != name.lower():
        return name
    try:
        target = redis_client.get(f"rd:{name}")
    except Exception:
        return name
    if target and target != name:
        print(f"   🔀 Comma-compound canonicalized: {name} -> {target}")
        return target
    return name


def extract_all_triples(sentence, allowed_predicates=None, literal_predicates=None,
                        predicate_ranges=None, quote_hints=None, predicate_aliases=None,
                        value_formats=None, fewshot_examples=None, verbose=True):
    """
    Full multi-triple extraction for ONE sentence.

    Returns:
      {
        "sentence": str,
        "processed": str,
        "extracted": [ {subject,predicate,object}, ... ],   # Node 1's raw text triples
        "results": [ {                                      # one per extracted triple
            "raw": {...},
            "status": "APPROVED" | "OVERRIDE" | "REJECTED_BY_SPARQL" | ...,
            "final_triple": "<s> <p> <o>" or an Extraction Failed message,
            "sub": "Abilene_Regional_Airport",   # Text2KGBench surface form
            "rel": "cityServed",
            "obj": "Abilene,_Texas"  or  '"1913-01-01"',
            "kept": bool
          } ],
        "triples": [ {"sub","rel","obj"}, ... ]             # the KEPT ones, benchmark format
      }
    """
    if verbose:
        print("\n" + "=" * 78)
        print(f"SENTENCE: {sentence}")
        print("=" * 78)

    base = {
        "original_sentence": sentence, "sentence": sentence,
        "retry_count": 0, "math_weights": None, "suggested_predicate": None,
        "best_triple_so_far": None, "best_score_so_far": None,
        "origin_subject": None, "origin_object": None,
        "extracted_triples": None, "allowed_predicates": allowed_predicates,
        "literal_predicates": literal_predicates or set(),
        "predicate_ranges": predicate_ranges or {},
        "quote_hints": quote_hints or {},
        "fewshot_examples": fewshot_examples
    }

    # ---- Node 0 + Node 1 run ONCE for the whole sentence ----
    st = dict(base)
    st.update(node_0_preprocessor(st))
    st.update(node_1_extractor(st))

    extracted = st.get("extracted_triples") or []
    if not extracted:
        return {"sentence": sentence, "processed": st.get("sentence", sentence),
                "extracted": [], "results": [], "triples": []}

    # ---- Nodes 2→3→4 run ONCE PER TRIPLE ----
    results = []
    for i, t in enumerate(extracted, 1):
        if verbose:
            print("\n" + "-" * 78)
            print(f"[TRIPLE {i}/{len(extracted)}]  {t['subject']} | {t['predicate']} | {t['object']}")
            print("-" * 78)

        tri_state = dict(base)
        tri_state.update({
            "sentence": st["sentence"],
            "raw_subject": t["subject"],
            "raw_predicate": t["predicate"],
            "raw_object": t["object"],
            # Entity lock anchors to THIS triple's mentions. Fresh per triple —
            # RE_SEARCH may rewrite raw_*, these stay put.
            "origin_subject": t["subject"],
            "origin_object": t["object"],
            "extracted_triples": extracted,
            "literal_predicates": literal_predicates or set(),
            "predicate_ranges": predicate_ranges or {},
            "quote_hints": quote_hints or {},
        })

        try:
            out = compiled_triple_graph.invoke(tri_state)
        except Exception as e:
            print(f"   🔴 Triple {i} crashed: {str(e)[:120]}")
            results.append({"raw": t, "status": "ERROR", "final_triple": "",
                            "sub": "", "rel": "", "obj": "", "kept": False})
            continue

        status = out.get("validation_status", "")
        final = out.get("final_triple", "") or ""
        uris = re.findall(r'<(.*?)>', final)

        # A literal-object triple looks like:  <sub> <pred> "2194"
        # -> 2 URIs plus a trailing quoted value, NOT 3 URIs.
        # A literal-object triple can arrive in EITHER shape:
        #     <sub> <pred> "01325"        (bare quoted literal)
        #     <sub> <pred> <"01325">      (quoted literal wrapped in angle brackets)
        # Node 3 stores the literal's uri field as '"01325"' and the triple builder
        # then wraps it in <>, producing the second form. The old regex only
        # matched the first, so every datatype triple fell through to the entity
        # branch and the literal normalizer NEVER RAN — which is why the
        # areaCode/postalCode/date fixes appeared loaded but had no effect.
        literal_match = (re.match(r'^<([^>]*)>\s+<([^>]*)>\s+"(.*)"\s*$', final)
                         or re.match(r'^<([^>]*)>\s+<([^>]*)>\s+<"(.*)">\s*$', final))

        kept = False
        sub = rel = obj = ""
        if status in ("APPROVED", "OVERRIDE"):
            if literal_match:
                kept = True
                sub = _local_name(literal_match.group(1))
                rel = _local_name(literal_match.group(2))
                _rng = (predicate_ranges or {}).get(rel, "")
                _qh = (quote_hints or {}).get(rel)
                _lv = literal_match.group(3).strip()
                # v14: same year/currency pre-normalisation as
                # _route_unresolved_value — datatype-path literals need it too.
                if re.fullmatch(r'(?:1[6-9]|20)\d{2}', _lv) and rel.endswith("Year"):
                    _lv = f"{_lv}-01-01"
                _cur = re.fullmatch(r'[\$£€]\s*([\d,]+(?:\.\d+)?)', _lv)
                if _cur:
                    _lv = _cur.group(1)
                lit_val, is_quoted = _normalize_literal(_lv, rel, _rng, _qh,
                                                        (value_formats or {}).get(rel))
                obj = f'"{lit_val}"' if is_quoted else lit_val
            elif len(uris) == 3:
                kept = True
                sub = _local_name(uris[0])
                rel = _local_name(uris[1])
                # Predicate alias remap: the LLM often emits a more-specific
                # predicate (currentClub, formerTeam) than the benchmark gold,
                # which collapses them to a generic one (club). Remap learned
                # from train, so we only collapse distinctions gold does NOT make.
                if predicate_aliases:
                    rel = predicate_aliases.get(rel.lower(), rel)
                # Restore a dropped middle initial when the resolver returned the
                # redirect form (Abraham_Ribicoff) but the mention — and gold —
                # carry the initial (Abraham A. Ribicoff -> Abraham_A._Ribicoff).
                sub = _restore_middle_initial(sub, str(t.get("subject", "")))
                sub = _canonicalize_comma_compound(sub, t.get("subject", ""))
                # LITERAL DECISION for entity-ranged predicates: did the object
                # actually resolve, or did Node 2 return nothing (fallback minted
                # a URI, flagged unresolved=True)? Unresolved -> quoted literal.
                obj_unresolved = False
                for cand in (out.get("scored_objects") or []):
                    if cand.get("uri") == uris[2]:
                        obj_unresolved = bool(cand.get("unresolved"))
                        break
                if obj_unresolved:
                    routed = _route_unresolved_value(t["object"], rel, quote_hints, value_formats)
                    obj = routed if routed is not None else f'"{t["object"].strip()}"'
                else:
                    obj = _restore_middle_initial(_local_name(uris[2]),
                                                  str(t.get("object", "")))
                    obj = _canonicalize_comma_compound(obj, t.get("object", ""))

        results.append({"raw": t, "status": status, "final_triple": final,
                        "sub": sub, "rel": rel, "obj": obj, "kept": kept})

    # ── BUG B FIX: DEDUPE ─────────────────────────────────────────────────────
    # Observed on "Trane, whose products include HVAC, was founded in La Crosse,
    # Wisconsin on 1913-01-01":
    #     (Trane, foundationPlace, La_Crosse,_Wisconsin)
    #     (Trane, foundationPlace, La_Crosse,_Wisconsin)   <- duplicate
    # Triple 3 (foundingDate/1913-01-01) got the wrong predicate, then RE_SEARCH
    # walked its object to "La Crosse, Wisconsin" — turning it into a copy of
    # triple 2. Duplicates are counted once by the scorer's set() on the gold
    # side but inflate len(pred), so every duplicate is a direct precision hit.
    # ── BUG B FIX: re-merge split compound literals ───────────────────────────
    # Observed: "The zip code areas in Darlington are DL1, DL2, DL3." produced
    #     (Darlington, postalCode, "DL1")
    #     (Darlington, postalCode, "DL2")
    #     (Darlington, postalCode, "DL3")
    # but gold is ONE triple: (Darlington, postalCode, "DL1, DL2, DL3").
    # Node 1 split the list. When several KEPT triples share the same subject AND
    # predicate and all objects are quoted literals, merge them back into a single
    # comma-joined literal.
    _by_sr = {}
    for r in results:
        if not r["kept"]:
            continue
        obj = str(r["obj"])
        if obj.startswith('"') and obj.endswith('"'):
            _by_sr.setdefault((r["sub"], r["rel"]), []).append(r)
    for (sub, rel), group in _by_sr.items():
        if len(group) > 1:
            parts = [str(g["obj"]).strip('"') for g in group]
            merged = '"' + ", ".join(parts) + '"'
            if verbose:
                print(f"   🔗 Merged {len(group)} split literals for "
                      f"({sub}, {rel}) -> {merged}")
            group[0]["obj"] = merged
            for g in group[1:]:
                g["kept"] = False

    seen, triples = set(), []
    for r in results:
        if not r["kept"]:
            continue
        key = (r["sub"].lower(), r["rel"].lower(), r["obj"].lower())
        if key in seen:
            if verbose:
                print(f"   ⚠️ Duplicate dropped: ({r['sub']}, {r['rel']}, {r['obj']})")
            continue
        seen.add(key)
        triples.append({"sub": r["sub"], "rel": r["rel"], "obj": r["obj"]})

    if verbose:
        print("\n" + "=" * 78)
        print(f"KEPT {len(triples)}/{len(extracted)} triple(s):")
        for tr in triples:
            print(f"   ({tr['sub']}, {tr['rel']}, {tr['obj']})")
        dropped = [r for r in results if not r["kept"]]
        for r in dropped:
            print(f"   ✗ dropped [{r['status']}]: {r['raw']['subject']} | "
                  f"{r['raw']['predicate']} | {r['raw']['object']}")
        print("=" * 78)

    return {"sentence": sentence, "processed": st.get("sentence", sentence),
            "extracted": extracted, "results": results, "triples": triples}


# ================================================================================
# 8. STANDALONE MATH-ONLY RUNNER (For Isolated Ablation)
# ================================================================================
def run_pure_math_only(sentence, raw_subject, raw_predicate, raw_object):
    """
    Runs ONLY Node 2 + Node 3 — no preprocessor, no extractor, no Judge.
    Feed gold-standard (s, p, o) strings to isolate the pure symbolic math score.
    """
    state: GraphState = {
        "original_sentence": sentence, "sentence": sentence,
        "raw_subject": raw_subject, "raw_predicate": raw_predicate, "raw_object": raw_object,
        "subject_candidates": [], "object_candidates": [],
        "scored_subjects": [], "scored_objects": [], "top_5_triples": [],
        "validation_status": "", "final_triple": None, "feedback_instruction": None,
        "retry_count": 0, "math_weights": None, "suggested_predicate": None,
        "best_triple_so_far": None, "best_score_so_far": None,
        # Node 1 is skipped here, so anchor the lock to the injected gold mentions.
        "origin_subject": raw_subject, "origin_object": raw_object,
        "extracted_triples": None, "allowed_predicates": None, "literal_predicates": set(), "predicate_ranges": {}, "quote_hints": {}
    }
    state.update(node_2_fetcher(state))
    state.update(node_3_math_engine(state))
    winner = state["top_5_triples"][0]
    return f"<{winner['subject']['uri']}> <{winner['predicate']}> <{winner['object']['uri']}>"


if __name__ == "__main__":
    # Text2KGBench-style sentences: several facts, chained entities, and values.
    demo = [
        ("Abilene Regional Airport serves the city of Abilene in Jones County, Texas, United States.",
         ["cityServed", "isPartOf", "country", "location", "runwayLength"]),
        ("Trane, whose products include HVAC, was founded in La Crosse, Wisconsin on 1913-01-01.",
         ["product", "foundationPlace", "foundingDate", "location"]),
        # Faithfulness must survive the multi-triple refactor: still Hillary, not Trump.
        ("Hillary Clinton won the 2016 United States presidential election.", None),
    ]
    for sent, allowed in demo:
        out = extract_all_triples(sent, allowed_predicates=allowed)
        print("\nBENCHMARK FORMAT:")
        for t in out["triples"]:
            print(f'   {{"sub": "{t["sub"]}", "rel": "{t["rel"]}", "obj": "{t["obj"]}"}}')
