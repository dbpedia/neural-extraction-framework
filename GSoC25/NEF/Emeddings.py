import os, math, time, sys, json, requests, re, unicodedata
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace

# === API key (Gemini Developer API) ===
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    from getpass import getpass
    API_KEY = getpass("Enter your Gemini API key: ").strip()

# == Gemini Embeddings REST (batch) ==
BATCH_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:batchEmbedContents"
HEADERS = {"Content-Type": "application/json"}

# ==== Helpers ====
def _localname(uri: str) -> str:
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]

def _split_camel(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r" \1", name)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s

def make_label_text(uri: str, rdfs_label: str | None, rdfs_comment: str | None) -> str:
    """
    Build a neutral, automatic label string for embedding.
    - mechanical localname split (no hardcoded meanings)
    - include rdfs:label / rdfs:comment from the ontology when available
    """
    ln = _localname(uri)
    ln_clean = _split_camel(ln)
    parts = [f"predicate: {ln_clean}"]
    if rdfs_label:
        parts.append(f"label: {rdfs_label}")
    if rdfs_comment:
        parts.append(f"comment: {rdfs_comment}")
    return ". ".join(parts)

# === Load DBpedia ontology ===
g = Graph()
# dbpedia ontology (public URL)
g.parse("http://dief.tools.dbpedia.org/server/ontology/dbpedia.owl")
print(f"Total triples in graph: {len(g)}")

# --- SPARQL: collect predicates with english labels/comments ---
query = """
SELECT ?p ?label ?comment WHERE {
  ?p a rdf:Property ;
     rdfs:label ?label .
  FILTER(LANG(?label) = "en")
  OPTIONAL {
    ?p rdfs:comment ?comment .
    FILTER(LANG(?comment) = "en")
  }
}
ORDER BY ?p
"""

df = pd.DataFrame(g.query(query), columns=["predicate", "label", "comment"])
print(f"Total predicates (rows): {len(df)}")
print(df.head(5))

# Extract unique predicate rows by URI (keep first label/comment seen)
df["predicate"] = df["predicate"].astype(str)
uniq = df.drop_duplicates(subset=["predicate"], keep="first").reset_index(drop=True)

predicates = uniq["predicate"].tolist()
labels = uniq["label"].astype(str).tolist()
comments = uniq["comment"].apply(lambda x: str(x) if not pd.isna(x) else "").tolist()

print(f"Unique predicate URIs: {len(predicates)}")

# === Build texts to embed (semantic, not raw URIs) ===
texts = [make_label_text(u, l, c if c else None) for u, l, c in zip(predicates, labels, comments)]

# === Embedding config ===
# Force output dimensionality so it matches your runtime (e.g., 768)
OUTPUT_DIM = 768
BATCH_SIZE = 100
MAX_RETRIES = 5
SLEEP_BETWEEN_CALLS = 0.4  # pacing for free-tier rate limits

def make_requests_payload(text_batch):
    """
    Each item in 'requests' is one EmbedContentRequest.
    We pass outputDimensionality PER REQUEST to guarantee the shape.
    """
    requests_payload = []
    for text in text_batch:
        requests_payload.append({
            "model": "models/embedding-001",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": OUTPUT_DIM,
            # Optional: "taskType": "RETRIEVAL_DOCUMENT",
        })
    return {"requests": requests_payload}

def embed_batch_rest(text_batch):
    # Simple retry with backoff
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = make_requests_payload(text_batch)
            resp = requests.post(
                f"{BATCH_ENDPOINT}?key={API_KEY}",
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=90,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Expected: {"embeddings": [{"values":[...]} , ... ]}
                if "embeddings" not in data or not isinstance(data["embeddings"], list):
                    raise RuntimeError(f"Unexpected response format: {data}")
                return [e["values"] for e in data["embeddings"]]
            else:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"HTTP {resp.status_code}: {err}")
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"[embed_batch_rest] attempt {attempt} failed: {e}\nSleeping {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Failed to embed after retries.")

# === Run embeddings in batches ===
all_vecs = []
n = len(texts)
num_batches = math.ceil(n / BATCH_SIZE)
print(f"Embedding {n} predicates (dim={OUTPUT_DIM}) in {num_batches} batch(es), size {BATCH_SIZE}...")

for b in range(num_batches):
    s = b * BATCH_SIZE
    e = min((b + 1) * BATCH_SIZE, n)
    batch_texts = texts[s:e]
    vecs = embed_batch_rest(batch_texts)
    if len(vecs) != len(batch_texts):
        raise RuntimeError(f"Vector count mismatch in batch {b+1}: got {len(vecs)} for {len(batch_texts)} texts")
    all_vecs.extend(vecs)
    print(f"  Batch {b+1}/{num_batches}: {len(batch_texts)} items → {len(vecs)} vectors")
    time.sleep(SLEEP_BETWEEN_CALLS)

# === Finalize
embeddings = np.asarray(all_vecs, dtype=np.float32)
print("\n✅ Done!")
print(f"Total vectors: {embeddings.shape[0]}")
print(f"Vector dim: {embeddings.shape[1] if embeddings.size else 'N/A'}")
print("Example label:", texts[0])
print("First vector (first 10 dims):", embeddings[0][:10].tolist() if embeddings.size else "N/A")

# === Save outputs ===
# We keep 'predicates.csv' as the ID list (URIs),
# and create a companion 'predicate_labels.csv' for transparency/debug.
pred_out = "predicates.csv"
labels_out = "predicate_labels.csv"
vec_out = "embeddings.npy"

pd.DataFrame({"predicate": predicates}).to_csv(pred_out, index=False)
pd.DataFrame({"predicate": predicates, "label_text": texts}).to_csv(labels_out, index=False)
np.save(vec_out, embeddings)

print(f"\nSaved:")
print(f"- {pred_out} (URIs)")
print(f"- {labels_out} (what was embedded)")
print(f"- {vec_out} (shape: {embeddings.shape})")
