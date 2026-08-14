import json, subprocess, re, statistics as st

TOTAL = 4643098
STRATA = 20
PER = 250
step = TOTAL // STRATA
texts = []
for i in range(STRATA):
    off = i * step
    q = ("SELECT ?o WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#comment> ?o } "
         f"OFFSET {off} LIMIT {PER}")
    cmd = ["ssh","-o","BatchMode=yes","root@91.99.92.217",
           f"curl -s -m 300 'http://localhost:7878/query' --data-urlencode "
           f"\"query={q}\" -H 'Accept: application/sparql-results+json'"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
        texts += [b['o']['value'] for b in d['results']['bindings']]
    except Exception as e:
        print("stratum", i, "failed", e)
print("raw sampled:", len(texts))
json.dump(texts, open("abs_sample_strat.json","w"))
