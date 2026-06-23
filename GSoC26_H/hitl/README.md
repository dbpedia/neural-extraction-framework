# Human-in-the-Loop Review Interface

Streamlit app for reviewing the ontology alignment layer's output —
accept, modify, or reject each suggested Hindi predicate → DBpedia
property mapping.

## Run locally

```bash
cd GSoC26_H/hitl
pip install -r requirements.txt
streamlit run hitl_app.py
```

## Data

Looks for `alignment_results_full_20k.jsonl` in `GSoC26_H/results/` or
`GSoC26_H/data/` automatically. Falls back to a small demo set if not
found, so the interface is always usable for a quick look.

## Deploy

Deployed via [Streamlit Community Cloud](https://share.streamlit.io) —
point it at this repo, branch `gsoc26h-development`, main file path
`GSoC26_H/hitl/hitl_app.py`. Streamlit Cloud uses the `requirements.txt`
in this same folder (not the repo root) for a fast, lightweight build.
