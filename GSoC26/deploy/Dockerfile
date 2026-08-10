# Pipeline service image — DRAFT (deployment packaging, post-v14-final).
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    rdflib redis pydantic requests numpy \
    langchain-core langchain-openai langgraph \
    sentence-transformers rank-bm25 \
    fastapi uvicorn

COPY src/autonomous_pipeline_v13.py src/surface_index.py src/chunked_runner.py \
     src/text2kg_harness.py src/repair_tiers.py src/per_sentence_timeout.py \
     data/dbpedia.owl ./
COPY deploy/serve.py ./

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
