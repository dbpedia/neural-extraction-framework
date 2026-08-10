#!/usr/bin/env bash
# One-time volume seeding for the extraction stack.
#
# Expects two data artifacts in deploy/data/ (download from the published
# snapshot links in DEPLOY.md, or copy from your build machine):
#   deploy/data/dump.rdb     — Redis surface-form index (16.2M mentions, ~1.4 GB)
#   deploy/data/oxigraph/    — oxigraph data directory (DBpedia triple load)
set -euo pipefail
cd "$(dirname "$0")"

PROJECT="neural-extraction-framework"   # must match `name:` in docker-compose.yml
VOL_INDEX="${PROJECT}_sf-index"
VOL_STORE="${PROJECT}_store"

[ -f data/dump.rdb ]   || { echo "missing data/dump.rdb — see DEPLOY.md";   exit 1; }
[ -d data/oxigraph ]   || { echo "missing data/oxigraph/ — see DEPLOY.md";  exit 1; }

docker volume create "$VOL_INDEX" >/dev/null
docker volume create "$VOL_STORE" >/dev/null

echo "Seeding Redis index volume ($VOL_INDEX)..."
docker run --rm -v "$VOL_INDEX":/dest -v "$(pwd)/data":/src alpine \
  cp /src/dump.rdb /dest/dump.rdb

echo "Seeding oxigraph store volume ($VOL_STORE)..."
docker run --rm -v "$VOL_STORE":/dest -v "$(pwd)/data/oxigraph":/src alpine \
  sh -c "cp -a /src/. /dest/"

echo "Done. Now: docker compose up"
