#!/bin/sh
set -e

if [ "$RUN_INGEST" = "true" ]; then
    python src/cli/ingest.py
fi

exec python src/cli/search.py