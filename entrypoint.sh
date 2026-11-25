#!/bin/sh
set -e

# Handle custom command arguments or fallback to keep-alive mode
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo ""
    echo "Container ready. (OCR engine: Surya — GPU accelerated)"
    echo "Launch the interactive numbered menu:"
    echo "  docker compose exec -it rag python -m src.cli.menu"
    echo ""
    tail -f /dev/null
fi
