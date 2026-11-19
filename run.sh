#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 src/generate_data.py
python3 src/segment.py
python3 src/evaluate.py
echo "Done. See metrics.json, data/customers_segmented.csv and screenshots/"
