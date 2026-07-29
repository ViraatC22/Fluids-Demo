#!/usr/bin/env bash
set -euo pipefail

export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/fluidsdemo-pycache"

python3 -m compileall -q app.py tests
python3 -m unittest discover -s tests -v
