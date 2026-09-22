#!/bin/bash
# Build a lecture trace, e.g.: ./compile.sh welcome
# Uses the local edtrace clone in editable mode so hot changes there are picked up.
uv run --with-editable edtrace/backend python -m edtrace.execute -m "$1"
