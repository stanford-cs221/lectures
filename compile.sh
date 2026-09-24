#!/bin/bash
# Build one or more lecture traces, e.g.: ./compile.sh welcome search.py mdp
# Uses the local edtrace clone in editable mode so hot changes there are picked up.
if [ $# -eq 0 ]; then
  echo "Usage: $0 <lecture> [<lecture> ...]" >&2
  exit 1
fi
modules=()
for f in "$@"; do
  modules+=("${f%.py}")
done
uv run --with-editable edtrace/backend python -m edtrace.execute -m "${modules[@]}"
