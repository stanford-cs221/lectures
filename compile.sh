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
# Run offline (from uv's cache) if everything is cached, so that no internet access is needed; otherwise go online.
offline=--offline
if ! uv run --offline --with-editable edtrace/backend python -c "" 2>/dev/null; then
  echo "Not everything is cached; using network access..." >&2
  offline=
fi
uv run $offline --with-editable edtrace/backend python -m edtrace.execute -m "${modules[@]}"
