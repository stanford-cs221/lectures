#!/bin/bash
# Rebuild the trace viewer (index.html + assets/) from the local edtrace clone (edtrace/frontend).
# Hosted at /lectures (GitHub Pages); serve.py rewrites that to / when serving at the root.  Commit the result.
set -e
cd "$(dirname "$0")"
if [ ! -d edtrace ]; then
  git clone https://github.com/percyliang/edtrace edtrace
fi
export VITE_EDTRACE_BASE_DIR=/lectures  # Hosted at stanford-cs221.github.io/lectures (not the checkout's directory name)
export VITE_EDTRACE_DIST_DIR=$PWD
npm --prefix edtrace/frontend install
rm -rf assets  # Remove stale hashed assets
npm --prefix edtrace/frontend run build
