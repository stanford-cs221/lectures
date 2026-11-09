#!/bin/bash
# Rebuild the trace viewer (index.html + assets/) from the local edtrace clone (edtrace/frontend).
# Commit the result so it shows up on GitHub Pages.
# Usage: ./build.sh [base]
#   base: URL path prefix the viewer is served from (default /lectures, for GitHub Pages).
#   For previewing elsewhere, e.g.: ./build.sh /api/projects/lectures/tree/main
#   (don't commit that build).
set -e
cd "$(dirname "$0")"
if [ ! -d edtrace ]; then
  git clone https://github.com/percyliang/edtrace edtrace
fi
export VITE_EDTRACE_BASE_DIR=${1:-/lectures}  # Default: hosted at stanford-cs221.github.io/lectures (not the checkout's directory name)
export VITE_EDTRACE_DIST_DIR=$PWD
npm --prefix edtrace/frontend install
if [ -n "$1" ]; then
  # When previewing, the viewer is opened as <base>/index.html, which the router doesn't match (it only has "/");
  # temporarily add a catch-all route to the edtrace clone.
  trap 'git -C edtrace checkout frontend/src/App.jsx' EXIT
  sed -i 's|^\( *\)<Route path="/" element={<TraceViewer />} />|&\n\1<Route path="*" element={<TraceViewer />} />|' edtrace/frontend/src/App.jsx
fi
rm -rf assets  # Remove stale hashed assets
npm --prefix edtrace/frontend run build
