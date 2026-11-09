#!/bin/bash
# Rebuild the trace viewer from the local edtrace clone (edtrace/frontend).
# Usage:
#   ./build.sh            Write index.html + assets/ here, hosted at /lectures (GitHub Pages). Commit the result.
#   ./build.sh --preview [prefix]
#                         Write to preview/ (gitignored) so it doesn't clash with the committed build.
#                         Served at <prefix>/preview/index.html (default prefix: /api/projects/lectures/tree/main).
set -e
cd "$(dirname "$0")"
if [ ! -d edtrace ]; then
  git clone https://github.com/percyliang/edtrace edtrace
fi
if [ "$1" = "--preview" ]; then
  preview=1
  export VITE_EDTRACE_BASE_DIR=${2:-/api/projects/lectures/tree/main}/preview
  export VITE_EDTRACE_DIST_DIR=$PWD/preview
  mkdir -p preview
  (cd preview && ln -sfn ../var var && ln -sfn ../images images)  # Share traces and images instead of copying
else
  export VITE_EDTRACE_BASE_DIR=/lectures  # Hosted at stanford-cs221.github.io/lectures (not the checkout's directory name)
  export VITE_EDTRACE_DIST_DIR=$PWD
fi
npm --prefix edtrace/frontend install
if [ -n "$preview" ]; then
  # The preview is opened as <base>/index.html, which the router doesn't match (it only has "/");
  # temporarily add a catch-all route to the edtrace clone.
  trap 'git -C edtrace checkout frontend/src/App.jsx' EXIT
  sed -i 's|^\( *\)<Route path="/" element={<TraceViewer />} />|&\n\1<Route path="*" element={<TraceViewer />} />|' edtrace/frontend/src/App.jsx
fi
rm -rf "$VITE_EDTRACE_DIST_DIR/assets"  # Remove stale hashed assets
npm --prefix edtrace/frontend run build
