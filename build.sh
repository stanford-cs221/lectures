#!/bin/bash
# Rebuild the trace viewer from the local edtrace clone (edtrace/frontend).
#   ./build.sh            Write index.html + assets/ here (hosted at /lectures); commit the result for GitHub Pages.
#   ./build.sh --preview  Write to dist/ (gitignored, base path /dist) so it can be viewed from this checkout
#                         at <server>/dist/index.html?trace=gibbs_sampling without touching the committed files.
set -e
cd "$(dirname "$0")"
if [ ! -d edtrace ]; then
  git clone https://github.com/percyliang/edtrace edtrace
fi
if [ "$1" = "--preview" ]; then
  export VITE_EDTRACE_BASE_DIR=/dist
  export VITE_EDTRACE_DIST_DIR=$PWD/dist
  mkdir -p dist
  (cd dist && ln -sfn ../var var && ln -sfn ../images images)  # Share traces and images instead of copying
else
  export VITE_EDTRACE_BASE_DIR=/lectures  # Hosted at stanford-cs221.github.io/lectures (not the checkout's directory name)
  export VITE_EDTRACE_DIST_DIR=$PWD
fi
npm --prefix edtrace/frontend install
rm -rf "$VITE_EDTRACE_DIST_DIR/assets"  # Remove stale hashed assets
npm --prefix edtrace/frontend run build
