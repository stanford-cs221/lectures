"""
Serve the trace viewer at the root of a site (e.g., https://example.com/?trace=gibbs_sampling).

The committed index.html and assets/ are built for GitHub Pages (base path /lectures), so rewrite that path to / on the
fly.  Only the viewer and the data it needs (traces, images) are served, not the rest of the repository.
"""
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = os.path.dirname(os.path.realpath(__file__))
PUBLIC_FILES = {"index.html"}
PUBLIC_DIRS = ("assets/", "var/traces/", "images/")
BASE_PATH = b"/lectures"  # What build.sh uses by default


def rewrite(body: bytes) -> bytes:
    """Make the viewer think it's hosted at / (both the asset URLs and the router's base path)."""
    return body.replace(BASE_PATH + b"/", b"/").replace(b'"' + BASE_PATH + b'"', b'"/"')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        requested = unquote(urlparse(self.path).path).lstrip("/") or "index.html"
        path = os.path.realpath(os.path.join(ROOT, requested))
        relative_path = os.path.relpath(path, ROOT)  # After resolving .. and symlinks
        allowed = relative_path in PUBLIC_FILES or relative_path.startswith(PUBLIC_DIRS)
        if not allowed or not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            body = f.read()
        if relative_path == "index.html" or relative_path.endswith(".js"):
            body = rewrite(body)
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8000))
    print(f"Serving on port {port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
