"""Minimal HTTP server that logs incoming attendance events.

Run this in a separate terminal during local testing.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "0.0.0.0"
PORT = 3000
PATH = "/api/attendance/events"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # silence default per-request log
        return

    def do_POST(self):
        if self.path != PATH:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        api_key = self.headers.get("X-API-Key", "")
        print("=" * 50)
        print("RECEIVED EVENT")
        print(f"Employee: {payload.get('employee_id')}")
        print(f"Name: {payload.get('name')}")
        print(f"Time: {payload.get('timestamp')}")
        print(f"Confidence: {payload.get('confidence')}")
        print(f"Type: {payload.get('event_type')}")
        if api_key:
            print(f"X-API-Key: {api_key}")
        print("=" * 50)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')


def main() -> int:
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Fake backend listening on http://localhost:{PORT}")
    print(f"POST endpoint: {PATH}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
