#!/usr/bin/env python3
"""Antigravity bridge - lets the Kaggle/Colab dubbing notebook use YOUR PC's `agy`
(Google Antigravity CLI) for translation, ahead of Groq/Cerebras in the fallback chain.

Run this on the same PC where `agy` is installed and already logged in (agy_headless.py
uses agy's cached credentials - see its docstring). It starts a local HTTP server and opens a
free Cloudflare quick tunnel to it, then prints a public URL + a token you paste into whichever
of these two you're using this session (this bridge is NOT used by the notebook's Cell 3
one-click pipeline - that one always stays Groq/Cerebras only):

  - Cell 2 (Kaggle/Colab, launches colab_server.py): paste into the
    antigravity_bridge_url / antigravity_bridge_token variables near the top of that cell.
  - web_gui.py, running on THIS SAME PC in "remote" mode against Cell 2: set
    ANTIGRAVITY_BRIDGE_URL / ANTIGRAVITY_BRIDGE_TOKEN in this PC's .env before starting it.

    python bridge_server.py

The tunnel URL changes every time you restart this script - re-paste it each session. Keep
this window open for the whole run; closing it (or your PC sleeping) just makes the bridge
tier fail over to Groq/Cerebras like normal, it won't break the pipeline.

A random shared-secret token is generated each run and printed alongside the URL. Without it,
anyone who discovers the tunnel URL while it's live could spend your agy quota.
"""

import os
import secrets
import sys
import threading
import time

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agy_headless import ask, AgyError, DEFAULT_MODEL

PORT = int(os.environ.get("BRIDGE_PORT", "8787"))
TOKEN = os.environ.get("BRIDGE_TOKEN") or secrets.token_urlsafe(24)
# The Flash-pinned default lives in agy_headless.DEFAULT_MODEL so it applies identically
# whether agy is called through this HTTP bridge or directly in-process (see
# video_dubbing_core._call_antigravity_local, used by web_gui.py's local execution mode).
MODEL = os.environ.get("AGY_MODEL") or DEFAULT_MODEL

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL})


@app.route("/translate", methods=["POST"])
def translate():
    if request.headers.get("X-Bridge-Token") != TOKEN:
        return jsonify({"error": "invalid or missing X-Bridge-Token"}), 401

    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "missing 'prompt'"}), 400

    try:
        # Capped comfortably below the Kaggle-side request timeout (90s - see
        # _call_antigravity_bridge in video_dubbing_core.py) so this always answers - success
        # or error - before the caller gives up and falls back to Groq/Cerebras on its own,
        # rather than both sides timing out independently and wasting the run.
        answer = ask(prompt, timeout=75)
        return jsonify({"answer": answer})
    except AgyError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"unexpected bridge error: {e}"}), 500


def main():
    server_thread = threading.Thread(
        # threaded=False (default) is deliberate: agy_headless.ask() looks up the active
        # conversation by cwd, which isn't concurrency-safe - two overlapping requests could
        # cross-read each other's answers. Serializing requests here is correct, not a
        # limitation - this pipeline only ever sends one translation request at a time anyway.
        target=lambda: app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()
    time.sleep(1)

    print("=" * 65)
    print("Antigravity bridge starting...")
    print(f"Local server: http://127.0.0.1:{PORT}")
    print(f"Model: {MODEL}  (override with the AGY_MODEL env var if this slug errors on your "
          f"install - check exact names with `agy models`)")

    public_url = ""
    try:
        from pycloudflared import try_cloudflare
        print("Opening Cloudflare quick tunnel...")
        tunnel_res = try_cloudflare(port=PORT)
        if hasattr(tunnel_res, "tunnel_url"):
            public_url = str(tunnel_res.tunnel_url)
        elif hasattr(tunnel_res, "url"):
            public_url = str(tunnel_res.url)
        elif str(tunnel_res).startswith("http"):
            public_url = str(tunnel_res)
        else:
            import re
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", str(tunnel_res))
            if m:
                public_url = m.group(0)
    except Exception as e:
        print(f"Could not open a Cloudflare tunnel ({e}). Falling back to local-only - this "
              f"only works if Kaggle/Colab can somehow reach {'http://127.0.0.1:' + str(PORT)}, "
              f"which it normally can't. Install pycloudflared and re-run: pip install pycloudflared")

    print("=" * 65)
    if public_url:
        print(f"Bridge URL:   {public_url}")
    print(f"Bridge token: {TOKEN}")
    print("Paste both into Cell 2's antigravity_bridge_url/token variables, and/or this PC's")
    print(".env (ANTIGRAVITY_BRIDGE_URL / ANTIGRAVITY_BRIDGE_TOKEN) if web_gui.py drives Cell 2 remotely.")
    print("Leave this window open for the duration of the dubbing run.")
    print("=" * 65)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopping bridge.")


if __name__ == "__main__":
    main()
