#!/usr/bin/env python3
"""Reliable non-interactive use of Google Antigravity's CLI (`agy`) on Windows.

Adapted from the community fix at
https://gist.github.com/allahsan/a9a9e9c8a49aecede67ce974e64ef3cf (credit: allahsan).

THE PROBLEM (two stacked failures):

1. agy's --print mode routes chat output through a TTY-only renderer: when stdout is a pipe
   (every programmatic caller, including this bridge), it completes the model call then prints
   NOTHING, exit 0. Upstream: github.com/google-antigravity/antigravity-cli/issues/76
   (+ gemini-cli #27466).
2. Errors are ALSO silent this way - e.g. quota exhaustion (RESOURCE_EXHAUSTED 429) produces the
   same empty exit-0, indistinguishable from success.

THE FIX: don't fight stdout. agy faithfully records every conversation step (model text, tool
calls, errors) in per-conversation SQLite DBs at ~/.gemini/antigravity-cli/conversations/. Run
agy with stdout thrown away, then read the ANSWER (or the real error) from the DB instead.

Usable as a library (ask()) or standalone:
    python agy_headless.py "<prompt>"
    python agy_headless.py --file prompt.txt

Env overrides: AGY_TIMEOUT (secs, default 300) - AGY_MODEL (--model value) - AGY_RAW=1 (skip
the no-tools preamble; by default one is prepended so Q&A prompts don't wander into tool use).
"""

import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
import time

CONV_DIR = os.path.expanduser("~/.gemini/antigravity-cli/conversations")
DEFAULT_TIMEOUT = int(os.environ.get("AGY_TIMEOUT", "300"))
# Pinned to Flash's lightest reasoning tier (not agy's default/Pro model) so translation - a
# short, simple task - doesn't burn through the same daily quota as heavier coding work you
# might also be doing with agy. Verified against a real agy v1.1.13 install (`agy models` and
# a live translate call), not a guess - but slugs can still drift on future agy releases; if
# this starts erroring, run `agy models` for the current list and override with AGY_MODEL.
DEFAULT_MODEL = "gemini-3.7-flash-low"
PREAMBLE = ("You are answering a question as an external reviewer. Answer directly in plain text. "
            "Do NOT use any tools, do not read files, do not run commands, do not browse. "
            "Just write your answer.\n\nQUESTION:\n")
NOISE = re.compile(
    r'^\["\'\]?\$?\[0-9a-f\-\]{20,}|^\*2\(|^b\$|^.?\{"|^bot-|^task-|^file:///'
    r'|^-\d+H?$|^[A-Za-z0-9_\-]{20,}$|toolAction|toolSummary'
)


class AgyError(RuntimeError):
    """Raised by ask() on any failure - timeout, agy-reported error, or no answer at all."""


# Runs of valid UTF-8 byte sequences (ASCII plus proper 2/3/4-byte continuations) - NOT just
# the printable-ASCII range (\x20-\x7e). Translation output is very often non-ASCII (accented
# Latin, or fully non-Latin scripts like Hindi/Arabic/Chinese, which is exactly what this
# dubbing pipeline needs most) - an ASCII-only byte match drops/corrupts every such character
# mid-word wherever it lands in the middle of a printable run, silently truncating the answer.
_UTF8_TEXT_RUN = re.compile(
    rb"(?:[\x09\x0a\x0d\x20-\x7e]|[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}|[\xf0-\xf4][\x80-\xbf]{3})+"
)


def _model_prose(payload: bytes) -> list:
    """Pull human prose out of a protobuf step blob, dropping routing IDs and structures."""
    out = []
    for t in _UTF8_TEXT_RUN.findall(payload or b""):
        if len(t) < 12:
            continue
        for line in t.decode("utf-8", "replace").split("\n"):
            s = line.strip()
            if len(s) < 2 or NOISE.match(s):
                continue
            # Require at least one letter in ANY script (not just ASCII a-zA-Z0-9) so a line
            # that's purely Hindi/Arabic/Chinese script, with no Latin letters or digits at
            # all, still counts as prose instead of getting dropped as noise.
            if not any(ch.isalpha() for ch in s):
                continue
            if " " in s or len(s) <= 40:
                out.append(s)
    return out


def _read_conversation(db_path: str, after_idx: int = -1):
    """Return (answer_text, error_text) from a conversation DB.

    Only steps with idx > after_idx are read - print mode CONTINUES the workspace's existing
    conversation, so callers must pass the pre-run max idx to avoid extracting stale exchanges.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    prose, errors = [], []
    for stype, payload in con.execute(
            "SELECT step_type, step_payload FROM steps WHERE idx > ? ORDER BY idx", (after_idx,)):
        b = payload or b""
        if not isinstance(b, bytes):
            b = str(b).encode()
        if b"RESOURCE_EXHAUSTED" in b or b"Encountered retryable error" in b or stype == 17:
            m = re.search(rb"(RESOURCE_EXHAUSTED[^|\x00]{0,160}|Individual quota reached[^\x00|]{0,120})", b)
            if m:
                errors.append(m.group(1).decode("utf-8", "replace").strip())
            continue
        if stype == 15:
            prose.extend(_model_prose(b))
    con.close()

    seen, dedup = set(), []
    for p in prose:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return "\n".join(dedup).strip(), ("; ".join(dict.fromkeys(errors)) if errors else "")


def _max_step_idx(db_path: str) -> int:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        (n,) = con.execute("SELECT COALESCE(MAX(idx), -1) FROM steps").fetchone()
        con.close()
        return n
    except Exception:
        return -1


def ask(prompt: str, timeout: int = None, model: str = None, raw: bool = False) -> str:
    """Sends prompt to agy headlessly and returns its plain-text answer.

    Raises AgyError on timeout, an agy-reported error (e.g. quota exhausted), or if agy never
    recorded a conversation at all (auth/startup failure).
    """
    timeout = timeout or DEFAULT_TIMEOUT
    model = model or os.environ.get("AGY_MODEL") or DEFAULT_MODEL

    text = prompt if raw or os.environ.get("AGY_RAW") == "1" else PREAMBLE + prompt
    text = re.sub(r"\s*\n\s*", " ", text).strip()

    before_steps = {p: _max_step_idx(p) for p in glob.glob(os.path.join(CONV_DIR, "*.db"))}

    argv = ["agy", "--print-timeout", f"{max(timeout - 30, 60)}s"]
    if model:
        argv += ["--model", model]
    argv += ["--print", text]

    try:
        subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=timeout, shell=False)
    except subprocess.TimeoutExpired:
        pass  # fall through - check the transcript anyway, agy may have written a partial answer

    db, after_idx = None, -1
    try:
        cwd_map = json.load(open(os.path.join(os.path.dirname(CONV_DIR), "cache",
                                               "last_conversations.json"), encoding="utf-8"))
        cid = cwd_map.get(os.getcwd()) or cwd_map.get(os.getcwd().replace("\\", "/"))
        if isinstance(cid, dict):
            cid = cid.get("conversation_id") or cid.get("id")
        if cid:
            cand = os.path.join(CONV_DIR, f"{cid}.db")
            if os.path.exists(cand):
                db, after_idx = cand, before_steps.get(cand, -1)
    except Exception:
        pass

    if db is None:
        grown = [p for p in glob.glob(os.path.join(CONV_DIR, "*.db"))
                 if _max_step_idx(p) > before_steps.get(p, -1)]
        if grown:
            db = max(grown, key=os.path.getmtime)
            after_idx = before_steps.get(db, -1)

    if db is None:
        raise AgyError("agy recorded no conversation steps (auth/startup failure?) - check "
                        "~/.gemini/antigravity-cli/cli.log")

    answer, error = _read_conversation(db, after_idx)
    if error and len(answer) < 40:
        answer = ""

    if answer:
        return answer
    if error:
        raise AgyError(f"agy failed: {error}")
    raise AgyError(f"conversation recorded but contained no model text (unknown failure) - "
                    f"inspect {os.path.basename(db)}")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = sys.argv[1:]
    if not args:
        sys.exit("usage: agy_headless.py <prompt> | --file <path>")

    prompt = open(args[1], encoding="utf-8").read() if args[0] == "--file" else args[-1]

    try:
        print(ask(prompt))
        sys.exit(0)
    except AgyError as e:
        print(f"agy-headless: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
