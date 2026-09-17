#!/usr/bin/env python3
"""Read an `audit_map_coverage` report and print scene verdicts, for a shell.

    python scripts/audit_verdicts.py REPORT.json passed [scene ...]
    python scripts/audit_verdicts.py REPORT.json failed [scene ...]

`passed` prints one space-separated line of scene ids, ready to hand to a
runner; `failed` prints one `scene: reason` per line. With no scene ids every
scene in the report is considered. A missing or unreadable report prints
nothing and exits 0 -- the caller decides what an empty list means, because
"no scene passed" and "the report is not there" want different messages.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    path, mode, wanted = sys.argv[1], sys.argv[2], set(sys.argv[3:])
    try:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
    except Exception:
        return 0
    rows = [r for r in report.get("scenes", [])
            if not wanted or r.get("scene") in wanted]
    if mode == "passed":
        names = sorted(r["scene"] for r in rows if r.get("verdict") == "pass")
        if names:
            print(" ".join(names))
    elif mode == "failed":
        for row in sorted(rows, key=lambda r: r.get("scene", "")):
            if row.get("verdict") != "pass":
                reasons = "; ".join(row.get("failures") or ["unknown"])
                print(f"{row.get('scene')}: {reasons}")
    else:
        print(f"unknown mode {mode!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
