"""One line of result for a finished (tag, scene), against the Y control.

Used by the campaign monitor so a scene finishing reports what it FOUND, not
just that it finished.
"""
from __future__ import annotations
import glob, json, os, re, sys

SINCE = "2026-09-03"   # tags are reused across campaigns; scope by date


def load(tag):
    out = {}
    for jd in sorted(glob.glob("outputs/2026-*/*")):
        if jd.split("/")[1] < SINCE:
            continue
        ov = os.path.join(jd, ".hydra", "overrides.yaml")
        if not os.path.exists(ov):
            continue
        m = re.search(r"\+run_tag=(\S+)", open(ov).read())
        if not m or m.group(1) != tag:
            continue
        ts = jd.split("/")[-2].replace("-", "") + "_" + jd.split("/")[-1].replace("-", "")
        p = f"outputs/{ts}/episodes.jsonl"
        if os.path.exists(p):
            for line in open(p):
                e = json.loads(line)
                out[e["episode_id"]] = e
    return out


def half(k):
    return "cross" if "cross_anchor" in k else "in"


def main():
    tag, scene = sys.argv[1], sys.argv[2]
    T, Y = load(tag), load("Y")
    keys = sorted(k for k in T if k.startswith(scene) and k in Y)
    if not keys:
        print(f"{tag} {scene}: no comparable episodes yet")
        return
    bits = []
    for h in ("in", "cross"):
        kk = [k for k in keys if half(k) == h]
        if not kk:
            continue
        t = sum(T[k]["success"] for k in kk)
        y = sum(Y[k]["success"] for k in kk)
        bits.append(f"{h} {t:.0f}/{len(kk)} (Y {y:.0f})")
    # The mechanism, not just the score. ZZ's whole prediction is that tracks
    # excluded by `min_presence` become candidates at all, so report the
    # episodes whose target track sat below the 0.45 gate under the control and
    # whether they now get proposed. SR can stay flat while this moves, and
    # that would still tell us the deadlock is real and lives downstream.
    def best_p(e):
        v = e.get("target_tracks") or []
        return max((t.get("p", 0.0) for t in v), default=None)

    gated = [k for k in keys
             if "cross_anchor" in k and (best_p(Y[k]) is not None)
             and best_p(Y[k]) < 0.45 and not Y[k]["success"]]
    mech = ""
    if gated:
        was = sum(1 for k in gated if (Y[k]["cand_n_obs"] or 0) > 0)
        now = sum(1 for k in gated if (T[k]["cand_n_obs"] or 0) > 0)
        rescued = sum(1 for k in gated if T[k]["success"])
        mech = (f"  | gated-by-presence {len(gated)}: proposed {was}->{now},"
                f" rescued {rescued}")

    # P(within 1.25 m) is the metric being optimised: pooled over 452
    # cross_anchor episodes, P(success|close) = 0.895 against 0.121 beyond, and
    # across conditions corr(P(close), SR) = 0.963. It has four times the
    # resolution of binary success on the same episodes.
    def close(e):
        return (e.get("gt_min_range_m") or 99) < 1.25

    ck = [k for k in keys if half(k) == "cross"]
    pcy = sum(1 for k in ck if close(Y[k])) / len(ck) if ck else 0.0
    pct = sum(1 for k in ck if close(T[k])) / len(ck) if ck else 0.0
    convy = ([Y[k]["success"] for k in ck if close(Y[k])] or [0])
    convt = ([T[k]["success"] for k in ck if close(T[k])] or [0])
    mech = (f"  | P(close) {pcy:.2f}->{pct:.2f}"
            f"  conv|close {sum(convy)/len(convy):.2f}->{sum(convt)/len(convt):.2f}")

    won = [k for k in keys if T[k]["success"] > Y[k]["success"]]
    lost = [k for k in keys if T[k]["success"] < Y[k]["success"]]
    tot = sum(T[k]["success"] for k in keys)
    ytot = sum(Y[k]["success"] for k in keys)
    print(f"{tag} {scene[:5]} n={len(keys)}: " + "  ".join(bits)
          + f"  | total {tot:.0f} vs Y {ytot:.0f} ({tot - ytot:+.0f})"
          + f"  | flips +{len(won)}/-{len(lost)}" + mech)


if __name__ == "__main__":
    main()
