"""Experiment F gate 4 (EXP_F_PROMPT.md section 3), after the chain. For each model:
agents_f1.jsonl holds every (request, wording) of the 104 team-symmetric requests x 6 wordings
exactly once, agents_f_choose.jsonl every (request, wording) of the 60 list requests x 6, no
record has an error field, the requests are the ones of the old runs, and every agent's user text
contains the request as the old sim-label record had it. Prints "GATE 4 PASS" on the last line
only if everything holds.
"""

import json
from collections import Counter

WORDINGS = ["sim-label@para", "sim-label@ordinal", "sim-label@first",
            "sim-rule@para", "sim-rule@direct", "sim-rule@first"]
NOTE = "\n\nYou are assistant "  # the original sim-label note starts here; the request is before it
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_gate(path, old_path, keep_old, n_requests):
    old = {r["id"]: r for r in load(old_path) if "error" not in r and r["protocol"] == "sim-label" and keep_old(r)}
    new = load(path)
    print(f"\n[{path}] {len(new)} records; old sim-label requests: {len(old)}")
    check(len(old) == n_requests, f"old run: {n_requests} requests")
    errs = [r for r in new if "error" in r]
    for r in errs:
        print(f"  error: {r['id']} {r['protocol']} {r['error']}")
    check(not errs, "no error field")
    keys = Counter((r["id"], r["protocol"]) for r in new)
    check(len(new) == n_requests * len(WORDINGS) and all(v == 1 for v in keys.values()),
          f"{n_requests} x {len(WORDINGS)} = {n_requests * len(WORDINGS)} records, each (request, wording) once")
    check(set(keys) == {(i, w) for i in old for w in WORDINGS}, "the requests of the old run x the 6 wordings")
    bad = agents = 0
    for r in new:
        o = old.get(r["id"])
        if o is None or "error" in r:
            continue
        for a, b in zip(r["agents"], o["agents"]):
            agents += 1
            request = b["user_text"].split(NOTE)[0]
            bad += a["user_text"].count(request) != 1 or a["i"] != b["i"]
        bad += len(r["agents"]) != len(o["agents"])
    check(bad == 0 and agents > 0, f"every agent's text holds the request once ({agents} agents, "
                                  f"{agents // len(WORDINGS)} per wording)")
    by = Counter(r["protocol"] for r in new)
    print("  " + ", ".join(f"{w}: {by[w]}" for w in WORDINGS))
    tags = Counter(r.get("decoding") for r in new)
    print(f"  decoding: {dict(tags)}")
    check(len(tags) == 1, "one decoding tag")


def main():
    for tag in ("qwen", "dream"):
        file_gate(f"results/{tag}/agents_f1.jsonl", f"results/{tag}/agents_c1.jsonl",
                  lambda r: r.get("subset") == "sym" and r.get("length_mode") == "oracle", 104)
        file_gate(f"results/{tag}/agents_f_choose.jsonl", f"results/{tag}/agents.jsonl",
                  lambda r: r.get("variant") == "list" and r.get("temperature") == 0.0 and r.get("seed") == 0, 60)
    print("\nGATE 4 PASS" if ok else "\nGATE 4 FAILED")


if __name__ == "__main__":
    main()
