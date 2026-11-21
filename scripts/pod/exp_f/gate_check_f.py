"""Experiment F gates 1-2 (EXP_F_PROMPT.md section 3), read from /tmp/gate_f after gate_f.sh.

Gate 1: 132 tests pass, `bash -n scripts/run_minimal.sh` prints nothing, the data files are
identical to experiment D's (md5), and the dry run printed every new wording. Gate 2: for both
models, every agent of the 5 + 5 gate teams (original sim-label and sim-rule) got the user text
and gave the output of the old record with the same request and protocol: BFCL against
results/<model>/agents_c1.jsonl, choose-N against agents.jsonl (sim-label) and agents_rule.jsonl
(sim-rule). Prints "ALL GATES PASS" on the last line only if everything holds.
"""

import json
import os
import re

T = "/tmp/gate_f"
WORDINGS = ["sim-label@para", "sim-label@ordinal", "sim-label@first",
            "sim-rule@para", "sim-rule@direct", "sim-rule@first"]
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def team_gate(name, new_path, old_paths, keep):
    new = load(new_path)
    old = {}
    for p in old_paths:
        for r in load(p):
            if "error" not in r and keep(r):
                old.setdefault((r["id"], r["protocol"]), r)
    print(f"\n[{name}] {len(new)} teams: {sorted({r['id'] for r in new})}")
    check(len(new) == 10 and sorted({r["protocol"] for r in new}) == ["sim-label", "sim-rule"],
          f"{name}: 5 requests x (sim-label, sim-rule)")
    check(all("error" not in r for r in new), f"{name}: no error field")
    same = diff = missing = 0
    for r in new:
        o = old.get((r["id"], r["protocol"]))
        if o is None or "error" in r:
            missing += 1
            print(f"  no old record for {r['id']} {r['protocol']}")
            continue
        for a, b in zip(r["agents"], o["agents"]):
            if a["user_text"] == b["user_text"] and a["text"] == b["text"]:
                same += 1
            else:
                diff += 1
                print(f"  DIFFERENT {r['id']} {r['protocol']} agent {a['i']}\n    old: {b['text']!r}\n    new: {a['text']!r}")
        if len(r["agents"]) != len(o["agents"]):
            diff += 1
            print(f"  DIFFERENT team size {r['id']} {r['protocol']}")
    print(f"  agents identical {same}, different {diff}, teams with no old record {missing}")
    check(same > 0 and diff == 0 and missing == 0, f"{name}: user text and output identical to the old records")


def main():
    print("== gate 1 (CPU)")
    times = open(f"{T}/times.txt").read()
    check("prepare_data exit=0" in times, "prepare_data.py exit 0")
    check(os.path.getsize(f"{T}/data_md5_diff.txt") == 0, "BFCL / choose files: md5 identical to experiment D's")
    tail = open(f"{T}/pytest.txt").read().strip().splitlines()[-1]
    print(f"  pytest: {tail}")
    check(re.search(r"\b132 passed\b", tail) and not re.search(r"failed|error", tail), "pytest: 132 passed")
    check(os.path.getsize(f"{T}/bash_n.txt") == 0, "bash -n scripts/run_minimal.sh: no output")
    for f in ("texts_bfcl.txt", "texts_choose.txt"):
        text = open(f"{T}/{f}", encoding="utf-8").read()
        heads = [line for line in text.splitlines() if line.startswith("=====")]
        check(len(heads) == 2 * len(WORDINGS) and all(sum(w in h for h in heads) == 2 for w in WORDINGS),
              f"{f}: 2 requests x {len(WORDINGS)} wordings printed")
    print("  old records:\n" + "".join("    " + line for line in open(f"{T}/old_lines.txt")))

    print("\n== gate 2 (GPU)")
    greedy = lambda r: r.get("variant") == "list" and r.get("temperature") == 0.0 and r.get("seed") == 0
    for tag in ("qwen", "dream"):
        team_gate(f"{tag} BFCL", f"{T}/{tag}_bfcl.jsonl", [f"results/{tag}/agents_c1.jsonl"],
                  lambda r: r.get("subset") == "sym" and r.get("length_mode") == "oracle")
        team_gate(f"{tag} choose-N", f"{T}/{tag}_choose.jsonl",
                  [f"results/{tag}/agents.jsonl", f"results/{tag}/agents_rule.jsonl"], greedy)
    print("\nALL GATES PASS" if ok else "\nGATES FAILED")


if __name__ == "__main__":
    main()
