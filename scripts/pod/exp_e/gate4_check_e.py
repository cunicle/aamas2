"""Experiment E gate 4 (EXP_E_PROMPT.md section 3), after both chains. For each appended file:
the first N lines (N from gate_e.sh, before the runs) are byte-identical to the copy in /tmp/old,
the number of new lines is exactly the runbook's, and no new record has an error field. Also
lists the new records by condition and counts the two new probe files. Prints
"GATE 4 PASS" on the last line only if everything holds.
"""

import json
import os
from collections import Counter

FILES = [  # file, new lines expected
    ("results/dream/bfcl_closer.jsonl", 400),
    ("results/qwen/bfcl_surplus.jsonl", 1200),
    ("results/llada2/bfcl_surplus.jsonl", 500),
]
PROBES = ["results/dream/length_prior_closer_s4.jsonl", "results/llada2/length_prior_s2.jsonl"]
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)


def main():
    for path, want in FILES:
        old_path = "/tmp/old/" + path.replace("/", "_")
        with open(old_path, "rb") as f:
            old = f.read()
        with open(path, "rb") as f:
            new = f.read()
        n_old = old.count(b"\n")
        lines = new.split(b"\n")
        assert lines[-1] == b"", f"{path} does not end with a newline"
        lines = lines[:-1]
        head = b"".join(line + b"\n" for line in lines[:n_old])
        added = [json.loads(line) for line in lines[n_old:]]
        print(f"\n[{path}] old {n_old} lines, now {len(lines)}, new {len(added)}")
        check(head == old, f"first {n_old} lines byte-identical to /tmp/old")
        check(len(added) == want, f"new lines = {want}")
        errs = [r for r in added if "error" in r]
        for r in errs:
            print(f"  error: {r['id']} {r.get('cfg_tag')} {r['error']}")
        check(not errs, "no error field in the new records")
        by = Counter((r.get("surplus", 0), r.get("cfg_tag"), r.get("closer_in_slot", False)) for r in added)
        for k, v in sorted(by.items(), key=str):
            print(f"  new: surplus={k[0]} cfg_tag={k[1]} closer_in_slot={k[2]}: {v}")
        keys = Counter((r["id"], r.get("surplus", 0), r.get("cfg_tag"), r.get("closer_in_slot", False))
                       for r in [json.loads(line) for line in lines])
        dup = [k for k, v in keys.items() if v > 1]
        check(not dup, f"no duplicate (id, surplus, cfg_tag, closer_in_slot) in the whole file ({len(dup)})")
    for path in PROBES:
        n = sum(1 for _ in open(path)) if os.path.exists(path) else 0
        print(f"\n[{path}] {n} slots")
        check(n == 3063, "probe: 3,063 slots")
    print("\nGATE 4 PASS" if ok else "\nGATE 4 FAILED")


if __name__ == "__main__":
    main()
