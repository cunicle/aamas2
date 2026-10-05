"""Experiment E gates 1-2 (EXP_E_PROMPT.md section 3), read from /tmp/gate_e after gate_e.sh.

Gate 1: 121 tests pass, `bash -n scripts/run_minimal.sh` prints nothing, BFCL files identical to
experiment D's (md5). Gate 2: every gate record's text equals the old record with the same id,
surplus and cfg_tag (Dream closer-in-slot s=2 k=1, Qwen s=8, LLaDA2.0 s=2 k=4), and the probe's
p_close is within 0.01 of the old record of the same slot (id, call, param). Prints
"ALL GATES PASS" on the last line only if everything holds.
"""

import json
import os
import re

T = "/tmp/gate_e"
OLD = "/tmp/old"
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def text_gate(name, new_path, old_path, want_n, want_surplus, want_tag, extra=None):
    new = load(new_path)
    old = {}
    for r in load(old_path):
        if extra and not extra(r):
            continue
        old.setdefault((r["id"], r.get("surplus", 0), r.get("cfg_tag")), r)
    print(f"\n[{name}] {len(new)} records: {[r['id'] for r in new]}")
    check(len(new) == want_n, f"{name}: {want_n} records")
    check(all("error" not in r for r in new), f"{name}: no error field")
    check(all(r.get("surplus", 0) == want_surplus and r.get("cfg_tag") == want_tag for r in new),
          f"{name}: every record has surplus={want_surplus}, cfg_tag={want_tag}")
    same = diff = missing = 0
    for r in new:
        o = old.get((r["id"], r.get("surplus", 0), r.get("cfg_tag")))
        if o is None:
            missing += 1
            print(f"  no old record for {r['id']}")
        elif o["text"] == r["text"]:
            same += 1
        else:
            diff += 1
            print(f"  DIFFERENT {r['id']}\n    old: {o['text']!r}\n    new: {r['text']!r}")
    print(f"  identical {same}, different {diff}, no old record {missing}")
    check(same == len(new) and diff == 0 and missing == 0, f"{name}: text identical to the old record")


def main():
    print("== gate 1 (CPU)")
    times = open(f"{T}/times.txt").read()
    check("prepare_data exit=0" in times, "prepare_data.py exit 0")
    check(os.path.getsize(f"{T}/data_md5_diff.txt") == 0, "BFCL / choose files: md5 identical to experiment D's")
    tail = open(f"{T}/pytest.txt").read().strip().splitlines()[-1]
    print(f"  pytest: {tail}")
    check(re.search(r"\b121 passed\b", tail) and not re.search(r"failed|error", tail), "pytest: 121 passed")
    check(os.path.getsize(f"{T}/bash_n.txt") == 0, "bash -n scripts/run_minimal.sh: no output")
    print("  old records:\n" + "".join("    " + line for line in open(f"{T}/old_lines.txt")))

    print("\n== gate 2 (GPU)")
    text_gate("Dream closer-in-slot s=2 k=1", f"{T}/dream_closer_s2.jsonl",
              f"{OLD}/results_dream_bfcl_closer.jsonl", 5, 2, "confidence_k1_tnone_bfull_T0.0",
              extra=lambda r: r.get("closer_in_slot", False))
    text_gate("Qwen AR s=8", f"{T}/qwen_s8.jsonl", f"{OLD}/results_qwen_bfcl_surplus.jsonl", 5, 8, "ar_greedy")
    text_gate("LLaDA2.0 s=2 k=4", f"{T}/llada2_s2_k4.jsonl", f"{OLD}/results_llada2_bfcl_surplus.jsonl",
              5, 2, "confidence_k4_tnone_b32_T0.0")
    old_ids = [r["id"] for r in load(f"{OLD}/results_llada2_bfcl_surplus.jsonl")[:5]]
    new_ids = [r["id"] for r in load(f"{T}/llada2_s2_k4.jsonl")]
    check(new_ids == old_ids, f"LLaDA2.0 --per-data 50 --limit 5 = the first 5 of the 100 old items {old_ids}")

    new = load(f"{T}/llada2_probe_s1.jsonl")
    old = {(r["id"], r["call"], r["param"]): r for r in load("results/llada2/length_prior_s1.jsonl")}
    ids = sorted({r["id"] for r in new})
    print(f"\n[LLaDA2.0 probe s=1] {len(new)} slots of {ids}")
    want = [k for k in old if k[0] in ids]
    check(len(new) == len(want) and len(new) > 0, f"probe: same slots as the old file for these items ({len(want)})")
    diffs = []
    for r in new:
        o = old.get((r["id"], r["call"], r["param"]))
        if o is None:
            print(f"  no old slot {r['id']} {r['call']} {r['param']}")
            diffs.append(float("inf"))
            continue
        d = abs(r["p_close"] - o["p_close"])
        diffs.append(d)
        print(f"  {r['id']} call {r['call']} {r['param']}: p_close old {o['p_close']:.6f} new {r['p_close']:.6f} |d| {d:.2e}")
    check(diffs and max(diffs) <= 0.01, f"probe: max |p_close - old| = {max(diffs) if diffs else 'n/a'} <= 0.01")

    print("\nALL GATES PASS" if ok else "\nSOME GATE FAILED")


if __name__ == "__main__":
    main()
