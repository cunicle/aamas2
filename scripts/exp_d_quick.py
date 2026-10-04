"""Quick numbers for experiment D, for the run report (the paper's analysis is done elsewhere).

  python scripts/exp_d_quick.py results/dream/bfcl_tolerant*.jsonl results/*/agents_d.jsonl \
      results/*/agents_pos.jsonl

Canvas records (run_dllm.py): set accuracy and parse rate per (file, decoding, surplus, lengths).
BFCL team records (run_agents_bfcl.py): set accuracy and unparsed teams per (file, protocol).
Choose-N team records (run_agents.py): teams with a duplicate city per (file, protocol, variant).
"""

import json
import sys
from collections import defaultdict


def main():
    canvas, bfcl, choose = defaultdict(list), defaultdict(list), defaultdict(list)
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            for r in map(json.loads, f):
                if "error" in r:
                    print(f"ERROR in {path}: {r['id']}: {r['error']}")
                    continue
                if "cfg_tag" in r:
                    key = (path, r["cfg_tag"], r.get("surplus", 0), r.get("length_mode", "oracle"))
                    canvas[key].append((r["diagnosis"]["correct"], r["syntax_ok"]))
                elif "team" in r:
                    bfcl[path, r["protocol"]].append((r["team"]["diagnosis"]["correct"], not r["team"]["syntax_ok"]))
                elif "cities" in r:
                    cities = [c for c in r["cities"] if c is not None]
                    choose[path, r["protocol"], r["variant"]].append(
                        (len(cities) != len(set(cities)), len(cities) < r["n"]))
    if canvas:
        print("| file | decoding | surplus | lengths | n | set acc % | parses % |\n|---|---|---|---|---|---|---|")
        for (path, tag, s, lm), v in sorted(canvas.items()):
            print(f"| {path} | {tag} | {s} | {lm} | {len(v)} | {100 * sum(a for a, _ in v) / len(v):.1f} "
                  f"| {100 * sum(b for _, b in v) / len(v):.1f} |")
    if bfcl:
        print("\n| file | protocol | teams | set acc % | unparsed % |\n|---|---|---|---|---|")
        for (path, p), v in sorted(bfcl.items()):
            print(f"| {path} | {p} | {len(v)} | {100 * sum(a for a, _ in v) / len(v):.1f} "
                  f"| {100 * sum(b for _, b in v) / len(v):.1f} |")
    if choose:
        print("\n| file | protocol | variant | teams | duplicate % | some agent unparsed % |\n|---|---|---|---|---|---|")
        for (path, p, var), v in sorted(choose.items()):
            print(f"| {path} | {p} | {var} | {len(v)} | {100 * sum(a for a, _ in v) / len(v):.1f} "
                  f"| {100 * sum(b for _, b in v) / len(v):.1f} |")


if __name__ == "__main__":
    main()
