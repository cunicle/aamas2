"""Proposition 1: closed form vs. the real Sampler on the symmetric toy model (CPU).

  python scripts/sim_prop1.py --trials 2000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.theory import fixed_k_sizes, p_valid_schedule, simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    args = ap.parse_args()
    print("| n | k | formula (sampling) | sampler T=1 | sampler greedy |")
    print("|---|---|---|---|---|")
    for n in (2, 3, 4, 6):
        for k in sorted({1, 2, 3, n}):
            if k > n:
                continue
            f = p_valid_schedule(n, fixed_k_sizes(n, k))
            s = simulate(n, k, trials=args.trials, temperature=1.0)
            g = simulate(n, k, trials=50, temperature=0.0, order="confidence")
            print(f"| {n} | {k} | {f:.3f} | {s:.3f} | {g:.3f} |")


if __name__ == "__main__":
    main()
