"""Set-level matching and the residual-error taxonomy (proposal §5).

Pipeline for one example:
  1. per (pred, gold) pair, a full diff using BFCL value semantics (check_param);
  2. optimal bipartite matching (Hungarian) on those diffs, so call order never matters;
  3. labels: cross-call coordination errors first, then single-call errors.

An example can carry several labels. `primary` picks one by PRIORITY for stacked
bar charts. The official BFCL verdict is reported alongside (`bfcl_valid`), so
aggregate accuracy stays comparable with prior work.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ptcdiag.eval.bfcl_checker import ast_checker, check_param, standardize_string

CROSS_CALL = [
    "duplicate_call",        # identical calls beyond what the gold set contains
    "omitted_call",          # fewer calls than gold (unmatched gold calls)
    "extra_call",            # more calls than gold, not explained by duplication
    "cross_binding",         # a value that belongs to another gold call (swap or collision)
    "chimera_value",         # a string spliced from two gold values
    "inconsistent_shared_arg",  # gold shares a value across calls, prediction does not
]
SINGLE_CALL = [
    "wrong_function",
    "missing_param",
    "unexpected_param",
    "type_error",
    "wrong_value",
]
OTHER = ["syntax_error", "no_call"]
PRIORITY = OTHER + CROSS_CALL + SINGLE_CALL
ALL_LABELS = PRIORITY

NAME_MISMATCH_COST = 100.0


@dataclass
class PairDiff:
    name_ok: bool
    missing: list = field(default_factory=list)
    unexpected: list = field(default_factory=list)
    bad: dict = field(default_factory=dict)  # param -> "type" | "value"

    @property
    def exact(self):
        return self.name_ok and not self.missing and not self.unexpected and not self.bad

    @property
    def cost(self):
        if not self.name_ok:
            return NAME_MISMATCH_COST
        return float(len(self.missing) + len(self.unexpected) + len(self.bad))


def value_ok(func_desc, param, value, acceptable):
    """BFCL verdict for a single parameter value."""
    if func_desc is None or param not in func_desc["parameters"]["properties"]:
        return False
    try:
        return check_param(func_desc["parameters"]["properties"], param, value, acceptable)["valid"]
    except Exception:
        return False


def pair_diff(functions, pred, gold):
    """Diff one predicted call {"name", "arguments"} against one gold (name, params)."""
    gname, gparams = gold
    if pred["name"] != gname:
        return PairDiff(name_ok=False)
    desc = next((f for f in functions if f["name"] == gname), None)
    props = desc["parameters"]["properties"] if desc else {}
    required = desc["parameters"].get("required", []) if desc else []
    args = pred["arguments"]
    d = PairDiff(name_ok=True)
    for p in required:
        if p not in args:
            d.missing.append(p)
    for p, acc in gparams.items():
        if p not in args and "" not in acc and p not in d.missing:
            d.missing.append(p)
    for p, v in args.items():
        if p not in props or p not in gparams:
            d.unexpected.append(p)
            continue
        r = check_param(props, p, v, gparams[p])
        if not r["valid"]:
            d.bad[p] = "type" if r.get("error_type", "").startswith("type_error") else "value"
    return d


def _canon(v):
    if isinstance(v, str):
        return standardize_string(v)
    if isinstance(v, (list, tuple)):
        return tuple(_canon(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _canon(x)) for k, x in v.items()))
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def call_key(call):
    return (call["name"], tuple(sorted((k, _canon(v)) for k, v in call["arguments"].items())))


def _words(s):
    return [w for w in s.lower().replace(",", " ").split() if w]


def chimera_kind(value, own, sib):
    """'word' / 'char' if `value`, in the slot of `own`, looks spliced from `own` and a sibling's
    value `sib`, else None. A splice puts a piece of the sibling in place of a piece of its own:

    word: every word of value comes from own or sib, with a word only own has, a word only the
          sibling has, and without some word of own ("New York" + "Mexico City" -> "New City").
    char: value == x[:i] + y[j:] with {x, y} = {own, sib}, both pieces at least 3 characters and the
          sibling's piece absent from own, compared after BFCL standardisation (shorter or shared
          pieces match by chance: "Boston" + "a", dates that share "-01").
    The caller also rules out values that only shorten or extend own (`extends_own`).
    """
    if not all(isinstance(x, str) for x in (value, own, sib)):
        return None
    sv, so, ss = standardize_string(value), standardize_string(own), standardize_string(sib)
    if sv in (so, ss) or so == ss or not sv:
        return None
    wv, wo, ws = set(_words(value)), set(_words(own)), set(_words(sib))
    if wv and wv <= (wo | ws) and (wv & (wo - ws)) and (wv & (ws - wo)) and (wo - wv):
        return "word"
    for x, y, sib_first in ((so, ss, False), (ss, so, True)):
        for i in range(CHIMERA_MIN_PIECE, len(x)):
            if sv.startswith(x[:i]):
                rest = sv[i:]
                piece = x[:i] if sib_first else rest
                if len(rest) >= CHIMERA_MIN_PIECE and y.endswith(rest) and len(rest) < len(y) \
                        and piece not in so:
                    return "char"
    return None


CHIMERA_MIN_PIECE = 3


def extends_own(value, own):
    """`value` only shortens or extends its own value `own` (a truncation, or an overfill that keeps
    every word of it: "Sothe" for "Sotheby", "San Francisco CA" for "San Francisco"). Such a value
    holds no piece of a sibling in place of its own, so it is not a chimera."""
    sv, so = standardize_string(value), standardize_string(own)
    return bool(sv) and (so.startswith(sv) or sv.startswith(so) or set(_words(own)) <= set(_words(value)))


@dataclass
class Diagnosis:
    correct: bool           # our set-level verdict (all gold calls matched exactly, no extras)
    bfcl_valid: bool        # official BFCL checker verdict
    bfcl_error_type: str
    labels: list
    primary: str
    details: list = field(default_factory=list)
    matching: list = field(default_factory=list)  # [(pred_idx, gold_idx)]

    def to_dict(self):
        return {
            "correct": self.correct, "bfcl_valid": self.bfcl_valid,
            "bfcl_error_type": self.bfcl_error_type, "labels": self.labels,
            "primary": self.primary, "details": self.details, "matching": self.matching,
        }


def _primary(labels):
    for lab in PRIORITY:
        if lab in labels:
            return lab
    return "correct"


def diagnose(example, parsed):
    """Classify one parsed output (ptcdiag.prompting.ParsedOutput) against an Example."""
    if not parsed.syntax_ok:
        return Diagnosis(False, False, "ast_decoder:decoder_failed", ["syntax_error"], "syntax_error",
                         details=[{"label": "syntax_error", "error": parsed.error}])

    preds = parsed.calls
    gold = example.gold_calls
    fns = example.functions
    bfcl = ast_checker(fns, parsed.model_output(), example.ground_truth, example.category)
    bfcl_valid = bool(bfcl["valid"])
    bfcl_err = bfcl.get("error_type", "") if not bfcl_valid else ""

    if not preds:
        details = [{"label": "omitted_call", "gold": gi} for gi in range(len(gold))]
        return Diagnosis(False, bfcl_valid, bfcl_err, ["no_call"], "no_call", details=details)

    diffs = [[pair_diff(fns, p, g) for g in gold] for p in preds]
    cost = np.array([[d.cost for d in row] for row in diffs])
    rows, cols = linear_sum_assignment(cost)
    matching = [(int(r), int(c)) for r, c in zip(rows, cols)]
    pred_to_gold = dict(matching)
    matched_gold = set(pred_to_gold.values())

    labels, details = set(), []

    # Duplicates: a group of identical predicted calls larger than the number of gold
    # calls that this call satisfies.
    groups = {}
    for i, p in enumerate(preds):
        groups.setdefault(call_key(p), []).append(i)
    surplus = set()
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        mult = sum(1 for gi in range(len(gold)) if diffs[idxs[0]][gi].exact)
        if len(idxs) > max(mult, 1):
            labels.add("duplicate_call")
            # the copies that are not needed for an exact gold match are surplus
            exact_ok = [i for i in idxs if i in pred_to_gold and diffs[i][pred_to_gold[i]].exact]
            extra = [i for i in idxs if i not in exact_ok][: len(idxs) - max(mult, 1)]
            surplus.update(extra)
            details.append({"label": "duplicate_call", "preds": idxs, "surplus": extra})

    for gi in range(len(gold)):
        if gi not in matched_gold:
            labels.add("omitted_call")
            details.append({"label": "omitted_call", "gold": gi})
    for pi in range(len(preds)):
        if pi not in pred_to_gold and pi not in surplus:
            labels.add("extra_call")
            details.append({"label": "extra_call", "pred": pi})

    for pi, gi in matching:
        d = diffs[pi][gi]
        if d.exact or pi in surplus:
            continue
        if not d.name_ok:
            labels.add("wrong_function")
            details.append({"label": "wrong_function", "pred": pi, "gold": gi,
                            "got": preds[pi]["name"], "want": gold[gi][0]})
            continue
        for p in d.missing:
            labels.add("missing_param")
            details.append({"label": "missing_param", "pred": pi, "gold": gi, "param": p})
        for p in d.unexpected:
            labels.add("unexpected_param")
            details.append({"label": "unexpected_param", "pred": pi, "gold": gi, "param": p})
        desc = example.function(gold[gi][0])
        for p, kind in d.bad.items():
            v = preds[pi]["arguments"][p]
            if kind == "type":
                labels.add("type_error")
                details.append({"label": "type_error", "pred": pi, "gold": gi, "param": p, "value": v})
                continue
            others = [gj for gj, (n, gp) in enumerate(gold)
                      if gj != gi and n == gold[gi][0] and p in gp]
            owner = next((gj for gj in others if value_ok(desc, p, v, gold[gj][1][p])), None)
            if owner is not None:
                # collision: another prediction already carries the owner's value
                partner = next((pj for pj, gj in matching if gj == owner), None)
                collision = partner is not None and p in preds[partner]["arguments"] and \
                    _canon(preds[partner]["arguments"][p]) == _canon(v)
                labels.add("cross_binding")
                details.append({"label": "cross_binding", "pred": pi, "gold": gi, "param": p,
                                "value": v, "owner_gold": owner,
                                "kind": "collision" if collision else "swap"})
                continue
            kind_c, pair = None, None
            mine = [a for a in gold[gi][1][p] if isinstance(a, str) and a != ""]
            theirs = [a for gj in others for a in gold[gj][1][p] if isinstance(a, str) and a != ""]
            if isinstance(v, str) and any(extends_own(v, a) for a in mine):
                mine = []  # a truncation or overfill of the own value, not a splice
            for a in mine:
                for b in theirs:
                    kind_c = chimera_kind(v, a, b)
                    if kind_c:
                        pair = (a, b)
                        break
                if kind_c:
                    break
            if kind_c:
                labels.add("chimera_value")
                details.append({"label": "chimera_value", "pred": pi, "gold": gi, "param": p,
                                "value": v, "sources": pair, "kind": kind_c})
                continue
            labels.add("wrong_value")
            details.append({"label": "wrong_value", "pred": pi, "gold": gi, "param": p, "value": v})

    # Shared arguments: gold calls of the same function that accept a common value for p.
    by_func = {}
    for gi, (n, gp) in enumerate(gold):
        by_func.setdefault(n, []).append(gi)
    gold_to_pred = {gi: pi for pi, gi in matching}
    for n, gis in by_func.items():
        if len(gis) < 2:
            continue
        for p in set.intersection(*(set(gold[gi][1]) for gi in gis)):
            accs = [{_canon(a) for a in gold[gi][1][p] if a != ""} for gi in gis]
            if not set.intersection(*accs):
                continue
            vals = [preds[gold_to_pred[gi]]["arguments"].get(p) for gi in gis if gi in gold_to_pred]
            vals = [_canon(v) for v in vals if v is not None]
            desc = example.function(n)
            any_bad = any(gi in gold_to_pred and p in preds[gold_to_pred[gi]]["arguments"]
                          and not value_ok(desc, p, preds[gold_to_pred[gi]]["arguments"][p], gold[gi][1][p])
                          for gi in gis)
            if len(set(vals)) > 1 and any_bad:
                labels.add("inconsistent_shared_arg")
                details.append({"label": "inconsistent_shared_arg", "function": n, "param": p,
                                "values": [str(v) for v in vals]})

    correct = not labels and len(preds) == len(gold)
    labels = [lab for lab in PRIORITY if lab in labels]
    return Diagnosis(correct, bfcl_valid, bfcl_err, labels, _primary(labels),
                     details=details, matching=matching)


def is_cross_call(labels):
    return any(lab in CROSS_CALL for lab in labels)


def is_single_call(labels):
    return any(lab in SINGLE_CALL for lab in labels)
