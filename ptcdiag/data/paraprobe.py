"""ParaProbe: synthetic parallel tool-call probes with controlled factors (proposal §6.3).

Factors (all recorded in Example.meta):
  n          number of calls
  entities   "single" (one-word names), "shared_prefix" (San Diego / San Jose ...),
             "multi_token" (two-word names without shared prefixes)
  ambiguity  "ordered"    - entities listed in call order
             "scrambled"  - entities mentioned inside a sentence in a different role/order
             "derived"    - entities given indirectly (country -> capital); knowledge-light
  shared     "none" | "shared" (one unit for all calls) | "per_call" (a unit per call)
  mix        "same" (one function, BFCL parallel) | "mixed" (two functions, parallel_multiple)

Ground truth follows BFCL conventions, so the same checker and taxonomy apply.
"""

import itertools
import random

from ptcdiag.types import Example

CITIES = {
    "single": ["Paris", "Tokyo", "Berlin", "Madrid", "Rome", "Cairo", "Lima", "Oslo",
               "Seoul", "Vienna", "Dublin", "Prague"],
    "shared_prefix": ["San Francisco", "San Diego", "San Jose", "San Antonio", "New York",
                      "New Delhi", "New Orleans", "Las Vegas", "Los Angeles", "Saint Louis",
                      "Saint Petersburg", "Port Louis"],
    "multi_token": ["Buenos Aires", "Mexico City", "Hong Kong", "Cape Town", "Kuala Lumpur",
                    "Tel Aviv", "Abu Dhabi", "Rio de Janeiro", "Addis Ababa", "Salt Lake City",
                    "Ho Chi Minh City", "Panama City"],
}
# Shared-prefix groups, so a draw of n cities shares prefixes as much as possible.
PREFIX_GROUPS = [["San Francisco", "San Diego", "San Jose", "San Antonio"],
                 ["New York", "New Delhi", "New Orleans"],
                 ["Las Vegas", "Los Angeles"],
                 ["Saint Louis", "Saint Petersburg"]]

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Germany": "Berlin", "Spain": "Madrid",
            "Italy": "Rome", "Egypt": "Cairo", "Peru": "Lima", "Norway": "Oslo",
            "South Korea": "Seoul", "Austria": "Vienna", "Ireland": "Dublin",
            "Czech Republic": "Prague"}

UNITS = ["celsius", "fahrenheit"]

WEATHER_FN = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "dict",
        "properties": {
            "city": {"type": "string", "description": "Name of the city, e.g. 'Paris'."},
            "unit": {"type": "string", "enum": UNITS,
                     "description": "Temperature unit. Default is celsius."},
        },
        "required": ["city"],
    },
}
TIME_FN = {
    "name": "get_local_time",
    "description": "Get the current local time in a city.",
    "parameters": {
        "type": "dict",
        "properties": {
            "city": {"type": "string", "description": "Name of the city, e.g. 'Paris'."},
            "format": {"type": "string", "enum": ["12h", "24h"],
                       "description": "Clock format. Default is 24h."},
        },
        "required": ["city"],
    },
}


def _draw_cities(rng, entities, n):
    if entities == "shared_prefix":
        groups = [g for g in PREFIX_GROUPS if len(g) >= 2]
        rng.shuffle(groups)
        out = []
        for g in groups:
            out += rng.sample(g, len(g))
            if len(out) >= n:
                break
        if len(out) < n:
            pool = [c for c in CITIES["shared_prefix"] if c not in out]
            out += rng.sample(pool, n - len(out))
        return out[:n]
    return rng.sample(CITIES[entities], n)


def _join(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


ROLES = [("I'm flying out of {}", "the city I'm flying out of"),
         ("my sister lives in {}", "the city where my sister lives"),
         ("our office is in {}", "the city where our office is"),
         ("the conference is in {}", "the conference city"),
         ("my parents retired to {}", "the city my parents retired to"),
         ("we have a client in {}", "our client's city")]


def _referents(rng, cities, ambiguity):
    """(context sentence, {city: phrase used to refer to it in the request})."""
    if ambiguity == "ordered":
        return "", {c: c for c in cities}
    if ambiguity == "derived":
        inv = {v: k for k, v in CAPITALS.items()}
        return "", {c: f"the capital of {inv[c]}" for c in cities}
    if ambiguity == "scrambled":
        roles = rng.sample(ROLES, len(cities))
        pairs = list(zip(cities, roles))
        rng.shuffle(pairs)  # mention order differs from call order
        context = "Some context: " + "; ".join(r[0].format(c) for c, r in pairs) + ". "
        return context, {c: r[1] for c, r in pairs}
    raise ValueError(ambiguity)


def make_example(rng, idx, n, entities, ambiguity, shared, mix):
    if ambiguity == "derived":
        entities = "single"  # capitals are one-word names in CAPITALS
    cities = _draw_cities(rng, entities, n)
    context, ref = _referents(rng, cities, ambiguity)

    kinds = ["weather"] * n if mix == "same" else \
        ["weather" if i % 2 == 0 else "time" for i in range(n)]
    weather = [c for c, k in zip(cities, kinds) if k == "weather"]

    units = [None] * n
    unit_text = ""
    if shared == "shared":
        u = rng.choice(UNITS)
        units = [u if k == "weather" else None for k in kinds]
        unit_text = f" Report every temperature in {u}."
    elif shared == "per_call":
        units = [rng.choice(UNITS) if k == "weather" else None for k in kinds]
        unit_text = " " + "; ".join(f"use {u} for {ref[c]}" for c, u, k in zip(cities, units, kinds)
                                    if k == "weather") + "."

    if mix == "same":
        fns = [WEATHER_FN]
        ask = f"{context}Get the current weather in {_join([ref[c] for c in cities])}.{unit_text}"
    else:
        fns = [WEATHER_FN, TIME_FN]
        t = [ref[c] for c, k in zip(cities, kinds) if k == "time"]
        ask = (f"{context}Get the current weather in {_join([ref[c] for c in weather])}, "
               f"and the local time in {_join(t)}.{unit_text}")

    gt = []
    for c, u, k in zip(cities, units, kinds):
        if k == "weather":
            gt.append({"get_weather": {"city": [c], "unit": [u] if u else ["celsius", ""]}})
        else:
            gt.append({"get_local_time": {"city": [c], "format": ["24h", ""]}})

    cat = "probe_parallel" if mix == "same" else "probe_parallel_multiple"
    return Example(
        id=f"probe_{idx}",
        category=cat,
        messages=[{"role": "user", "content": ask.strip()}],
        functions=fns,
        ground_truth=gt,
        meta={"source": "paraprobe", "n": n, "entities": entities, "ambiguity": ambiguity,
              "shared": shared, "mix": mix, "cities": cities},
    )


DEFAULT_GRID = {
    "n": [2, 3, 4, 6],
    "entities": ["single", "shared_prefix", "multi_token"],
    "ambiguity": ["ordered", "scrambled", "derived"],
    "shared": ["none", "shared", "per_call"],
    "mix": ["same", "mixed"],
}


def generate(per_cell=20, seed=0, grid=None):
    grid = grid or DEFAULT_GRID
    rng = random.Random(seed)
    out, idx = [], 0
    keys = list(grid)
    seen = set()
    for combo in itertools.product(*(grid[k] for k in keys)):
        cell = dict(zip(keys, combo))
        if cell["ambiguity"] == "derived" and cell["entities"] != "single":
            continue  # derived always uses one-word capitals; avoid duplicate cells
        key = tuple(sorted(cell.items()))
        if key in seen:
            continue
        seen.add(key)
        for _ in range(per_cell):
            out.append(make_example(rng, idx, **cell))
            idx += 1
    return out
