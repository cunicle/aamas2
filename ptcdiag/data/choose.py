"""Choose-N probes: requests that leave open which calls to make.

  list:  "Get the current weather in any two of these cities: Denver, Paris, Austin, ..."
  open:  "Get the current weather in two different US cities of your choice."

Every call's city slot accepts every allowed city, so the BFCL checker passes any choice;
what matters is whether the N calls pick N *different* cities (scripts/choose_analysis.py).
All listed cities, and the first allowed city that sets the slot length, are one token
for the Qwen2 (Dream, Qwen2.5) and Ling (LLaDA2.0) tokenizers, so every slot has the same
length: the canvas says nothing about which call gets which city. In the open variant
the request does not either, so nothing but the decoding order keeps two calls apart.
"""

import random

from ptcdiag.data.paraprobe import WEATHER_FN
from ptcdiag.types import Example

# one token for Dream / Qwen2.5 / LLaDA2.0 (checked with their tokenizers)
ONE_TOKEN = ["Paris", "Berlin", "London", "Bern", "Singapore", "Toronto", "Chicago", "Boston",
             "Denver", "Seattle", "Miami", "Dallas", "Houston", "Atlanta", "Phoenix", "Austin",
             "Portland", "Detroit"]
ONE_TOKEN_US = ["Chicago", "Boston", "Denver", "Seattle", "Miami", "Dallas", "Houston", "Atlanta",
                "Phoenix", "Austin", "Portland", "Detroit"]
# any US city counts in the open variant (the slot length still admits one token)
US_CITIES = ONE_TOKEN_US + [
    "New York", "Los Angeles", "San Francisco", "San Diego", "San Jose", "San Antonio", "Philadelphia",
    "Washington", "Las Vegas", "Nashville", "Baltimore", "Milwaukee", "Albuquerque", "Tucson", "Fresno",
    "Sacramento", "Omaha", "Raleigh", "Minneapolis", "Cleveland", "Pittsburgh", "Cincinnati", "Orlando",
    "Tampa", "Memphis", "Louisville", "Columbus", "Charlotte", "Indianapolis", "Jacksonville", "Honolulu",
    "Anchorage", "Buffalo", "Newark", "Oakland", "Tulsa", "Wichita", "Boise", "Reno", "Salem", "Dayton",
    "Richmond", "Madison", "Spokane", "Tacoma", "Savannah", "Charleston", "Kansas City", "St. Louis",
    "Salt Lake City", "New Orleans", "Fort Worth", "El Paso", "Oklahoma City", "Long Beach", "Mesa",
    "Aurora", "Anaheim", "Riverside", "Lexington", "Stockton", "Durham", "Lincoln", "Orange",
]
CONTEXTS = ["", "I'm planning a road trip. ", "For a quick comparison, please help. ",
            "I need a few examples for a demo. ", "Surprise me. "]
PHRASES = ["different US cities of your choice", "different cities in the United States",
           "different American cities, any you like"]
WORDS = {2: "two", 3: "three", 4: "four"}


def _join(items):
    return ", ".join(items[:-1]) + " and " + items[-1]


def make_example(idx, n, variant, ctx, cities=None, phrase=None):
    if variant == "list":
        ask = f"{ctx}Get the current weather in any {WORDS[n]} of these cities: {_join(cities)}."
        allowed = cities
    elif variant == "open":
        ask = f"{ctx}Get the current weather in {WORDS[n]} {phrase}."
        cities, allowed = [], US_CITIES
    else:
        raise ValueError(variant)
    return Example(
        id=f"choose_{idx}",
        category="probe_parallel",
        messages=[{"role": "user", "content": ask}],
        functions=[WEATHER_FN],
        ground_truth=[{"get_weather": {"city": list(allowed)}} for _ in range(n)],
        meta={"source": "choose", "variant": variant, "n": n, "listed": cities},
    )


def generate(per_cell=20, seed=0):
    """For n in {2, 3, 4}: per_cell "list" items (n + 3 random cities in random order) and
    every context x phrase of the "open" variant (15; greedy decoding gives one answer
    per prompt, so open prompts are not repeated)."""
    rng = random.Random(seed)
    out = []
    for n in (2, 3, 4):
        for _ in range(per_cell):
            out.append(make_example(len(out), n, "list", rng.choice(CONTEXTS), cities=rng.sample(ONE_TOKEN, n + 3)))
        for ctx in CONTEXTS:
            for phrase in PHRASES:
                out.append(make_example(len(out), n, "open", ctx, phrase=phrase))
    return out
