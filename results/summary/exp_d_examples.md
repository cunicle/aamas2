# Experiment D: two +1 canvases of the format-tolerant slots (Dream, k=1)

## parallel_multiple_14

request: Give me the population size of tigers in Bangladesh and India for the last 5 years. Also provide the projected population size of tigers in Nepal and Malaysia for the next 10 years.

tolerant: correct=True labels=[]; original interface +1: correct=False labels=['wrong_value']

| call | parameter | type | tolerant slot tokens (+1) | raw value | decoded value | original interface slot tokens (+1) |
|---|---|---|---|---|---|---|
| 0 | country | string | Bang · ladesh · \\n | `Bangladesh\n` | `Bangladesh` (filler removed) | Bang · ladesh ·  country |
| 0 | species | string | t · iger · \\n | `tiger\n` | `tiger` (filler removed) | T · iger ·  species |
| 0 | years | integer |   · 5 · \\n | ` 5\n` | ` 5` (filler removed) |   · 5 · 0 |
| 1 | country | string | India · \\n | `India\n` | `India` (filler removed) | India ·  country |
| 1 | species | string | t · iger · \\n | `tiger\n` | `tiger` (filler removed) | T · iger ·  species |
| 1 | years | integer |   · 5 · \\n | ` 5\n` | ` 5` (filler removed) |   · 5 · 0 |
| 2 | country | string | N · ep · al · \\n | `Nepal\n` | `Nepal` (filler removed) | N · ep · al ·  country |
| 2 | species | string | t · iger · \\n | `tiger\n` | `tiger` (filler removed) | T · iger ·  species |
| 2 | years | integer |   · 1 · 0 · \\n | ` 10\n` | ` 10` (filler removed) |   · 1 · 0 · 0 |
| 3 | country | string | Mal · aysia · \\n | `Malaysia\n` | `Malaysia` (filler removed) | Mal · aysia ·  country |
| 3 | species | string | t · iger · \\n | `tiger\n` | `tiger` (filler removed) | T · iger ·  species |
| 3 | years | integer |   · 1 · 0 · \\n | ` 10\n` | ` 10` (filler removed) |   · 1 · 0 · 0 |

tolerant text: `[{"name": "animal_population.get_history", "arguments": {"country": "Bangladesh", "species": "tiger", "years": 5}}, {"name": "animal_population.get_history", "arguments": {"country": "India", "species": "tiger", "years": 5}}, {"name": "animal_population.get_projection", "arguments": {"country": "Nepal", "species": "tiger", "years": 10}}, {"name": "animal_population.get_projection", "arguments": {"country": "Malaysia", "species": "tiger", "years": 10}}]`

original +1 text: `[{"name": "animal_population.get_history", "arguments": {"country": "Bangladesh country", "species": "Tiger species", "years": 50}}, {"name": "animal_population.get_history", "arguments": {"country": "India country", "species": "Tiger species", "years": 50}}, {"name": "animal_population.get_projection", "arguments": {"country": "Nepal country", "species": "Tiger species", "years": 100}}, {"name": "animal_population.get_projection", "arguments": {"country": "Malaysia country", "species": "Tiger species", "years": 100}}]`

## parallel_14

request: Calculate the Present Value of an investment paying $1000 per year, with an interest rate of 5%, for 10, 20 and 30 years.

tolerant: correct=True labels=[]; original interface +1: correct=False labels=['wrong_value']

| call | parameter | type | tolerant slot tokens (+1) | raw value | decoded value | original interface slot tokens (+1) |
|---|---|---|---|---|---|---|
| 0 | payment_per_year | integer |   · 1 · 0 · 0 · 0 · i | ` 1000i` | ` 1000` (filler removed) |   · 1 · 0 · 0 · 0 · 0 |
| 0 | interest_rate | float |   · 0 · . · 0 · 5 · f | ` 0.05f` | ` 0.05` (filler removed) |   · 0 · . · 0 · 5 · 0 |
| 0 | years | integer |   · 1 · 0 · i | ` 10i` | ` 10` (filler removed) |   · 1 · 0 · 0 |
| 1 | payment_per_year | integer |   · 1 · 0 · 0 · 0 · i | ` 1000i` | ` 1000` (filler removed) |   · 1 · 0 · 0 · 0 · 0 |
| 1 | interest_rate | float |   · 0 · . · 0 · 5 · f | ` 0.05f` | ` 0.05` (filler removed) |   · 0 · . · 0 · 5 · 0 |
| 1 | years | integer |   · 2 · 0 · i | ` 20i` | ` 20` (filler removed) |   · 2 · 0 · 0 |
| 2 | payment_per_year | integer |   · 1 · 0 · 0 · 0 · i | ` 1000i` | ` 1000` (filler removed) |   · 1 · 0 · 0 · 0 · 0 |
| 2 | interest_rate | float |   · 0 · . · 0 · 5 · f | ` 0.05f` | ` 0.05` (filler removed) |   · 0 · . · 0 · 5 · 0 |
| 2 | years | integer |   · 3 · 0 · i | ` 30i` | ` 30` (filler removed) |   · 3 · 0 · 0 |

tolerant text: `[{"name": "calculate_present_value", "arguments": {"payment_per_year": 1000, "interest_rate": 0.05, "years": 10}}, {"name": "calculate_present_value", "arguments": {"payment_per_year": 1000, "interest_rate": 0.05, "years": 20}}, {"name": "calculate_present_value", "arguments": {"payment_per_year": 1000, "interest_rate": 0.05, "years": 30}}]`

original +1 text: `[{"name": "calculate_present_value", "arguments": {"payment_per_year": 10000, "interest_rate": 0.050, "years": 100}}, {"name": "calculate_present_value", "arguments": {"payment_per_year": 10000, "interest_rate": 0.050, "years": 200}}, {"name": "calculate_present_value", "arguments": {"payment_per_year": 10000, "interest_rate": 0.050, "years": 300}}]`

## parallel_48

request: In a game of Pokemon GO, what moves can a Pikachu learn? Also, check if Bulbasaur can learn a specific move named 'Solar Beam'.

tolerant: correct=True labels=[]; original interface +1: correct=False labels=['wrong_value']

| call | parameter | type | tolerant slot tokens (+1) | raw value | decoded value | original interface slot tokens (+1) |
|---|---|---|---|---|---|---|
| 0 | pokemon | string | P · ik · achu · \\n | `Pikachu\n` | `Pikachu` (filler removed) | P · ik · achu · ' |
| 0 | move | string | Run · \\n | `Run\n` | `Run` (filler removed) | Run · ' |
| 1 | pokemon | string | B · ul · bas · aur · \\n | `Bulbasaur\n` | `Bulbasaur` (filler removed) | B · ul · bas · aur · ' |
| 1 | move | string | Solar ·  Beam · \\n | `Solar Beam\n` | `Solar Beam` (filler removed) | Solar ·  Beam · ' |

tolerant text: `[{"name": "PokemonGO.get_moves", "arguments": {"pokemon": "Pikachu", "move": "Run"}}, {"name": "PokemonGO.get_moves", "arguments": {"pokemon": "Bulbasaur", "move": "Solar Beam"}}]`

original +1 text: `[{"name": "PokemonGO.get_moves", "arguments": {"pokemon": "Pikachu'", "move": "Run'"}}, {"name": "PokemonGO.get_moves", "arguments": {"pokemon": "Bulbasaur'", "move": "Solar Beam'"}}]`

## parallel_116

request: "In a population of butterflies, the frequency of the dominant allele for wing color is 0.7. Can you calculate the frequency of the homozygous dominant genotype (AA), heterozygous genotype (Aa), and homozygous recessive genotype (aa) using the Hardy Weinberg Principle?"

tolerant: correct=True labels=[]; original interface +1: correct=False labels=['duplicate_call', 'wrong_value']

| call | parameter | type | tolerant slot tokens (+1) | raw value | decoded value | original interface slot tokens (+1) |
|---|---|---|---|---|---|---|
| 0 | allele_frequency | float |   · 0 · . · 7 · f | ` 0.7f` | ` 0.7` (filler removed) |   · 0 · . · 7 · 0 |
| 0 | genotype | string | AA · \\n | `AA\n` | `AA` (filler removed) | AA ·  AA |
| 1 | allele_frequency | float |   · 0 · . · 7 · f | ` 0.7f` | ` 0.7` (filler removed) |   · 0 · . · 7 · 0 |
| 1 | genotype | string | A · a · \\n | `Aa\n` | `Aa` (filler removed) | A · a ·  aa |
| 2 | allele_frequency | float |   · 0 · . · 7 · f | ` 0.7f` | ` 0.7` (filler removed) |   · 0 · . · 7 · 0 |
| 2 | genotype | string | aa · \\n | `aa\n` | `aa` (filler removed) | aa ·  aa |

tolerant text: `[{"name": "calculate_genotype_frequency", "arguments": {"allele_frequency": 0.7, "genotype": "AA"}}, {"name": "calculate_genotype_frequency", "arguments": {"allele_frequency": 0.7, "genotype": "Aa"}}, {"name": "calculate_genotype_frequency", "arguments": {"allele_frequency": 0.7, "genotype": "aa"}}]`

original +1 text: `[{"name": "calculate_genotype_frequency", "arguments": {"allele_frequency": 0.70, "genotype": "AA AA"}}, {"name": "calculate_genotype_frequency", "arguments": {"allele_frequency": 0.70, "genotype": "Aa aa"}}, {"name": "calculate_genotype_frequency", "arguments": {"allele_frequency": 0.70, "genotype": "aa aa"}}]`

## parallel_2 (still wrong: an overfill next to removed filler)

request: Calculate the resistance of a wire with a length of 5m and cross sectional area 0.01m² with resistivity of copper and aluminum

tolerant: correct=False labels=['wrong_value']; original interface +1: correct=False labels=['wrong_value']

| call | parameter | type | tolerant slot tokens (+1) | raw value | decoded value | slot class | original interface slot tokens (+1) |
|---|---|---|---|---|---|---|---|
| 0 | length | integer |   · 5 · 0 | ` 50` | ` 50` | overfill |   · 5 · 0 |
| 0 | area | float |   · 0 · . · 0 · 0 · 1 | ` 0.001` | ` 0.001` | other |   · 0 · . · 0 · 0 · 1 |
| 0 | resistivity | string | c · opper · \\n | `copper\n` | `copper` | own | c · opper · " |
| 1 | length | integer |   · 5 · 0 | ` 50` | ` 50` | overfill |   · 5 · 0 |
| 1 | area | float |   · 0 · . · 0 · 0 · 1 | ` 0.001` | ` 0.001` | other |   · 0 · . · 0 · 0 · 1 |
| 1 | resistivity | string | al · uminum · \\n | `aluminum\n` | `aluminum` | own | al · uminum · " |

tolerant text: `[{"name": "calculate_resistance", "arguments": {"length": 50, "area": 0.001, "resistivity": "copper"}}, {"name": "calculate_resistance", "arguments": {"length": 50, "area": 0.001, "resistivity": "aluminum"}}]`

original +1 text: `[{"name": "calculate_resistance", "arguments": {"length": 50, "area": 0.001, "resistivity": "copper"}}, {"name": "calculate_resistance", "arguments": {"length": 50, "area": 0.001, "resistivity": "aluminum"}}]`


# Experiment D: position-agent teams (pos-anon)

## BFCL parallel_17 (n=3)

user text of every agent (the request, unchanged):

> Get the latest closing prices and volumes for Apple Inc., Google LLC., and Microsoft Corporation in the New York Stock Exchange

### Dream: team correct=True labels=[]

- agent 1: `[{"name": "get_stock_data", "arguments": {"symbol": "AAPL", "data_points": ["price", "volume"]}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}]`
- agent 2: `[{"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "GOOG", "data_points": ["price", "volume"]}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}]`
- agent 3: `[{"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "MSFT", "data_points": ["price", "volume"]}}]`

  team: `[{"name": "get_stock_data", "arguments": {"symbol": "AAPL", "data_points": ["price", "volume"]}}, {"name": "get_stock_data", "arguments": {"symbol": "GOOG", "data_points": ["price", "volume"]}}, {"name": "get_stock_data", "arguments": {"symbol": "MSFT", "data_points": ["price", "volume"]}}]`

### Qwen: team correct=False labels=['type_error', 'wrong_value']

- agent 1: `[{"name": "get_stock_data", "arguments": {"symbol": "AAPL", "data_points": ["price", "volume"]}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}]`
- agent 2: `[{"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "GOO", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}]`
- agent 3: `[{"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}]`

  team: `[{"name": "get_stock_data", "arguments": {"symbol": "AAPL", "data_points": ["price", "volume"]}}, {"name": "get_stock_data", "arguments": {"symbol": "GOO", "data_points": null}}, {"name": "get_stock_data", "arguments": {"symbol": "...", "data_points": null}}]`

## choose-N choose_35 (list, n=3)

user text of every agent:

> Get the current weather in any three of these cities: Boston, Dallas, Toronto, Austin, Chicago and Paris.

### Dream: cities ['Boston', 'Dallas', 'Chicago']

- agent 1: `[{"name": "get_weather", "arguments": {"city": "Boston"}}, {"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "..."}}]`
- agent 2: `[{"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "Dallas"}}, {"name": "get_weather", "arguments": {"city": "..."}}]`
- agent 3: `[{"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "Chicago"}}]`

### Qwen: cities ['Boston', '...', '...']

- agent 1: `[{"name": "get_weather", "arguments": {"city": "Boston"}}, {"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "..."}}]`
- agent 2: `[{"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "..."}}]`
- agent 3: `[{"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "..."}}]`

## choose-N choose_20 (open, n=2)

> Get the current weather in two different US cities of your choice.

### Dream: cities ['Chicago', 'Chicago']

- agent 1: `[{"name": "get_weather", "arguments": {"city": "Chicago"}}, {"name": "get_weather", "arguments": {"city": "..."}}]`
- agent 2: `[{"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "Chicago"}}]`

### Qwen: cities ['New', '...']

- agent 1: `[{"name": "get_weather", "arguments": {"city": "New"}}, {"name": "get_weather", "arguments": {"city": "..."}}]`
- agent 2: `[{"name": "get_weather", "arguments": {"city": "..."}}, {"name": "get_weather", "arguments": {"city": "..."}}]`

