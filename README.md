# table-allocations
Allocate people to tables based on their seating preferences, using a genetic algorithm
([jubilant-octo-palm-tree](https://github.com/mhbardsley/jubilant-octo-palm-tree)).

## Prerequisites
- An installation of Go: https://go.dev/dl/

## Setup
- `go install github.com/mhbardsley/table-allocations@latest`
- Create a JSON file holding people, their preferences and table capacities. See `sample.json` as an example. By default the program looks for `input.json`.

## Running the program
- `table-allocations [flags]`

All flags are optional. If your input file is not named `input.json`, supply `-f`, e.g. `table-allocations -f sample.json`.

The `-m` flag picks what to optimise:
- `sum` — total number of preferences satisfied
- `count` — number of people with at least one satisfied preference
- `hybrid` (default) — prioritises `count`, breaking ties on `sum`

The `-d` flag sets the runtime (default `5s`); longer runs find better solutions. Use `-h` to see all flags.