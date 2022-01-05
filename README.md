# table-allocations
Algorithm for allocating tables based on preferences, using simulated annealing

## Prerequisities
- An installation of Go: https://go.dev/dl/

## Setup
- `go install github.com/mhbardsley/table-allocations`
- Create a JSON file to hold people, their preferences and table capacities. See `sample.json` as an example. (Note: the program will, by default, look for a `input.json` file)

## Running the program
- `table-allocations [flags]`
- Note: there may be a small delay between running the program and getting the result (see below for inspecting flags to change speed/optimisation trade-off)

All flags are optional and most do not need touching. If you have not named your JSON file `input.json`, you need to supply an `-f` flag, e.g. `table-allocations -f sample.json` will carry out the algorithm on the sample data.

Another useful flag is `-m`, which specifies what is being optimised. There are two options: `sum` (default), which will optimise the total number of preferences satisfied; `count`, which will optimise the number of people with at least 1 satisfied preference. To choose `count`, for example, use `table-allocations -m count`.

For all other flags (which don't really need tweaking), you can run with the `-h` flag, i.e. `table-allocations -h`.