// Command table-allocations: read a JSON seating problem and print an optimised seating plan.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/mhbardsley/table-allocations/allocations"
)

func main() {
	mode := flag.String("m", "hybrid", "What to optimise: sum | count | hybrid")
	file := flag.String("f", "input.json", "Path to the JSON input file")
	population := flag.Int("p", 500, "Population size for the genetic algorithm")
	runtime := flag.Duration("d", 5*time.Second, "How long to run the genetic algorithm for")
	flag.Parse()

	raw, err := os.ReadFile(*file)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}

	var prob allocations.Problem
	if err := json.Unmarshal(raw, &prob); err != nil {
		log.Fatalf("error parsing input: %v", err)
	}

	result, err := allocations.Allocate(prob, allocations.Options{
		Mode:           allocations.Mode(*mode),
		PopulationSize: *population,
		Runtime:        *runtime,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Found a solution where %d people are given a preference (i.e. %d people have not been allocated at least one of their preferences). %d preferences are given in total.\n",
		result.Stats.PeopleSatisfied,
		result.Stats.TotalPeople-result.Stats.PeopleSatisfied,
		result.Stats.PreferencesSatisfied)
	for i, t := range result.Tables {
		fmt.Printf("\nTable %d (capacity %d)\n", i, t.Capacity)
		for _, name := range t.People {
			fmt.Printf("- %s\n", name)
		}
	}
}
