package allocations

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	algo "github.com/mhbardsley/jubilant-octo-palm-tree"
)

// TestBenchHillClimbVsBareGA is not a regression test — it compares GA-only vs GA+hill-climb
// at various durations on sample.json. Skipped unless BENCH=1 is set in the environment.
// Run with: BENCH=1 go test -run TestBenchHillClimbVsBareGA -v ./allocations
func TestBenchHillClimbVsBareGA(t *testing.T) {
	if os.Getenv("BENCH") == "" {
		t.Skip("set BENCH=1 to run this bench")
	}
	raw, err := os.ReadFile("../sample.json")
	if err != nil {
		t.Fatal(err)
	}
	var prob Problem
	if err := json.Unmarshal(raw, &prob); err != nil {
		t.Fatal(err)
	}
	plusOnes := make(map[string]string, len(prob.PlusOnes))
	for _, p := range prob.PlusOnes {
		plusOnes[p.PersonOne] = p.PersonTwo
	}
	totalPrefs := 0
	for _, p := range prob.People {
		totalPrefs += len(p.Preferences)
	}
	score, err := scorer(ModeHybrid, plusOnes, len(prob.People), totalPrefs)
	if err != nil {
		t.Fatal(err)
	}

	cfg := func(d time.Duration) algo.Config[*assignment] {
		deadline := time.Now().Add(d)
		return algo.Config[*assignment]{
			PopulationSize:      500,
			GenerateIndividual:  generator(prob.People, prob.Tables, score),
			Crossover:           crossover(prob.Tables, score),
			ContinuingCondition: func() bool { return time.Now().Before(deadline) },
		}
	}
	count := func(a *assignment) int { c, _, _ := scoreParts(a.tables, plusOnes); return c }

	for _, d := range []time.Duration{1 * time.Second, 5 * time.Second, 15 * time.Second} {
		bare, memetic := 0, 0
		const trials = 3
		for range trials {
			bare += count(algo.RunGeneticAlgorithm(cfg(d)))

			c := cfg(d)
			c.Elitism = 1
			c.LocalSearch = localOptimize
			memetic += count(algo.RunGeneticAlgorithm(c))
		}
		fmt.Printf("duration=%-4s bare-GA avg=%.1f  memetic-GA avg=%.1f  delta=%.1f\n",
			d, float64(bare)/trials, float64(memetic)/trials,
			float64(memetic-bare)/trials)
	}
}
