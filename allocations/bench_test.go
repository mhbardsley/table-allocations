package allocations

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
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

// TestBenchAllocateAtScale exercises the public Allocate against three sizes
// and reports peopleSatisfied/total. Comparison is against the algorithm's
// own previous behavior — we want to see the larger sizes actually return
// useful results in a reasonable runtime, not be dominated by per-child
// hill-climb. Skipped unless BENCH=1 is set.
func TestBenchAllocateAtScale(t *testing.T) {
	if os.Getenv("BENCH") == "" {
		t.Skip("set BENCH=1 to run this bench")
	}
	for _, n := range []int{25, 100, 200} {
		prob := generateScaleTestProblem(n, 10)
		for _, d := range []time.Duration{2 * time.Second, 10 * time.Second, 30 * time.Second} {
			start := time.Now()
			res, err := Allocate(prob, Options{Mode: ModeHybrid, Runtime: d})
			elapsed := time.Since(start)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Printf("n=%-3d budget=%-4s elapsed=%-7s satisfied=%d/%d prefs=%d\n",
				n, d, elapsed.Round(100*time.Millisecond),
				res.Stats.PeopleSatisfied, res.Stats.TotalPeople,
				res.Stats.PreferencesSatisfied)
		}
	}
}

// generateScaleTestProblem builds a deterministic problem with `n` people
// across tables of `tableSize` (last table absorbs the remainder if needed).
// Each person picks 3 random preferences from the rest of the pool, which
// approximates real form input — most prefs land on someone the GA has to
// shuffle around to satisfy.
func generateScaleTestProblem(n, tableSize int) Problem {
	r := rand.New(rand.NewPCG(42, 0))
	people := make([]Person, n)
	for i := range people {
		people[i] = Person{Name: fmt.Sprintf("P%03d", i)}
	}
	for i := range people {
		for k := 0; k < 3; k++ {
			j := r.IntN(n - 1)
			if j >= i {
				j++
			}
			people[i].Preferences = append(people[i].Preferences, people[j].Name)
		}
	}
	tables := []int{}
	remaining := n
	for remaining > tableSize {
		tables = append(tables, tableSize)
		remaining -= tableSize
	}
	if remaining > 0 {
		tables = append(tables, remaining)
	}
	return Problem{People: people, Tables: tables}
}
