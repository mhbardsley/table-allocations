package allocations

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
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
		totalPrefs := 0
		for _, p := range prob.People {
			totalPrefs += len(p.Preferences)
		}
		for _, d := range []time.Duration{2 * time.Second, 10 * time.Second, 30 * time.Second} {
			start := time.Now()
			res, err := Allocate(prob, Options{Mode: ModeHybrid, Runtime: d})
			elapsed := time.Since(start)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Printf("uniform-random n=%-3d budget=%-4s elapsed=%-7s satisfied=%d/%d prefs=%d/%d\n",
				n, d, elapsed.Round(100*time.Millisecond),
				res.Stats.PeopleSatisfied, res.Stats.TotalPeople,
				res.Stats.PreferencesSatisfied, totalPrefs)
		}
	}
}

// TestBenchAllocateClustered exercises Allocate against problems whose
// preferences mostly come from inside a small "friend group" — closer to
// real form data than uniform random. With clustered preferences there's
// genuine structure to exploit; if the algorithm can't find it, we have
// a problem.
func TestBenchAllocateClustered(t *testing.T) {
	if os.Getenv("BENCH") == "" {
		t.Skip("set BENCH=1 to run this bench")
	}
	for _, cfg := range []struct {
		n, tableSize, clusterSize int
		intraRate                 float64
	}{
		{n: 100, tableSize: 10, clusterSize: 8, intraRate: 0.85},
		{n: 200, tableSize: 10, clusterSize: 8, intraRate: 0.85},
		{n: 200, tableSize: 10, clusterSize: 10, intraRate: 0.95},
	} {
		prob := generateClusteredScaleTestProblem(cfg.n, cfg.tableSize, cfg.clusterSize, cfg.intraRate)
		totalPrefs := 0
		for _, p := range prob.People {
			totalPrefs += len(p.Preferences)
		}
		for _, d := range []time.Duration{5 * time.Second, 30 * time.Second} {
			start := time.Now()
			res, err := Allocate(prob, Options{Mode: ModeHybrid, Runtime: d})
			elapsed := time.Since(start)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Printf("clustered n=%d cluster=%d intra=%.0f%% budget=%-4s elapsed=%-7s satisfied=%d/%d prefs=%d/%d\n",
				cfg.n, cfg.clusterSize, cfg.intraRate*100, d, elapsed.Round(100*time.Millisecond),
				res.Stats.PeopleSatisfied, res.Stats.TotalPeople,
				res.Stats.PreferencesSatisfied, totalPrefs)
		}
	}
}

// TestBenchVsSimulatedAnnealing compares the current Allocate (memetic GA)
// against a clean reproduction of the simulated-annealing solver that lived
// in main.go before the GA refactor (commit f59d9cc). Same problems, same
// runtime budget.
func TestBenchVsSimulatedAnnealing(t *testing.T) {
	if os.Getenv("BENCH") == "" {
		t.Skip("set BENCH=1 to run this bench")
	}
	type setup struct {
		name string
		prob Problem
	}
	setups := []setup{
		{"uniform-100", generateScaleTestProblem(100, 10)},
		{"uniform-200", generateScaleTestProblem(200, 10)},
		{"clustered-200-c10-i95", generateClusteredScaleTestProblem(200, 10, 10, 0.95)},
		{"clustered-200-c8-i85", generateClusteredScaleTestProblem(200, 10, 8, 0.85)},
	}
	for _, s := range setups {
		totalPrefs := 0
		for _, p := range s.prob.People {
			totalPrefs += len(p.Preferences)
		}
		for _, d := range []time.Duration{5 * time.Second, 30 * time.Second} {
			ga, err := Allocate(s.prob, Options{Mode: ModeHybrid, Runtime: d})
			if err != nil {
				t.Fatal(err)
			}
			sa := saAllocate(s.prob, d)
			fmt.Printf("%-25s budget=%-4s GA prefs=%d/%d  SA prefs=%d/%d  (Δ=%+d)\n",
				s.name, d, ga.Stats.PreferencesSatisfied, totalPrefs,
				sa.Stats.PreferencesSatisfied, totalPrefs,
				ga.Stats.PreferencesSatisfied-sa.Stats.PreferencesSatisfied)
		}
	}
}

// saAllocate is a clean parallel-tempering simulated-annealing solver against
// the same Problem/Result interface as Allocate. Reproduces the algorithm
// that lived in main.go before commit f59d9cc, but rebuilt against the
// current internals (delta-friendly score function, in-place swap, etc.).
// Used only by the benchmark — not exported.
func saAllocate(prob Problem, runtime time.Duration) Result {
	plusOnes := make(map[string]string, len(prob.PlusOnes))
	for _, p := range prob.PlusOnes {
		plusOnes[p.PersonOne] = p.PersonTwo
	}
	totalPrefs := 0
	for _, p := range prob.People {
		totalPrefs += len(p.Preferences)
	}
	score, _ := scorer(ModeHybrid, plusOnes, len(prob.People), totalPrefs)

	deadline := time.Now().Add(runtime)
	const replicas = 6
	const baseTemp = 1.0
	const coolingRate = 0.999
	type result struct {
		state []table
		cost  float64
	}
	results := make(chan result, replicas)
	for r := 0; r < replicas; r++ {
		tempScale := 1.0
		for i := 0; i < r; i++ {
			tempScale *= 2 // each replica is hotter than the last
		}
		go func(temp float64) {
			people := slices.Clone(prob.People)
			rand.Shuffle(len(people), func(i, j int) { people[i], people[j] = people[j], people[i] })
			state := pack(people, prob.Tables)
			cost := score(state)
			best := copyTables(state)
			bestCost := cost

			for time.Now().Before(deadline) {
				for k := 0; k < 1000; k++ {
					i := rand.IntN(len(state))
					j := rand.IntN(len(state) - 1)
					if j >= i {
						j++
					}
					a, b := &state[i], &state[j]
					pi, pj := rand.IntN(a.capacity), rand.IntN(b.capacity)
					before := score([]table{*a, *b})
					swapAt(a, b, pi, pj)
					after := score([]table{*a, *b})
					delta := after - before
					if delta >= 0 || rand.Float64() < expApprox(delta/temp) {
						cost += delta
						if cost > bestCost {
							best = copyTables(state)
							bestCost = cost
						}
					} else {
						swapAt(a, b, pi, pj)
					}
				}
				temp *= coolingRate
			}
			results <- result{best, bestCost}
		}(baseTemp * tempScale)
	}
	var best []table
	bestCost := -1e18
	for r := 0; r < replicas; r++ {
		got := <-results
		if got.cost > bestCost {
			best = got.state
			bestCost = got.cost
		}
	}
	count, sum, _ := scoreParts(best, plusOnes)
	totalSeated := 0
	out := make([]SeatedTable, len(best))
	for i, t := range best {
		names := make([]string, len(t.people))
		for j, p := range t.people {
			names[j] = p.Name
		}
		out[i] = SeatedTable{Capacity: t.capacity, People: names}
		totalSeated += len(t.people)
	}
	return Result{Tables: out, Stats: Stats{PeopleSatisfied: count, TotalPeople: totalSeated, PreferencesSatisfied: sum}}
}

func copyTables(src []table) []table {
	out := make([]table, len(src))
	for i, t := range src {
		out[i] = table{
			capacity: t.capacity,
			people:   slices.Clone(t.people),
			members:  make(map[string]struct{}, len(t.members)),
		}
		for k := range t.members {
			out[i].members[k] = struct{}{}
		}
	}
	return out
}

// expApprox approximates math.Exp for x < 0 using a series cutoff; SA only
// ever calls Exp on non-positive arguments and the precise tail doesn't
// matter for the accept/reject decision.
func expApprox(x float64) float64 {
	// Real math.Exp via the standard library — keeps intent obvious. The
	// "approx" name was a placeholder; benching showed math.Exp isn't a hot
	// spot here.
	return math.Exp(x)
}

// TestBenchVsPlantedOptimum runs Allocate against problems whose optimal
// score is known by construction — we lay out the people in cliques first,
// then build preferences that point inside (and partly outside) those
// cliques, so the per-person and per-pref maxima are arithmetic.
//
// Reports how close the GA gets to those known maxima at scale.
func TestBenchVsPlantedOptimum(t *testing.T) {
	if os.Getenv("BENCH") == "" {
		t.Skip("set BENCH=1 to run this bench")
	}
	for _, c := range []struct {
		name      string
		n, table  int
		intraOnly bool // true = all prefs intra-clique (optimum = 100%)
	}{
		{"clique-perfect-100", 100, 10, true},
		{"clique-perfect-200", 200, 10, true},
		{"clique-spill-100", 100, 10, false},
		{"clique-spill-200", 200, 10, false},
	} {
		prob, optPeople, optPrefs := generatePlantedProblem(c.n, c.table, c.intraOnly)
		totalPrefs := 0
		for _, p := range prob.People {
			totalPrefs += len(p.Preferences)
		}
		for _, d := range []time.Duration{5 * time.Second, 30 * time.Second} {
			res, err := Allocate(prob, Options{Mode: ModeHybrid, Runtime: d})
			if err != nil {
				t.Fatal(err)
			}
			fmt.Printf("%-20s budget=%-4s  GA people=%d/%d (opt=%d, gap=%d)  GA prefs=%d/%d (opt=%d, gap=%d, %.1f%% of opt)\n",
				c.name, d,
				res.Stats.PeopleSatisfied, c.n, optPeople, optPeople-res.Stats.PeopleSatisfied,
				res.Stats.PreferencesSatisfied, totalPrefs, optPrefs, optPrefs-res.Stats.PreferencesSatisfied,
				100*float64(res.Stats.PreferencesSatisfied)/float64(optPrefs))
		}
	}
}

// generatePlantedProblem lays out n people in n/table cliques of size `table`.
// If intraOnly, every preference points inside the clique → optimum is
// trivially every clique-as-table, with 100% people and 100% prefs satisfied.
// Otherwise, 2 of every 3 prefs point inside the clique, 1 to an adjacent
// clique — the clique-as-table partition still satisfies every person and
// 2/3 of every preference.
func generatePlantedProblem(n, tableSize int, intraOnly bool) (Problem, int, int) {
	r := rand.New(rand.NewPCG(123, 456))
	people := make([]Person, n)
	for i := range people {
		people[i] = Person{Name: fmt.Sprintf("P%03d", i)}
	}
	cliqueOf := func(i int) int { return i / tableSize }
	nCliques := n / tableSize
	for i := range people {
		ci := cliqueOf(i)
		// Intra picks: 3 if intraOnly, otherwise 2.
		intra := 3
		if !intraOnly {
			intra = 2
		}
		picked := map[int]bool{i: true}
		// Pick `intra` distinct other people from the same clique.
		for k := 0; k < intra; k++ {
			for {
				j := ci*tableSize + r.IntN(tableSize)
				if !picked[j] {
					picked[j] = true
					people[i].Preferences = append(people[i].Preferences, people[j].Name)
					break
				}
			}
		}
		if !intraOnly {
			// One pick from an adjacent clique (wraps).
			adj := (ci + 1) % nCliques
			j := adj*tableSize + r.IntN(tableSize)
			people[i].Preferences = append(people[i].Preferences, people[j].Name)
		}
	}
	tables := make([]int, nCliques)
	for i := range tables {
		tables[i] = tableSize
	}
	// Optima: under the obvious clique-as-table partition,
	//   intraOnly: every person has 3/3 prefs satisfied → optPeople=n, optPrefs=3n
	//   spillover: every person has 2/3 prefs satisfied → optPeople=n, optPrefs=2n
	optPeople := n
	optPrefs := 3 * n
	if !intraOnly {
		optPrefs = 2 * n
	}
	return Problem{People: people, Tables: tables}, optPeople, optPrefs
}

// generateClusteredScaleTestProblem mimics real form input where attendees
// mostly want to sit with people from their friend group. People are split
// into clusters of `clusterSize`; each preference picks from the same cluster
// with probability `intraRate`, otherwise from the rest of the pool.
func generateClusteredScaleTestProblem(n, tableSize, clusterSize int, intraRate float64) Problem {
	r := rand.New(rand.NewPCG(42, 0))
	people := make([]Person, n)
	for i := range people {
		people[i] = Person{Name: fmt.Sprintf("P%03d", i)}
	}
	clusterOf := func(idx int) (lo, hi int) {
		lo = (idx / clusterSize) * clusterSize
		hi = lo + clusterSize
		if hi > n {
			hi = n
		}
		return
	}
	for i := range people {
		for k := 0; k < 3; k++ {
			var j int
			if r.Float64() < intraRate {
				lo, hi := clusterOf(i)
				if hi-lo > 1 {
					j = lo + r.IntN(hi-lo-1)
					if j >= i {
						j++
					}
				} else {
					j = r.IntN(n - 1)
					if j >= i {
						j++
					}
				}
			} else {
				j = r.IntN(n - 1)
				if j >= i {
					j++
				}
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
