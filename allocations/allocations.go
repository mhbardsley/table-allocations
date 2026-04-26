// Package allocations runs a memetic genetic algorithm to assign people to tables
// such that as many as possible sit with at least one of their preferred companions.
//
// The exported surface is intentionally small: a Problem in, a Result out.
// Internals (mutation, crossover, hill-climb) stay unexported so the library can
// evolve without breaking callers.
package allocations

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"time"

	algo "github.com/mhbardsley/jubilant-octo-palm-tree"
)

// Person is one attendee with their (zero or more) preferred companions.
// Preferences reference other Person.Name values; unknown names are silently
// ignored at scoring time, so partial data is harmless.
type Person struct {
	Name        string   `json:"name"`
	Preferences []string `json:"preferences"`
}

// PlusOne is a hard constraint: PersonOne and PersonTwo MUST sit at the same table.
// Solutions that violate any plus-one rank below every non-violating solution.
type PlusOne struct {
	PersonOne string `json:"personOne"`
	PersonTwo string `json:"personTwo"`
}

// Problem is the input to Allocate. Sum(Tables) must equal len(People).
type Problem struct {
	People   []Person  `json:"people"`
	Tables   []int     `json:"tables"`
	PlusOnes []PlusOne `json:"plusOnes"`
}

// Mode picks the fitness function. "hybrid" prioritises CountSatisfied; "sum" optimises
// total preferences satisfied; "count" optimises only the number of people who get any
// preference at all.
type Mode string

const (
	ModeHybrid Mode = "hybrid"
	ModeSum    Mode = "sum"
	ModeCount  Mode = "count"
)

// Options tunes the search. Zero values are sensible defaults.
type Options struct {
	Mode           Mode          // default ModeHybrid
	PopulationSize int           // default 500
	Runtime        time.Duration // default 5s
}

// SeatedTable is one table in the final plan, in seating order.
type SeatedTable struct {
	Capacity int      `json:"capacity"`
	People   []string `json:"people"`
}

// Stats summarises how good the returned plan is.
type Stats struct {
	PeopleSatisfied      int `json:"peopleSatisfied"`
	TotalPeople          int `json:"totalPeople"`
	PreferencesSatisfied int `json:"preferencesSatisfied"`
}

// Result is the output of Allocate.
type Result struct {
	Tables []SeatedTable `json:"tables"`
	Stats  Stats         `json:"stats"`
}

// Allocate runs the genetic algorithm and returns the best plan found before
// Options.Runtime elapses. Returns a non-nil error only for invalid input
// (sum mismatch, duplicate names, unknown plus-one references, etc.).
func Allocate(prob Problem, opts Options) (Result, error) {
	if err := validate(prob); err != nil {
		return Result{}, err
	}
	mode := opts.Mode
	if mode == "" {
		mode = ModeHybrid
	}
	pop := opts.PopulationSize
	if pop <= 0 {
		pop = 500
	}
	runtime := opts.Runtime
	if runtime <= 0 {
		runtime = 5 * time.Second
	}

	plusOnes := make(map[string]string, len(prob.PlusOnes))
	for _, p := range prob.PlusOnes {
		plusOnes[p.PersonOne] = p.PersonTwo
	}

	totalPrefs := 0
	for _, p := range prob.People {
		totalPrefs += len(p.Preferences)
	}

	score, err := scorer(mode, plusOnes, len(prob.People), totalPrefs)
	if err != nil {
		return Result{}, err
	}

	deadline := time.Now().Add(runtime)
	polish := func(a *assignment) {
		localOptimizeUntil(a, deadline)
	}
	cfg := algo.Config[*assignment]{
		PopulationSize:      pop,
		GenerateIndividual:  generator(prob.People, prob.Tables, score),
		Crossover:           crossover(prob.Tables, score),
		ContinuingCondition: func() bool { return time.Now().Before(deadline) },
		Elitism:             1,
		// Polish the elite once per generation. At any scale the elite is the
		// individual most worth refining, and concentrating local search there
		// keeps generations cheap so the GA actually evolves.
		EliteLocalSearch: polish,
	}
	// On small problems, the per-pair scoring is fast enough that running local
	// search on every child too is a clear win. On larger problems, per-child
	// polish dominates the runtime budget so completely that the GA fails to
	// evolve a single generation; the elite-only path above is enough there.
	// The threshold is conservative — at 50 people / 10-seat tables a single
	// hill-climb pass runs in a couple of milliseconds.
	if len(prob.People) <= 50 {
		cfg.LocalSearch = polish
	}
	best := algo.RunGeneticAlgorithm(cfg)

	count, sum, _ := scoreParts(best.tables, plusOnes)
	totalSeated := 0
	out := make([]SeatedTable, len(best.tables))
	for i, t := range best.tables {
		names := make([]string, len(t.people))
		for j, p := range t.people {
			names[j] = p.Name
		}
		out[i] = SeatedTable{Capacity: t.capacity, People: names}
		totalSeated += len(t.people)
	}
	return Result{
		Tables: out,
		Stats:  Stats{PeopleSatisfied: count, TotalPeople: totalSeated, PreferencesSatisfied: sum},
	}, nil
}

// ---- internals ----

// table holds the seated people plus a name set for O(1) "is X at this table?" lookups.
type table struct {
	capacity int
	people   []Person
	members  map[string]struct{}
}

// assignment is a candidate seating plan; it implements algo.Individual.
type assignment struct {
	tables []table
	score  func([]table) float64
}

func (a *assignment) Fitness() float64 { return a.score(a.tables) }

func (a *assignment) Mutate() { swapTwo(a.tables) }

// swapAt swaps the people at positions pa, pb across tables a, b, keeping member sets in sync.
func swapAt(a, b *table, pa, pb int) {
	delete(a.members, a.people[pa].Name)
	delete(b.members, b.people[pb].Name)
	a.people[pa], b.people[pb] = b.people[pb], a.people[pa]
	a.members[a.people[pa].Name] = struct{}{}
	b.members[b.people[pb].Name] = struct{}{}
}

// swapTwo swaps a randomly-chosen pair of people across two distinct tables.
func swapTwo(ts []table) {
	if len(ts) < 2 {
		return
	}
	i := rand.IntN(len(ts))
	j := rand.IntN(len(ts) - 1)
	if j >= i {
		j++
	}
	swapAt(&ts[i], &ts[j], rand.IntN(ts[i].capacity), rand.IntN(ts[j].capacity))
}

// localOptimize performs greedy pairwise-swap hill-climbing until no swap improves fitness.
// The GA explores; this exploits — it polishes the best individual to a local optimum.
func localOptimize(a *assignment) {
	localOptimizeUntil(a, time.Time{})
}

// localOptimizeUntil is localOptimize with a hard wall-clock budget. A zero
// deadline disables the budget (equivalent to localOptimize). The deadline is
// checked at the table-pair level, frequent enough to keep the bail latency
// well under a second on large inputs and rare enough that the time.Now()
// overhead doesn't dominate the inner swap loop.
//
// Uses delta scoring: a swap involving tables i and j only affects the score
// contribution of tables i and j (every other table's people, preferences,
// and plus-one status are untouched). Re-scoring just those two instead of
// the whole plan turns the per-swap cost from O(N people × P prefs) into
// O(2 cap × P prefs), an ~N/(2 cap)× speedup that keeps the elite polish
// affordable at hundreds of attendees.
//
// (The "early-return on penalty" branch in the score function makes delta
// scoring approximate when the global penalty count is non-zero AND the swap
// doesn't touch a violating pair: we may apply or skip neutral swaps that
// don't actually change global fitness. Harmless — global fitness is
// unchanged either way — and the elite, which is what we polish, almost
// always has zero violations because penalties dominate every other term.)
func localOptimizeUntil(a *assignment, deadline time.Time) {
	hasDeadline := !deadline.IsZero()
	pair := make([]table, 2)
	for {
		if hasDeadline && time.Now().After(deadline) {
			return
		}
		improved := false
		for i := range a.tables {
			if hasDeadline && time.Now().After(deadline) {
				return
			}
			for j := i + 1; j < len(a.tables); j++ {
				// table.people / table.members are reference types, so this
				// aliases the live tables; no need to refresh after each swap.
				pair[0] = a.tables[i]
				pair[1] = a.tables[j]
				for pi := range a.tables[i].capacity {
					for pj := range a.tables[j].capacity {
						before := a.score(pair)
						swapAt(&a.tables[i], &a.tables[j], pi, pj)
						if a.score(pair) > before {
							improved = true
						} else {
							swapAt(&a.tables[i], &a.tables[j], pi, pj)
						}
					}
				}
			}
		}
		if !improved {
			return
		}
	}
}

// scoreParts returns (people with ≥1 satisfied preference, total preferences satisfied, plus-one violations).
func scoreParts(ts []table, plusOnes map[string]string) (count, sum, penalties int) {
	for _, t := range ts {
		for _, p := range t.people {
			if pair, ok := plusOnes[p.Name]; ok {
				if _, together := t.members[pair]; !together {
					penalties++
				}
			}
			satisfied := false
			for _, pref := range p.Preferences {
				if _, ok := t.members[pref]; ok {
					sum++
					satisfied = true
				}
			}
			if satisfied {
				count++
			}
		}
	}
	return
}

// scorer returns the fitness function for the chosen mode. Plus-one violations always dominate
// (a single violation beats any non-violating solution by going negative).
func scorer(mode Mode, plusOnes map[string]string, totalPeople, totalPrefs int) (func([]table) float64, error) {
	switch mode {
	case ModeSum:
		return func(ts []table) float64 {
			_, sum, pen := scoreParts(ts, plusOnes)
			if pen > 0 {
				return -float64(pen)
			}
			return float64(sum)
		}, nil
	case ModeCount:
		return func(ts []table) float64 {
			count, _, pen := scoreParts(ts, plusOnes)
			if pen > 0 {
				return -float64(pen)
			}
			return float64(count)
		}, nil
	case ModeHybrid:
		// Weight count high enough that any improvement in count outranks any sum tradeoff.
		weight := float64(max(totalPeople, totalPrefs))
		return func(ts []table) float64 {
			count, sum, pen := scoreParts(ts, plusOnes)
			if pen > 0 {
				return -float64(pen)
			}
			return float64(count)*weight + float64(sum)
		}, nil
	default:
		return nil, fmt.Errorf("unknown mode %q (want sum, count, or hybrid)", mode)
	}
}

// pack distributes a flat slice of people into tables of the given capacities.
func pack(people []Person, capacities []int) []table {
	out := make([]table, len(capacities))
	pos := 0
	for i, c := range capacities {
		seated := slices.Clone(people[pos : pos+c])
		members := make(map[string]struct{}, c)
		for _, p := range seated {
			members[p.Name] = struct{}{}
		}
		out[i] = table{capacity: c, people: seated, members: members}
		pos += c
	}
	return out
}

func generator(people []Person, capacities []int, score func([]table) float64) func() *assignment {
	return func() *assignment {
		shuffled := slices.Clone(people)
		rand.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})
		return &assignment{tables: pack(shuffled, capacities), score: score}
	}
}

// crossover applies order crossover (OX) so the child is a valid permutation of all attendees:
// copy a contiguous slice from parent A, then fill the remaining seats in parent B's order.
func crossover(capacities []int, score func([]table) float64) func(*assignment, *assignment) *assignment {
	total := 0
	for _, c := range capacities {
		total += c
	}
	return func(a, b *assignment) *assignment {
		flatA := flatten(a.tables, total)
		flatB := flatten(b.tables, total)

		i, j := rand.IntN(total), rand.IntN(total)
		if i > j {
			i, j = j, i
		}

		child := make([]Person, total)
		used := make(map[string]struct{}, total)
		for k := i; k < j; k++ {
			child[k] = flatA[k]
			used[flatA[k].Name] = struct{}{}
		}
		pos := j
		for k := 0; k < total; k++ {
			p := flatB[(j+k)%total]
			if _, taken := used[p.Name]; taken {
				continue
			}
			child[pos%total] = p
			pos++
		}
		return &assignment{tables: pack(child, capacities), score: score}
	}
}

func flatten(ts []table, total int) []Person {
	out := make([]Person, 0, total)
	for _, t := range ts {
		out = append(out, t.people...)
	}
	return out
}

// validate checks that the problem is internally consistent before the GA runs.
func validate(prob Problem) error {
	if len(prob.People) == 0 {
		return errors.New("no people supplied")
	}
	if len(prob.Tables) == 0 {
		return errors.New("no tables supplied")
	}
	totalSeats := 0
	for i, c := range prob.Tables {
		if c <= 0 {
			return fmt.Errorf("table %d has non-positive capacity %d", i, c)
		}
		totalSeats += c
	}
	if totalSeats != len(prob.People) {
		return fmt.Errorf("seat/people mismatch: %d people, %d seats across %d tables",
			len(prob.People), totalSeats, len(prob.Tables))
	}
	names := make(map[string]struct{}, len(prob.People))
	for _, p := range prob.People {
		if p.Name == "" {
			return errors.New("person with empty name")
		}
		if _, dup := names[p.Name]; dup {
			return fmt.Errorf("duplicate person name %q", p.Name)
		}
		names[p.Name] = struct{}{}
	}
	for i, po := range prob.PlusOnes {
		if _, ok := names[po.PersonOne]; !ok {
			return fmt.Errorf("plusOne %d: %q not in people", i, po.PersonOne)
		}
		if _, ok := names[po.PersonTwo]; !ok {
			return fmt.Errorf("plusOne %d: %q not in people", i, po.PersonTwo)
		}
	}
	return nil
}
