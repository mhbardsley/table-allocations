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
	"sync/atomic"
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

	// Reserve a small slice of the budget for a final sum-only polish pass
	// (only meaningful for ModeHybrid). The hybrid score weights count so
	// heavily that once count saturates the gradient toward more prefs is
	// barely visible; a dedicated post-GA sweep squeezes out the remaining
	// sum at fixed count. Skipped on short runs (<10s) — taking time away
	// from the GA hurts more than the polish gains at small budgets, and
	// converges in under a second on 200 people anyway.
	polishBudget := time.Duration(0)
	if mode == ModeHybrid && runtime >= 10*time.Second {
		polishBudget = 2 * time.Second
	}
	gaRuntime := runtime - polishBudget

	// Multistart: split the budget into N epochs, run a fresh GA per epoch,
	// take the best across epochs. A single GA tends to get stuck in whatever
	// basin its initial population landed in; restarting gets us a fresh draw
	// at a different basin, and the variance between draws is real enough to
	// gain measurable quality on large inputs.
	const epochs = 4
	epochBudget := gaRuntime / epochs

	mkCfg := func(deadline time.Time) algo.Config[*assignment] {
		polish := func(a *assignment) { localOptimizeUntil(a, deadline) }
		cfg := algo.Config[*assignment]{
			PopulationSize:      pop,
			GenerateIndividual:  generator(prob.People, prob.Tables, score),
			Crossover:           crossover(prob.Tables, score),
			ContinuingCondition: func() bool { return time.Now().Before(deadline) },
			Elitism:             1,
			// Polish the elite once per generation; concentrating local search
			// on the single best individual keeps generations cheap enough that
			// the GA still evolves at scale.
			EliteLocalSearch: polish,
		}
		// On small problems, per-pair scoring is fast enough that running local
		// search on every child too is a clear win and matches the original
		// memetic GA behaviour for the sample-sized inputs the project was
		// originally tuned against.
		if len(prob.People) <= 50 {
			cfg.LocalSearch = polish
		}
		return cfg
	}

	var best *assignment
	for e := 0; e < epochs; e++ {
		epochDeadline := time.Now().Add(epochBudget)
		candidate := algo.RunGeneticAlgorithm(mkCfg(epochDeadline))
		if best == nil || candidate.Fitness() > best.Fitness() {
			best = candidate
		}
	}

	// Squeeze remaining sum out of the elite without disturbing count or
	// plus-one constraints. No-op for ModeCount/ModeSum (count not the
	// dominant term, or sum already maximised by the GA).
	if polishBudget > 0 {
		polishDeadline := time.Now().Add(polishBudget)
		polishSumPreservingCount(best, plusOnes, polishDeadline)
	}

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

// Mutate is a mixture of small and large perturbations. A single swap is fine
// for fine-tuning a near-optimal arrangement, but on larger inputs the GA
// trends to local optima the hill-climb can't escape (every single-swap
// neighbour is worse, even though some 3- or 5-swap neighbour is much
// better). Mixing in occasional bigger moves — multi-swap chains and a
// full reshuffle of two tables — gives crossover and mutation a fighting
// chance of escaping those basins.
func (a *assignment) Mutate() {
	switch r := rand.Float64(); {
	case r < 0.6:
		swapTwo(a.tables)
	case r < 0.85:
		swapTwo(a.tables)
		swapTwo(a.tables)
		swapTwo(a.tables)
	case r < 0.95:
		for i := 0; i < 5; i++ {
			swapTwo(a.tables)
		}
	default:
		reshuffleTwoTables(a.tables)
	}
}

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

// reshuffleTwoTables takes two random tables, pools their occupants, and
// redistributes them randomly. Used as a high-magnitude mutation to escape
// the local optima that single-swap mutation can't break out of.
func reshuffleTwoTables(ts []table) {
	if len(ts) < 2 {
		return
	}
	i := rand.IntN(len(ts))
	j := rand.IntN(len(ts) - 1)
	if j >= i {
		j++
	}
	a, b := &ts[i], &ts[j]
	pool := make([]Person, 0, a.capacity+b.capacity)
	pool = append(pool, a.people...)
	pool = append(pool, b.people...)
	rand.Shuffle(len(pool), func(x, y int) { pool[x], pool[y] = pool[y], pool[x] })

	a.members = make(map[string]struct{}, a.capacity)
	for k := range a.capacity {
		a.people[k] = pool[k]
		a.members[pool[k].Name] = struct{}{}
	}
	b.members = make(map[string]struct{}, b.capacity)
	for k := range b.capacity {
		b.people[k] = pool[a.capacity+k]
		b.members[pool[a.capacity+k].Name] = struct{}{}
	}
}

// localOptimize performs greedy pairwise-swap hill-climbing until no swap improves fitness.
// The GA explores; this exploits — it polishes the best individual to a local optimum.
func localOptimize(a *assignment) {
	localOptimizeUntil(a, time.Time{})
}

// polishSumPreservingCount squeezes more preferences out of an already-good
// assignment without dropping the satisfied-people count. Hybrid mode
// dominates with weight*count; once count is at its local maximum the
// gradient toward more sum is faint and the GA largely stops investing in
// it. This pass goes pairwise across tables and accepts any swap that
// (a) doesn't introduce a plus-one penalty, (b) doesn't reduce satisfied
// count, and (c) increases the per-pref sum. Delta-scored on the two
// affected tables only — same speed envelope as localOptimizeUntil.
func polishSumPreservingCount(a *assignment, plusOnes map[string]string, deadline time.Time) {
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
				pair[0] = a.tables[i]
				pair[1] = a.tables[j]
				for pi := range a.tables[i].capacity {
					for pj := range a.tables[j].capacity {
						c0, s0, p0 := scoreParts(pair, plusOnes)
						swapAt(&a.tables[i], &a.tables[j], pi, pj)
						c1, s1, p1 := scoreParts(pair, plusOnes)
						if p1 <= p0 && c1 >= c0 && s1 > s0 {
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
	// Half the population is greedy-seeded (each person joins the table with
	// the most of their preferences already seated), half stays uniform-random
	// for diversity. The greedy half drops us into a much better starting
	// basin; the random half keeps the GA from being stuck around one
	// near-optimum the greedy heuristic misses.
	var n atomic.Int64
	return func() *assignment {
		seq := n.Add(1)
		var seated []Person
		if seq%2 == 0 {
			seated = greedyArrangement(people, capacities)
		} else {
			seated = slices.Clone(people)
			rand.Shuffle(len(seated), func(i, j int) {
				seated[i], seated[j] = seated[j], seated[i]
			})
		}
		return &assignment{tables: pack(seated, capacities), score: score}
	}
}

// greedyArrangement walks the (shuffled) people and places each at the
// not-yet-full table with the most of their preferences already seated. Ties
// broken in favour of less-full tables to keep capacities balanced as we go.
// Result is a permutation suitable for pack(). Not optimal, but a much
// better starting point than uniform-random for the GA.
func greedyArrangement(people []Person, capacities []int) []Person {
	order := slices.Clone(people)
	rand.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })

	buckets := make([][]Person, len(capacities))
	for i := range buckets {
		buckets[i] = make([]Person, 0, capacities[i])
	}
	placed := make(map[string]int, len(people))
	for _, p := range order {
		bestIdx, bestScore, bestSpace := -1, -1, -1
		for i, b := range buckets {
			if len(b) >= capacities[i] {
				continue
			}
			s := 0
			for _, pref := range p.Preferences {
				if at, ok := placed[pref]; ok && at == i {
					s++
				}
			}
			space := capacities[i] - len(b)
			if s > bestScore || (s == bestScore && space > bestSpace) {
				bestIdx, bestScore, bestSpace = i, s, space
			}
		}
		buckets[bestIdx] = append(buckets[bestIdx], p)
		placed[p.Name] = bestIdx
	}
	out := make([]Person, 0, len(people))
	for _, b := range buckets {
		out = append(out, b...)
	}
	return out
}

// crossover combines two parents into a fresh child. Half the time we use
// the table-aware variant — inherit some intact tables from parent A and
// fill the remaining seats from parent B's flat order — which preserves
// good cluster structure parents have built up. The other half uses
// classic order crossover (OX), which keeps a different kind of
// neighbour-order signal alive in the gene pool.
func crossover(capacities []int, score func([]table) float64) func(*assignment, *assignment) *assignment {
	total := 0
	for _, c := range capacities {
		total += c
	}
	ox := orderCrossover(capacities, score, total)
	tableAware := tableInheritCrossover(capacities, score, total)
	return func(a, b *assignment) *assignment {
		if rand.Float64() < 0.5 {
			return tableAware(a, b)
		}
		return ox(a, b)
	}
}

// orderCrossover is the original OX: copy a contiguous slice from parent A,
// then fill the remaining seats in parent B's order, skipping anyone already
// placed. Operates on a flat permutation; ignores table boundaries.
func orderCrossover(capacities []int, score func([]table) float64, total int) func(*assignment, *assignment) *assignment {
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

// tableInheritCrossover inherits a random subset of parent A's tables intact
// and fills the rest with parent B's people in walking order, skipping
// anyone already placed. This carries forward whole clusters parent A may
// have discovered, which is the kind of structure single-swap mutation can't
// build from scratch and OX easily destroys.
func tableInheritCrossover(capacities []int, score func([]table) float64, total int) func(*assignment, *assignment) *assignment {
	nTables := len(capacities)
	return func(a, b *assignment) *assignment {
		// Inherit between 1 and nTables-1 of A's tables; the count itself is
		// random per call so the GA explores at multiple "block sizes."
		inheritCount := 1 + rand.IntN(maxInt(1, nTables-1))
		// Pick which of A's tables to inherit by shuffling indices and taking
		// the first inheritCount.
		idx := make([]int, nTables)
		for i := range idx {
			idx[i] = i
		}
		rand.Shuffle(nTables, func(x, y int) { idx[x], idx[y] = idx[y], idx[x] })
		keep := make([]bool, nTables)
		for i := 0; i < inheritCount; i++ {
			keep[idx[i]] = true
		}

		// Allocate the flat child buffer + used set.
		child := make([]Person, total)
		used := make(map[string]struct{}, total)
		// Place inherited tables first into the same slots they occupied in A.
		// pack() walks tables in order, so the slot for table i starts at
		// sum(capacities[:i]).
		offset := 0
		for i := 0; i < nTables; i++ {
			if keep[i] {
				for k, p := range a.tables[i].people {
					child[offset+k] = p
					used[p.Name] = struct{}{}
				}
			}
			offset += capacities[i]
		}
		// Fill the remaining slots from B's flat walk, skipping anyone already
		// placed. start point chosen randomly so we don't always favour B's
		// table 0.
		flatB := flatten(b.tables, total)
		bStart := rand.IntN(total)
		offset = 0
		for i := 0; i < nTables; i++ {
			if keep[i] {
				offset += capacities[i]
				continue
			}
			for slot := 0; slot < capacities[i]; slot++ {
				for {
					p := flatB[bStart%total]
					bStart++
					if _, ok := used[p.Name]; ok {
						continue
					}
					child[offset+slot] = p
					used[p.Name] = struct{}{}
					break
				}
			}
			offset += capacities[i]
		}
		return &assignment{tables: pack(child, capacities), score: score}
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
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
