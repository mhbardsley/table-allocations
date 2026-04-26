// Package allocations assigns people to tables so that as many of their
// preferred companions sit next to them as possible.
//
// The algorithm is Large Neighborhood Search: start from a greedy seed,
// then loop — destroy a random fraction of the seating, rebuild it greedily,
// hill-climb the result, accept if it scores better. Replaces an earlier
// memetic genetic algorithm; benches against planted-optimum problems show
// LNS reaches the global optimum on cleanly-clustered inputs at 200 people
// where the GA was leaving 20–30% on the table.
package allocations

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"time"
)

// Person is one attendee with their (zero or more) preferred companions.
// Preferences reference other Person.Name values; unknown names are silently
// ignored at scoring time, so partial data is harmless.
type Person struct {
	Name        string   `json:"name"`
	Preferences []string `json:"preferences"`
}

// PlusOne is a hard constraint: PersonOne and PersonTwo MUST sit at the same
// table. Solutions that violate any plus-one rank below every non-violating
// solution.
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

// Mode picks the fitness function. "hybrid" prioritises CountSatisfied;
// "sum" optimises total preferences satisfied; "count" optimises only the
// number of people who get any preference at all.
type Mode string

const (
	ModeHybrid Mode = "hybrid"
	ModeSum    Mode = "sum"
	ModeCount  Mode = "count"
)

// Options tunes the search. Zero values are sensible defaults.
type Options struct {
	Mode    Mode          // default ModeHybrid
	Runtime time.Duration // default 5s

	// PopulationSize is no longer used (the algorithm is now LNS rather than
	// a population-based GA). Kept on the struct so callers compiled against
	// the old API don't break.
	PopulationSize int
}

// SeatedTable is one table in the final plan.
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

// Allocate searches for a good seating plan and returns the best one found
// before Options.Runtime elapses. Returns a non-nil error only for invalid
// input (seat/people mismatch, duplicate names, unknown plus-one references).
func Allocate(prob Problem, opts Options) (Result, error) {
	if err := validate(prob); err != nil {
		return Result{}, err
	}
	mode := opts.Mode
	if mode == "" {
		mode = ModeHybrid
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

	// Initial state: greedy seeding gets us into a much better basin than a
	// uniform-random start.
	bestState := pack(greedyArrangement(prob.People, prob.Tables), prob.Tables)
	bestScore := score(bestState)

	// Large Neighborhood Search loop. Each iteration: clone the current best,
	// pull a random fraction of people out of their tables, reinsert them
	// greedily into whichever table holds most of their preferences, polish
	// with hill-climbing, accept if the new arrangement scores higher.
	// Different ruin fractions on each iteration mean we explore both small
	// perturbations and large rebuilds.
	for time.Now().Before(deadline) {
		cand := cloneTables(bestState)
		removed := ruin(cand, 0.2+rand.Float64()*0.3) // 20–50% of seats
		recreate(cand, removed)
		localOptimizeUntil(cand, score, deadline)
		if s := score(cand); s > bestScore {
			bestState = cloneTables(cand)
			bestScore = s
		}
	}

	count, sum, _ := scoreParts(bestState, plusOnes)
	totalSeated := 0
	out := make([]SeatedTable, len(bestState))
	for i, t := range bestState {
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

// table holds the seated people plus a name set for O(1) membership lookups.
type table struct {
	capacity int
	people   []Person
	members  map[string]struct{}
}

// swapAt swaps the people at positions pa, pb across tables a, b, keeping
// the member sets in sync.
func swapAt(a, b *table, pa, pb int) {
	delete(a.members, a.people[pa].Name)
	delete(b.members, b.people[pb].Name)
	a.people[pa], b.people[pb] = b.people[pb], a.people[pa]
	a.members[a.people[pa].Name] = struct{}{}
	b.members[b.people[pb].Name] = struct{}{}
}

// scoreParts returns (people with ≥1 satisfied preference, total preferences
// satisfied, plus-one violations).
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

// scorer returns the fitness function for the chosen mode. Plus-one
// violations always dominate (a single violation beats any non-violating
// solution by going negative).
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
		// Weight count high enough that any improvement in count outranks any
		// sum tradeoff.
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

// greedyArrangement walks the (shuffled) people and places each at the
// not-yet-full table with the most of their preferences already seated. Ties
// broken in favour of the table with the most empty seats so capacities stay
// balanced as we go. Result is a permutation suitable for pack(). Not
// optimal, but a much better starting point than uniform-random.
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

// ---- LNS primitives ----

// ruin pulls a random `fraction` of people out of their tables and returns
// them. The vacated slots are filled later by recreate().
func ruin(tables []table, fraction float64) []Person {
	all := make([]Person, 0)
	for _, t := range tables {
		all = append(all, t.people...)
	}
	rand.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
	k := int(float64(len(all)) * fraction)
	if k < 1 {
		k = 1
	}
	removed := all[:k]
	removedSet := make(map[string]bool, k)
	for _, p := range removed {
		removedSet[p.Name] = true
	}
	for ti := range tables {
		kept := tables[ti].people[:0]
		for _, p := range tables[ti].people {
			if !removedSet[p.Name] {
				kept = append(kept, p)
			}
		}
		tables[ti].people = kept
		members := make(map[string]struct{}, tables[ti].capacity)
		for _, p := range kept {
			members[p.Name] = struct{}{}
		}
		tables[ti].members = members
	}
	return slices.Clone(removed)
}

// recreate places each removed person at the table with the most of their
// preferences already seated, with capacity remaining. Tie-broken in favour
// of more empty seats so load stays balanced.
func recreate(tables []table, removed []Person) {
	rand.Shuffle(len(removed), func(i, j int) { removed[i], removed[j] = removed[j], removed[i] })
	for _, p := range removed {
		bestIdx, bestScore, bestSpace := -1, -1, -1
		for i := range tables {
			free := tables[i].capacity - len(tables[i].people)
			if free <= 0 {
				continue
			}
			s := 0
			for _, pref := range p.Preferences {
				if _, ok := tables[i].members[pref]; ok {
					s++
				}
			}
			if s > bestScore || (s == bestScore && free > bestSpace) {
				bestIdx, bestScore, bestSpace = i, s, free
			}
		}
		tables[bestIdx].people = append(tables[bestIdx].people, p)
		tables[bestIdx].members[p.Name] = struct{}{}
	}
}

// cloneTables deep-copies a seating arrangement. We need this so we can hold
// onto the current best while a candidate is being shaken up.
func cloneTables(src []table) []table {
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

// ---- local search ----

// localOptimize performs greedy pairwise-swap hill-climbing until no swap
// improves fitness. Used by the LNS loop to polish each candidate.
func localOptimize(tables []table, score func([]table) float64) {
	localOptimizeUntil(tables, score, time.Time{})
}

// localOptimizeUntil is localOptimize with a hard wall-clock budget. A zero
// deadline disables the budget. Uses delta scoring: a swap involving tables
// i and j only affects the score contribution of those two tables, so we
// re-score just that pair instead of the whole plan — turns the per-swap
// cost from O(N people × P prefs) into O(2 cap × P prefs).
func localOptimizeUntil(tables []table, score func([]table) float64, deadline time.Time) {
	hasDeadline := !deadline.IsZero()
	pair := make([]table, 2)
	for {
		if hasDeadline && time.Now().After(deadline) {
			return
		}
		improved := false
		for i := range tables {
			if hasDeadline && time.Now().After(deadline) {
				return
			}
			for j := i + 1; j < len(tables); j++ {
				// table.people / table.members are reference types, so this
				// aliases the live tables; no need to refresh after each swap.
				pair[0] = tables[i]
				pair[1] = tables[j]
				for pi := range tables[i].capacity {
					for pj := range tables[j].capacity {
						before := score(pair)
						swapAt(&tables[i], &tables[j], pi, pj)
						if score(pair) > before {
							improved = true
						} else {
							swapAt(&tables[i], &tables[j], pi, pj)
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

// validate checks the problem is internally consistent before search runs.
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
