// Allocate people to tables to maximise satisfied seating preferences.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"slices"
	"time"

	algo "github.com/mhbardsley/jubilant-octo-palm-tree"
)

type person struct {
	Name        string   `json:"name"`
	Preferences []string `json:"preferences"`
}

type plusOne struct {
	PersonOne string `json:"personOne"`
	PersonTwo string `json:"personTwo"`
}

type problem struct {
	People   []person  `json:"people"`
	Tables   []int     `json:"tables"`
	PlusOnes []plusOne `json:"plusOnes"`
}

// table holds the seated people plus a name set for O(1) "is X at this table?" lookups.
type table struct {
	capacity int
	people   []person
	members  map[string]struct{}
}

// assignment is a candidate seating plan; it implements algo.Individual.
type assignment struct {
	tables []table
	score  func([]table) float64
}

func (a *assignment) Fitness() float64 { return a.score(a.tables) }

func (a *assignment) Mutate() { swapTwo(a.tables) }

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
	a, b := &ts[i], &ts[j]
	pa := rand.IntN(a.capacity)
	pb := rand.IntN(b.capacity)
	a.people[pa], b.people[pb] = b.people[pb], a.people[pa]
	delete(a.members, b.people[pb].Name)
	delete(b.members, a.people[pa].Name)
	a.members[a.people[pa].Name] = struct{}{}
	b.members[b.people[pb].Name] = struct{}{}
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
func scorer(mode string, plusOnes map[string]string, totalPeople, totalPrefs int) (func([]table) float64, error) {
	switch mode {
	case "sum":
		return func(ts []table) float64 {
			_, sum, pen := scoreParts(ts, plusOnes)
			if pen > 0 {
				return -float64(pen)
			}
			return float64(sum)
		}, nil
	case "count":
		return func(ts []table) float64 {
			count, _, pen := scoreParts(ts, plusOnes)
			if pen > 0 {
				return -float64(pen)
			}
			return float64(count)
		}, nil
	case "hybrid":
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
		return nil, fmt.Errorf("unknown cost function %q (want sum, count, or hybrid)", mode)
	}
}

// pack distributes a flat slice of people into tables of the given capacities.
func pack(people []person, capacities []int) []table {
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

func generator(people []person, capacities []int, score func([]table) float64) func() *assignment {
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

		child := make([]person, total)
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

func flatten(ts []table, total int) []person {
	out := make([]person, 0, total)
	for _, t := range ts {
		out = append(out, t.people...)
	}
	return out
}

func printSolution(a *assignment, plusOnes map[string]string) {
	count, sum, _ := scoreParts(a.tables, plusOnes)
	total := 0
	for _, t := range a.tables {
		total += len(t.people)
	}
	fmt.Printf("Found a solution where %d people are given a preference (i.e. %d people have not been allocated at least one of their preferences). %d preferences are given in total.\n",
		count, total-count, sum)
	for i, t := range a.tables {
		fmt.Printf("\nTable %d (capacity %d)\n", i, t.capacity)
		for _, p := range t.people {
			fmt.Printf("- %s\n", p.Name)
		}
	}
}

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

	var prob problem
	if err := json.Unmarshal(raw, &prob); err != nil {
		log.Fatalf("error parsing input: %v", err)
	}

	plusOnes := make(map[string]string, len(prob.PlusOnes))
	for _, p := range prob.PlusOnes {
		plusOnes[p.PersonOne] = p.PersonTwo
	}

	totalPrefs := 0
	for _, p := range prob.People {
		totalPrefs += len(p.Preferences)
	}

	score, err := scorer(*mode, plusOnes, len(prob.People), totalPrefs)
	if err != nil {
		log.Fatal(err)
	}

	deadline := time.Now().Add(*runtime)
	best := algo.RunGeneticAlgorithm(algo.Config[*assignment]{
		PopulationSize:      *population,
		GenerateIndividual:  generator(prob.People, prob.Tables, score),
		Crossover:           crossover(prob.Tables, score),
		ContinuingCondition: func() bool { return time.Now().Before(deadline) },
	})

	printSolution(best, plusOnes)
}
