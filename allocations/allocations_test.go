package allocations

import "testing"

func newAssignment(t *testing.T, mode Mode, plusOnes map[string]string, layout [][]string, prefs map[string][]string) *assignment {
	t.Helper()
	totalPeople, totalPrefs := 0, 0
	caps := make([]int, len(layout))
	flat := []Person{}
	for i, names := range layout {
		caps[i] = len(names)
		for _, n := range names {
			flat = append(flat, Person{Name: n, Preferences: prefs[n]})
			totalPeople++
			totalPrefs += len(prefs[n])
		}
	}
	score, err := scorer(mode, plusOnes, totalPeople, totalPrefs)
	if err != nil {
		t.Fatalf("scorer: %v", err)
	}
	return &assignment{tables: pack(flat, caps), score: score}
}

func TestLocalOptimizeFixesObviousMisseat(t *testing.T) {
	// A wants C (at table 1); B is indifferent. Initial: [A, B][C, D]. Swap A↔C gives A a friend.
	prefs := map[string][]string{"A": {"C"}, "B": nil, "C": nil, "D": nil}
	a := newAssignment(t, ModeHybrid, nil, [][]string{{"A", "B"}, {"C", "D"}}, prefs)

	before := a.score(a.tables)
	localOptimize(a)
	after := a.score(a.tables)

	if after <= before {
		t.Fatalf("expected improvement, before=%v after=%v", before, after)
	}
	together := false
	for _, tbl := range a.tables {
		_, hasA := tbl.members["A"]
		_, hasC := tbl.members["C"]
		if hasA && hasC {
			together = true
		}
	}
	if !together {
		t.Fatalf("expected A and C to be seated together after optimization")
	}
}

func TestLocalOptimizeIsNoOpAtLocalOptimum(t *testing.T) {
	// Already optimal: A+B mutual, C+D mutual.
	prefs := map[string][]string{"A": {"B"}, "B": {"A"}, "C": {"D"}, "D": {"C"}}
	a := newAssignment(t, ModeHybrid, nil, [][]string{{"A", "B"}, {"C", "D"}}, prefs)

	before := a.score(a.tables)
	localOptimize(a)
	if a.score(a.tables) != before {
		t.Fatalf("expected no change at local optimum, got before=%v after=%v", before, a.score(a.tables))
	}
}

func TestLocalOptimizeRespectsPlusOnes(t *testing.T) {
	// A and B are a plus-one pair. Start them apart; localOptimize must reunite them.
	prefs := map[string][]string{"A": nil, "B": nil, "C": nil, "D": nil}
	plusOnes := map[string]string{"A": "B"}
	a := newAssignment(t, ModeHybrid, plusOnes, [][]string{{"A", "C"}, {"B", "D"}}, prefs)

	localOptimize(a)
	_, _, pen := scoreParts(a.tables, plusOnes)
	if pen != 0 {
		t.Fatalf("plus-one violation after optimization: pen=%d", pen)
	}
}

// TestLocalOptimizeReachesKnownOptimum constructs a small instance whose optimum is obvious
// (every person's first preference is mutual and pair-aligned to a table size of 2), then
// scrambles it and asserts hill-climbing recovers the optimum. No external fixture, no GA.
func TestLocalOptimizeReachesKnownOptimum(t *testing.T) {
	// Six people in three couples; tables of two. Optimum: everyone with their partner.
	prefs := map[string][]string{
		"A": {"B"}, "B": {"A"},
		"C": {"D"}, "D": {"C"},
		"E": {"F"}, "F": {"E"},
	}
	// Scrambled start: nobody seated with their partner.
	a := newAssignment(t, ModeHybrid, nil, [][]string{{"A", "C"}, {"B", "E"}, {"D", "F"}}, prefs)
	localOptimize(a)
	count, sum, _ := scoreParts(a.tables, nil)
	if count != 6 || sum != 6 {
		t.Fatalf("expected count=6 sum=6 (all paired), got count=%d sum=%d", count, sum)
	}
}
