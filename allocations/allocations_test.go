package allocations

import (
	"testing"
	"time"
)

// scenario builds a (tables, score) pair from a layout-and-prefs spec for the
// hill-climb tests. Lighter than the old GA's `assignment` struct since
// localOptimize now takes the tables and score directly.
func scenario(t *testing.T, mode Mode, plusOnes map[string]string, layout [][]string, prefs map[string][]string) ([]table, func([]table) float64) {
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
	return pack(flat, caps), score
}

func TestLocalOptimizeFixesObviousMisseat(t *testing.T) {
	// A wants C (at table 1); B is indifferent. Initial: [A, B][C, D]. Swap A↔C gives A a friend.
	prefs := map[string][]string{"A": {"C"}, "B": nil, "C": nil, "D": nil}
	tables, score := scenario(t, ModeHybrid, nil, [][]string{{"A", "B"}, {"C", "D"}}, prefs)

	before := score(tables)
	localOptimize(tables, score)
	after := score(tables)

	if after <= before {
		t.Fatalf("expected improvement, before=%v after=%v", before, after)
	}
	together := false
	for _, tbl := range tables {
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
	tables, score := scenario(t, ModeHybrid, nil, [][]string{{"A", "B"}, {"C", "D"}}, prefs)

	before := score(tables)
	localOptimize(tables, score)
	if score(tables) != before {
		t.Fatalf("expected no change at local optimum, got before=%v after=%v", before, score(tables))
	}
}

func TestLocalOptimizeRespectsPlusOnes(t *testing.T) {
	// A and B are a plus-one pair. Start them apart; localOptimize must reunite them.
	prefs := map[string][]string{"A": nil, "B": nil, "C": nil, "D": nil}
	plusOnes := map[string]string{"A": "B"}
	tables, score := scenario(t, ModeHybrid, plusOnes, [][]string{{"A", "C"}, {"B", "D"}}, prefs)

	localOptimize(tables, score)
	_, _, pen := scoreParts(tables, plusOnes)
	if pen != 0 {
		t.Fatalf("plus-one violation after optimization: pen=%d", pen)
	}
}

// TestLocalOptimizeReachesKnownOptimum constructs a small instance whose
// optimum is obvious (every person's first preference is mutual and
// pair-aligned to a table size of 2), then scrambles it and asserts
// hill-climbing recovers the optimum. No external fixture.
func TestLocalOptimizeReachesKnownOptimum(t *testing.T) {
	// Six people in three couples; tables of two. Optimum: everyone with their partner.
	prefs := map[string][]string{
		"A": {"B"}, "B": {"A"},
		"C": {"D"}, "D": {"C"},
		"E": {"F"}, "F": {"E"},
	}
	tables, score := scenario(t, ModeHybrid, nil, [][]string{{"A", "C"}, {"B", "E"}, {"D", "F"}}, prefs)
	localOptimize(tables, score)
	count, sum, _ := scoreParts(tables, nil)
	if count != 6 || sum != 6 {
		t.Fatalf("expected count=6 sum=6 (all paired), got count=%d sum=%d", count, sum)
	}
}

// localOptimizeUntil with a deadline already in the past must return without
// improving the assignment — the whole point is bailing out fast on big inputs.
func TestLocalOptimizeUntilBailsPastDeadline(t *testing.T) {
	prefs := map[string][]string{"A": {"C"}, "B": nil, "C": nil, "D": nil}
	tables, score := scenario(t, ModeHybrid, nil, [][]string{{"A", "B"}, {"C", "D"}}, prefs)
	before := score(tables)

	localOptimizeUntil(tables, score, time.Now().Add(-1*time.Second))
	if score(tables) != before {
		t.Fatalf("expected no change when called past deadline, score moved from %v", before)
	}
}

// With a comfortable deadline, localOptimizeUntil must behave exactly like
// the unbounded localOptimize on a small input that converges in
// milliseconds.
func TestLocalOptimizeUntilWithSlackEqualsLocalOptimize(t *testing.T) {
	prefs := map[string][]string{
		"A": {"B"}, "B": {"A"},
		"C": {"D"}, "D": {"C"},
		"E": {"F"}, "F": {"E"},
	}
	tables, score := scenario(t, ModeHybrid, nil, [][]string{{"A", "C"}, {"B", "E"}, {"D", "F"}}, prefs)
	localOptimizeUntil(tables, score, time.Now().Add(5*time.Second))
	count, sum, _ := scoreParts(tables, nil)
	if count != 6 || sum != 6 {
		t.Fatalf("expected count=6 sum=6 with slack deadline, got count=%d sum=%d", count, sum)
	}
}
