package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"strconv"
	"time"
)

// define the datatypes needed, namely people and tables
type person struct {
	Name        string   `json:"name"` // must be unique
	Preferences []string `json:"preferences"`
}

type table struct {
	capacity  int
	people    []person
	peopleMap map[string]bool
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

// the main annealing function
func anneal(people []person, tables []table, plusOnes map[string]string, costFunction func([]table, map[string]string) float64, baseTemperature float64, finalTemperature float64, coolingRate float64, internalIterations int, swapCount int, concurrentAnnealerCount int) (result []table) {
	initialSolution := randomInitialisation(people, tables)

	// create a channel for concurrent annealers of differing temperatures
	annealerSolution := make(chan []table)
	annealerCost := make(chan float64)

	annealerSolutions := make([][]table, concurrentAnnealerCount)
	annealerCosts := make([]float64, concurrentAnnealerCount)

	for i := 0; i < concurrentAnnealerCount; i++ {
		annealerSolutions[i] = copyAssignment(initialSolution)
		annealerCosts[i] = costFunction(initialSolution, plusOnes)
	}

	// while we haven't hit the final temperature
	for baseTemperature > finalTemperature {

		for i := 0; i < concurrentAnnealerCount; i++ {
			go annealerInternalIterator(annealerSolutions[i], plusOnes, costFunction, baseTemperature*math.Pow(2, float64(i)), internalIterations, swapCount, annealerSolution, annealerCost)
			annealerSolutions[i] = <-annealerSolution
			annealerCosts[i] = <-annealerCost
		}

		// If a hotter goroutine has a better solution than a colder one then we swap the solutions
		for i := concurrentAnnealerCount - 1; i > 0; i-- {
			if annealerCosts[i] > annealerCosts[i-1] {
				annealerSolutions[i], annealerSolutions[i-1] = annealerSolutions[i-1], annealerSolutions[i]
				annealerCosts[i], annealerCosts[i-1] = annealerCosts[i-1], annealerCosts[i]
			}
		}

		// Cool all of the goroutines
		baseTemperature *= coolingRate
	}

	return annealerSolutions[0]
}

// Gets a neighbouring candidate solution and runs the probibalistic steps of the annealing process as many times as
// specified by the internalIterations count.
func annealerInternalIterator(candidateSolution []table, plusOnes map[string]string, costFunction func([]table, map[string]string) float64, temperature float64, internalIterations int, swapCount int, as chan []table, ac chan float64) {

	// Set updatedSolution and updatedCost to the current values associated with candidateSolution
	updatedSolution := copyAssignment(candidateSolution)
	updatedCost := costFunction(updatedSolution, plusOnes)

	for i := 0; i < internalIterations; i++ {
		newCandidateSolution := getNeighbour(updatedSolution, swapCount)
		newCandidateCost := costFunction(newCandidateSolution, plusOnes)

		// if the cost is more then switch to that solution
		if newCandidateCost > updatedCost {
			updatedSolution = newCandidateSolution
			updatedCost = newCandidateCost

			// And finally switch to a more costly solution randomly based on the acceptance probablity
		} else {
			ap := acceptanceProbability(updatedCost, newCandidateCost, temperature)

			if ap > rand.Float64() {
				updatedSolution = newCandidateSolution
				updatedCost = newCandidateCost
			}
		}
	}

	as <- updatedSolution
	ac <- updatedCost
}

// Gets a neighbouring candidate solution to the current one
func getNeighbour(currentAssignment []table, swapCount int) (neighbourAssignment []table) {

	cal := len(currentAssignment)

	neighbourAssignment = copyAssignment(currentAssignment)

	for i := 0; i < swapCount; i++ {
		// generate two distinct random numbers so we know we are shuffling people in different tables
		randOne := rand.Intn(cal)
		randTwo := rand.Intn(cal - 1)

		if randTwo >= randOne {
			randTwo++
		}
		tableOne := neighbourAssignment[randOne]
		tableTwo := neighbourAssignment[randTwo]

		// generate two further indexes for the people
		randThree := rand.Intn(tableOne.capacity)
		randFour := rand.Intn(tableTwo.capacity)

		personOne := tableOne.people[randThree]
		personTwo := tableTwo.people[randFour]

		// Swap the two randomly selected elements
		tableOne.people[randThree], tableTwo.people[randFour] = personTwo, personOne
		delete(tableOne.peopleMap, personOne.Name)
		delete(tableTwo.peopleMap, personTwo.Name)
		tableOne.peopleMap[personTwo.Name] = true
		tableTwo.peopleMap[personOne.Name] = true
	}

	return neighbourAssignment
}

// the cost function is the sum of preferences
func sumFunction(assignment []table, plusOnes map[string]string) (cost float64) {
	// need to make sure the penalty for not having a plus one is greater than any possible combination of preferences
	noOfPenalties := 0
	cost = 0
	for _, table := range assignment {
		for _, person := range table.people {
			plusOne, exists := plusOnes[person.Name]
			if exists && !table.peopleMap[plusOne] {
				noOfPenalties++
			}
			for _, preference := range person.Preferences {
				if table.peopleMap[preference] {
					cost++
				}
			}
		}
	}
	if noOfPenalties > 0 {
		cost = float64(-noOfPenalties)
	}
	return cost
}

// the cost function is the count of people with >= 1 preferences
func countFunction(assignment []table, plusOnes map[string]string) (cost float64) {
	// need to make sure the penalty for not having a plus one is greater than any possible combination of preferences
	noOfPenalties := 0
	cost = 0
	for _, table := range assignment {
		for _, person := range table.people {
			plusOne, exists := plusOnes[person.Name]
			if exists && !table.peopleMap[plusOne] {
				noOfPenalties++
			}
			for _, preference := range person.Preferences {
				if table.peopleMap[preference] {
					cost++
					break
				}
			}
		}
	}
	if noOfPenalties > 0 {
		cost = float64(-noOfPenalties)
	}
	return cost
}

// this cost function presents a hybrid - prioritising everyone having >= 1 preference whilst keeping as many preferences
func hybridFunction(assignment []table, plusOnes map[string]string) (cost float64) {
	noOfPeople := getNoOfPeople(assignment)
	totalPrefs := getTotalPrefs(assignment)
	highestPossibleCost := math.Max(float64(noOfPeople), float64(totalPrefs))
	count := countFunction(assignment, plusOnes)
	sum := sumFunction(assignment, plusOnes)
	return count*highestPossibleCost + sum
}

// getTotalPrefs returns the total number of preferences across the assignment
func getTotalPrefs(assignment []table) int {
	current := 0
	for _, table := range assignment {
		for _, person := range table.people {
			current += len(person.Preferences)
		}
	}
	return current
}

// getNoOfPeople returns the number of people in the assignment
func getNoOfPeople(assignment []table) int {
	current := 0
	for _, table := range assignment {
		current += len(table.people)
	}
	return current
}

func acceptanceProbability(oldCost float64, newCost float64, temperature float64) (probability float64) {
	return math.Exp((newCost - oldCost) / temperature)
}

// randomly assigns people to tables
func randomInitialisation(people []person, tables []table) (assignment []table) {
	assignment = tables

	for i := range people {
		j := rand.Intn(i + 1)
		people[i], people[j] = people[j], people[i]
	}

	// now just fill forwards
	pos := 0
	for i, table := range assignment {
		assignment[i].people = people[pos : pos+table.capacity]
		for _, person := range table.people {
			table.peopleMap[person.Name] = true
		}
		pos += table.capacity
	}
	return assignment
}

// copies the assignment
func copyAssignment(initialAssignment []table) (copiedAssignment []table) {
	size := len(initialAssignment)

	copiedAssignment = make([]table, size)

	for i := 0; i < size; i++ {
		copiedAssignment[i].capacity = initialAssignment[i].capacity
		copiedAssignment[i].people = make([]person, copiedAssignment[i].capacity)
		copiedAssignment[i].peopleMap = make(map[string]bool)
		copy(copiedAssignment[i].people, initialAssignment[i].people)
		for k, v := range initialAssignment[i].peopleMap {
			copiedAssignment[i].peopleMap[k] = v
		}
	}

	return copiedAssignment
}

func printSolution(solution []table, plusOnes map[string]string) {
	fmt.Printf("Found a solution where %d people are given a preference (i.e. %d people have not been allocated at least one of their preferences). %d preferences are given in total", int(countFunction(solution, plusOnes)), getNoOfPeople(solution)-int(countFunction(solution, plusOnes)), int(sumFunction(solution, plusOnes)))
	fmt.Println()
	fmt.Println()
	for tableNo, table := range solution {
		fmt.Printf("Table %d (capacity %d)", tableNo, table.capacity)
		fmt.Println()
		for _, person := range table.people {
			fmt.Printf("- %s", person.Name)
			fmt.Println()
		}
		if tableNo < len(solution)-1 {
			fmt.Println()
		}
	}
}

func main() {
	// generate the random seed
	rand.Seed(time.Now().Unix())

	costFunctionPtr := flag.String("m", "hybrid", "Whether the program should: maximise the total number of satisifed preferences; maximise the number of people with at least 1 satisfied preference; provide a hybrid of these")
	filePtr := flag.String("f", "input.json", "The filename to be checked")
	baseTemperaturePtr := flag.String("b", "1.0", "The lowest base temperature for the concurrent annealers (temperature increases by 2^i for each goroutine i) - lower is quicker; higher is more optimal")
	endTemperaturePtr := flag.String("e", "0.00001", "The lowest final temperature for the concurrent annealers (temperature increases by 2^i for each goroutine i) - lower is more optimal; higher is quicker")
	coolingRatePtr := flag.String("c", "0.9", "The rate of cooling for each step in the annealing process (a number greater than 0 and less than 1) - closer to 0 is quicker; closer to 1 is more optimal")
	iterationPtr := flag.String("i", "1000", "The number of iterations at each step of the annealing process - lower is quicker; higher is more optimal")
	swapPtr := flag.String("s", "1", "The number of swaps in each iteration of the anneling process - lower is quicker; higher is more optimal")
	concurrentAnnealerPtr := flag.String("a", "6", "The number of concurrent annealing goroutines")

	flag.Parse()

	baseTemperature, _ := strconv.ParseFloat(*baseTemperaturePtr, 64)
	endTemperature, _ := strconv.ParseFloat(*endTemperaturePtr, 64)
	coolingRate, _ := strconv.ParseFloat(*coolingRatePtr, 64)
	internalIterations, _ := strconv.Atoi(*iterationPtr)
	swapCount, _ := strconv.Atoi(*swapPtr)
	annealerCount, _ := strconv.Atoi(*concurrentAnnealerPtr)

	problemRaw, err := ioutil.ReadFile(*filePtr)
	if err != nil {
		log.Fatal("error opening file: ", err)
	}

	// unmarshall data into payload
	var problemContent problem
	err = json.Unmarshal(problemRaw, &problemContent)
	if err != nil {
		log.Fatal("error making sense of input file: ", err)
	}

	// convert the slice of table capacities into a slice of table structs
	initialTables := make([]table, len(problemContent.Tables))

	for i := range problemContent.Tables {
		initialTables[i].capacity = problemContent.Tables[i]
		initialTables[i].people = make([]person, initialTables[i].capacity)
		initialTables[i].peopleMap = make(map[string]bool)
	}

	// parse through the plus-ones
	plusOnes := make(map[string]string)
	for _, p := range problemContent.PlusOnes {
		plusOnes[p.PersonOne] = p.PersonTwo
	}

	var costFunction func([]table, map[string]string) float64
	switch *costFunctionPtr {
	case "hybrid":
		costFunction = hybridFunction
	case "sum":
		costFunction = sumFunction
	case "count":
		costFunction = countFunction
	default:
		log.Fatal("provided cost function parameter not understood")
	}

	solution := anneal(problemContent.People, initialTables, plusOnes, costFunction, baseTemperature, endTemperature, coolingRate, internalIterations, swapCount, annealerCount)

	printSolution(solution, plusOnes)
}
