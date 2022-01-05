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

type problem struct {
	People []person `json:"people"`
	Tables []int    `json:"tables"`
}

// the main annealing function
func anneal(people []person, tables []table, baseTemperature float64, finalTemperature float64, coolingRate float64, internalIterations int, swapCount int, concurrentAnnealerCount int) (result []table) {
	initialSolution := randomInitialisation(people, tables)

	// create a channel for concurrent annealers of differing temperatures
	annealerSolution := make(chan []table)
	annealerCost := make(chan int)

	annealerSolutions := make([][]table, concurrentAnnealerCount)
	annealerCosts := make([]int, concurrentAnnealerCount)

	for i := 0; i < concurrentAnnealerCount; i++ {
		annealerSolutions[i] = copyAssignment(initialSolution)
		annealerCosts[i] = costFunction(initialSolution)
	}

	// while we haven't hit the final temperature
	for baseTemperature > finalTemperature {

		for i := 0; i < concurrentAnnealerCount; i++ {
			go annealerInternalIterator(annealerSolutions[i], baseTemperature*math.Pow(2, float64(i)), internalIterations, swapCount, annealerSolution, annealerCost)
			annealerSolutions[i] = <-annealerSolution
			annealerCosts[i] = <-annealerCost
		}

		// If a hotter goroutine has a better solution than a colder one then we swap the solutions
		for i := concurrentAnnealerCount - 1; i > 0; i-- {
			if annealerCosts[i] < annealerCosts[i-1] {
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
func annealerInternalIterator(candidateSolution []table, temperature float64, internalIterations int, swapCount int, as chan []table, ac chan int) {

	// Set updatedSolution and updatedCost to the current values associated with candidateSolution
	updatedSolution := copyAssignment(candidateSolution)
	updatedCost := costFunction(updatedSolution)

	for i := 0; i < internalIterations; i++ {
		newCandidateSolution := getNeighbour(updatedSolution, swapCount)
		newCandidateCost := costFunction(newCandidateSolution)

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
func costFunction(assignment []table) (cost int) {
	cost = 0
	for _, table := range assignment {
		for _, person := range table.people {
			for _, preference := range person.Preferences {
				if table.peopleMap[preference] {
					cost++
				}
			}
		}
	}
	return cost
}

func acceptanceProbability(oldCost int, newCost int, temperature float64) (probability float64) {
	return math.Exp((float64)(newCost-oldCost) / temperature)
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
	for _, table := range assignment {
		table.people = people[pos : pos+table.capacity]
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

func printSolution(solution []table) {
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

	filePtr := flag.String("f", "input.json", "The filename to be checked")
	baseTemperaturePtr := flag.String("b", "1.0", "The lowest base temperature for the concurrent annealers (temperature increases by 2^i for each goroutine i)")
	endTemperaturePtr := flag.String("e", "0.00001", "The lowest final temperature for the concurrent annealers (temperature increases by 2^i for each goroutine i)")
	coolingRatePtr := flag.String("c", "0.9", "The rate of cooling for each step in the annealing process (a number greater than 0 and less than 1)")
	iterationPtr := flag.String("i", "1000", "The number of iterations at each step of the annealing process")
	swapPtr := flag.String("s", "1", "The number of swaps in each iteration of the anneling process")
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

	solution := anneal(problemContent.People, initialTables, baseTemperature, endTemperature, coolingRate, internalIterations, swapCount, annealerCount)

	printSolution(solution)
}
