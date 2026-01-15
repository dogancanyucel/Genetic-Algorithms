import math  # Import the math module for mathematical functions like sqrt
import random  # Import the random module for generating random numbers and sampling
import os  # Import os to check if files exist
import statistics  # Import statistics for calculating mean, median, stdev

try:  # Start a try block to handle potential import errors
    import matplotlib.pyplot as plt  # Attempt to import matplotlib for plotting graphs
    HAS_MATPLOTLIB = True  # Set flag to True if import is successful
except ImportError:  # Catch the error if matplotlib is not installed
    print("WARNING: matplotlib not installed. Graphs will not be shown.")  # Inform the user regarding missing library
    HAS_MATPLOTLIB = False  # Set flag to False so plotting code is skipped later

# --- ITEM 1: Data Structure (Class Structure) ---
class City:  # Define a class named City to represent a city's data
    def __init__(self, id, x, y):  # Constructor method to initialize the object
        self.id = int(id)  # Convert the ID to an integer and store it
        self.x = float(x)  # Convert the X coordinate to a float and store it
        self.y = float(y)  # Convert the Y coordinate to a float and store it

    def __repr__(self):  # Define how the object is represented as a string
        return f"City(ID={self.id}, X={self.x}, Y={self.y})"  # Return a formatted string with city details


# --- ITEM 1: Parser Function ---
def parse_tsp_file(filename):  # Define a function to read TSP files
    cities = []  # Initialize an empty list to store City objects
    if not os.path.exists(filename):  # Check if the file exists at the specified path
        print(f"WARNING: File '{filename}' not found.")  # Print a warning if file is missing
        return []  # Return an empty list since no file was found

    try:  # Start a try block to handle file reading errors
        with open(filename, 'r') as file: # Open the file in read mode ('r')
            lines = file.readlines() # Read all lines from the file into a list
            parsing = False # Initialize a flag to track if coordinate data has started
            for line in lines:  # Iterate through each line in the file
                if "NODE_COORD_SECTION" in line: # Check if the line indicates the start of coordinates
                    parsing = True  # Set the parsing flag to True
                    continue  # Skip to the next iteration (next line)
                if "EOF" in line: # Check if the line indicates the End Of File
                    break  # Exit the loop as reading is done

                if parsing:  # If we are currently in the coordinate section
                    parts = line.strip().split() # Remove whitespace and split line into parts (ID, X, Y)
                    if len(parts) >= 3:  # Ensure there are at least 3 parts (ID, X, Y)
                        new_city = City(parts[0], parts[1], parts[2]) # Create a new City object with parsed data
                        cities.append(new_city) # Add the new city object to the list
        print(f"SUCCESSFUL: Loaded {len(cities)} cities from file {filename}.")  # Print success message with count
        return cities  # Return the list of city objects
    except Exception as e: # Catch any errors that occur during file processing
        print(f"ERROR: {e}")  # Print the specific error message
        return []  # Return an empty list on failure

# --- ITEM 2: Distance Function ---
def calculate_distance(city1, city2):  # Define function to calculate distance between two cities
    x_diff = city1.x - city2.x  # Calculate the difference in X coordinates
    y_diff = city1.y - city2.y  # Calculate the difference in Y coordinates
    distance = math.sqrt(x_diff ** 2 + y_diff ** 2) # Apply Pythagorean theorem to find Euclidean distance
    return distance  # Return the calculated distance

# --- ITEM 3 & 4: Solution Storage & Random Solution ---
def create_random_solution(cities):  # Define function to generate a random tour
    random_solution = random.sample(cities, len(cities)) # Create a random permutation of the cities list
    return random_solution  # Return the shuffled list of cities

# --- EXTRA (PART 2): Fitness Calculation ---
def calculate_fitness(solution):  # Define function to calculate total distance (fitness) of a tour
    total_distance = 0  # Initialize total distance counter to zero
    num_cities = len(solution)  # Get the total number of cities in the solution

    for i in range(num_cities):  # Iterate through indices of the cities
        city_start = solution[i]  # Get the current city
        city_end = solution[(i + 1) % num_cities]  # Get the next city (wrap around to start using modulo)
        total_distance += calculate_distance(city_start, city_end) # Add distance between current and next city
    return total_distance  # Return the total cumulative distance

#info
def info(solution):  # Define a function to display information about a solution
    score = calculate_fitness(solution) # Calculate the fitness (total distance) of the solution

    #element id convert int to str
    route_ids = [str(city.id) for city in solution]  # Create a list of city IDs as strings
    route_str = " ".join(route_ids) # Join IDs with a space to make a single string

    print(f"Solution Route: {route_str}")  # Print the route sequence
    print(f"Score(fitness): {score:.4f}") # Print the score formatted to 4 decimal places

#greedy
def solve_greedy(cities, start_index=0):  # Define the Greedy (Nearest Neighbor) algorithm function
    """
    Greedy Algorithm (Nearest Neighbor):
    Starts at a chosen node and always picks the closest unvisited city next.
    """
    if not cities: return []  # Return empty list if input cities list is empty

    #1. Make a copy to preserve the original list.
    unvisited = cities.copy()  # Create a copy of the cities list to track unvisited nodes

    # 2. choose your start city
    if start_index >= len(unvisited): start_index = 0  # Reset index to 0 if provided index is out of bounds
    current_city = unvisited.pop(start_index) # Remove the starting city from unvisited list and set as current
    solution = [current_city]  # Initialize solution list with the starting city

    # 3. Turn around until all the cities are gone.
    while unvisited:  # Loop as long as there are cities left in unvisited list
        closest_city = None  # Initialize variable to store the closest city found
        min_distance = float("inf")  # Set minimum distance to infinity initially

        # --- Part A:Just find the NEAREST one. ---
        for candidate in unvisited:  # Iterate through all remaining unvisited cities
            dist = calculate_distance(current_city, candidate)  # Calculate distance from current city to candidate
            if dist < min_distance:  # Check if this distance is smaller than current minimum
                min_distance = dist  # Update minimum distance
                closest_city = candidate  # Update closest city to this candidate

        # --- Part B: Once you find it, GO and DELETE IT FROM THE LIST. ---
        if closest_city:  # If a closest city was found
            current_city = closest_city  # Move to the closest city
            solution.append(current_city)  # Add it to the solution path
            unvisited.remove(current_city)  # Remove it from the unvisited list
        else:  # Fallback (should typically not be reached if list is valid)
            break  # Break the loop

    return solution  # Return the constructed greedy path

def create_population(cities , population_size):  # Define function to create a population of solutions
    population = []  # Initialize empty list for population
    for _ in range(population_size):# Loop 'population_size' times (variable _ is ignored)
        solution = create_random_solution(cities)  # Create a random solution
        population.append(solution)  # Add solution to population list
    return population  # Return the full population

"""
The `create_initial_population` function allows 
me to add a specific number of Greedy individuals to the population.
"""
def create_initial_population(cities,pop_size,greedy_count = 0):  # Function to create population mixed with greedy solutions
    population = []  # Initialize empty population list
    for i in range(greedy_count):  # Loop to create specific number of greedy solutions
        start_node_index = i%len(cities) # Calculate starting index (modulo ensures valid index)
        greedy_sol = solve_greedy(cities,start_index = start_node_index)  # Generate greedy solution from that start node
        population.append(greedy_sol)  # Add greedy solution to population

    remaining_count = pop_size - greedy_count  # Calculate how many more random solutions are needed
    if remaining_count <0 : remaining_count = 0  # Ensure remaining count is not negative
    # If we tell it to generate -5 solutions, it can't generate -5, so it will generate 0.

    for _ in range(remaining_count):  # Loop to create the remaining random solutions
        random_sol = create_random_solution(cities)  # Generate a random solution
        population.append(random_sol)  # Add random solution to population
    return population  # Return the mixed population


# --- ITEM 13: Population Information ---
def info_population(population):  # Define function to print statistics about the population
    if not population:  # Check if population list is empty
        print("Population is empty.")  # Print warning
        return  # Exit function
    # 1. Calculate everyone's fitness score.
    scores = [calculate_fitness(sol) for sol in population]  # Create list of scores for all solutions

    # 2. Extract the statistics.
    best_score = min(scores)  # Find the minimum (best) score
    worst_score = max(scores)  # Find the maximum (worst) score
    avg_score=sum(scores)/len(scores)  # Calculate the average score

    # 3. Median calculation (Middle value)
    sorted_scores = sorted(scores)  # Sort scores in ascending order
    mid_index = len(scores) // 2 # Calculate the middle index (integer division)
    median_score = sorted_scores[mid_index]  # Get the score at the middle index

    print(f"Population Size : {len(population)}")  # Print population size
    print(f"Best Score : {best_score:.4f}")  # Print best score
    print(f"Worst Score : {worst_score:4f}")  # Print worst score
    print(f"Average Score : {avg_score:.4f}")  # Print average score
    print(f"Median Score : {median_score:.4f}")  # Print median score

#task 14
def tournament_selection(population,tournament_size=5):  # Define tournament selection for parent selection
    #1. Select candidates randomly.
    #This is a precaution to prevent errors if the population size is smaller than the tournament size.
    k = min(len(population),tournament_size)  # Set sample size (smaller of population length or tournament size)
    candidates = random.sample(population,k)  # Randomly select 'k' candidates from population

    #2. Find the best (shortest distance) candidate among the options.
    best_candidate = None  # Initialize best candidate container
    best_score = float('inf')  # Initialize best score to infinity

    for candidate in candidates:  # Iterate through selected candidates
        score = calculate_fitness(candidate)  # Calculate fitness of the candidate
        if score < best_score:  # If this score is better (lower) than current best
            best_score = score  # Update best score
            best_candidate = candidate  # Update best candidate

    return best_candidate  # Return the winner of the tournament

def ordered_crossover(parent1,parent2):  # Define Ordered Crossover (OX1) function
    size =len(parent1)  # Get the number of cities (genome length)
    cut1,cut2 = sorted(random.sample(range(size),2)) # Pick two random cut points and sort them
    #cut1 8 cut2 2 is god is still work but cut1 2 cut2 8 it will be work to empty list

    child = [None] * size  # Initialize child with None values of correct size

    child[cut1:cut2] = parent1[cut1:cut2]  # Copy the segment between cuts from Parent 1 to Child

    current_ids = set(city.id for city in child[cut1:cut2])  # Create a set of IDs already in the child for fast lookup

    p2_index = 0  # Initialize index tracker for Parent 2
    for i in range(size):  # Iterate through all positions in the child
        if child[i] is None:  # If the position is empty (None)
            while p2_index < size and parent2[p2_index].id in current_ids: # Skip genes in Parent 2 that are already in Child
                p2_index += 1  # Move to next gene in Parent 2

            if p2_index < size:  # If we haven't exhausted Parent 2
                child[i] = parent2[p2_index]  # Fill the empty spot with the gene from Parent 2
                current_ids.add(parent2[p2_index].id)  # Add this ID to the set of used IDs

    return child  # Return the created child solution

def inversion_mutation(solution,mutation_rate=0.1):  # Define Inversion Mutation function
    #Roll the dice (a random number between 0.0 and 1.0)
    # mutation_rate = there is a 10% chance that we will make a random change (mutation) in its genes.
    if random.random() < mutation_rate:  # Check if random number is less than mutation rate
        #Make a copy so as not to alter the original list.
        mutated_sol = solution[:]  # Create a shallow copy of the solution
        # 2. Select two random breakpoints.
        size = len(solution)  # Get size of solution
        idx1 , idx2 = random.sample(range(size),2)  # Pick two random indices
        start,end = min(idx1,idx2),max(idx1,idx2)  # Determine start (min) and end (max) indices

        # 3. Invert the intermediate part (Inversion)
        # In Python, [::-1] is the inversion operation.
        segment = mutated_sol[start:end+1]  # Extract the segment to be inverted
        mutated_sol[start:end+1] = segment[::-1]  # Replace segment with its reverse

        return mutated_sol  # Return the mutated solution
    # If the probability didn't work, send it back as is.
    return solution  # Return original solution if no mutation occurred

def create_new_generation(previous_population,mutation_rate=0.1,crossover_rate=0.8): # Function to create next generation
    #previous_population = we select parent(mom and dad)
    #crossover_rate = Theres an 80% chance we'll mix the parents genes and create a hybrid child. 20% same genes to parent
    #mutation_rate = After the child is created, there is a 10% chance that we will make a random change (mutation) in its genes.
    new_population = []  # Initialize list for new generation
    pop_size = len(previous_population)  # Get size of current population

    while len(new_population) < pop_size:  # Loop until new population is full
        parent1 = tournament_selection(previous_population , tournament_size=5) # Select Parent 1 via tournament
        parent2 = tournament_selection(previous_population , tournament_size=5) # Select Parent 2 via tournament
        #Select 5 people from the population and choose the one
        # with the shortest path (best) and assign it as "Parent".

        #selection
        if random.random() < crossover_rate:  # Check if crossover should happen based on rate
            child = ordered_crossover(parent1,parent2)  # Create child using crossover
        else:
            child = parent1[:] # [:] cloning to the list - Child is just a clone of Parent 1
            #If I hadn’t used [:], the mutations I made on the child would also have changed the parent.

        #crossover
        if None in child:  # Safety check: if child has empty spots (shouldn't happen)
            continue  # Skip adding this child and try again

        #mutation
        child= inversion_mutation(child,mutation_rate)  # Apply mutation to the child


        if len(child) != len(parent1):  # Safety check: ensure child length is correct
            continue  # Skip if length is wrong

        new_population.append(child)  # Add the child to the new population
    return new_population  # Return the complete new generation

def solve_tsp_genetic(cities, pop_size=100, iterations=3000, greedy_count=20, mutation_rate=0.1, crossover_rate=0.85): # Main GA function
    #pop_size =number of pirates searching for treasure
    #iterations = Epoch

    current_pop = create_initial_population(cities, pop_size, greedy_count)  # Create the starting population

    best_solution = min(current_pop, key=calculate_fitness) # Find the initial best solution. `key` tells you which field you will be ranked by.
    best_score = calculate_fitness(best_solution)  # Calculate score of the best solution
    fitness_history = [best_score]  # Initialize history log with starting best score

    for i in range(1, iterations + 1):  # Loop through generations (iterations)

        current_pop = create_new_generation(current_pop, mutation_rate, crossover_rate)  # Evolve population to next generation

        current_best = min(current_pop, key=calculate_fitness)  # Find the best individual in current generation
        current_score = calculate_fitness(current_best)  # Calculate its score

        if current_score < best_score:  # If current generation's best is better than global best
            best_score = current_score  # Update global best score
            best_solution = current_best[:]  # Update global best solution (copy)

        fitness_history.append(best_score)  # Append current best score to history log

    return best_solution, best_score, fitness_history  # Return final results

def plot_results(history , best_route):  # Function to visualize results
    if not HAS_MATPLOTLIB:  # Check if matplotlib is available
        print("Matplotlib is missing. Skipping graphs.")  # Print error if missing
        return  # Exit function

    #1. Graph: Learning Curve (Score vs Epoch)
    plt.figure(figsize=(12,5)) # Create a new figure window of size 12x5

    # Left side: Scoreboard
    plt.subplot(1,2,1) # Select the first subplot (1 row, 2 cols, index 1)
    plt.plot(history)  # Plot the fitness history data
    plt.title("Genetic Algorithm Convergence")  # Set title for this plot
    plt.xlabel("Epoch(Generation)")  # Label X-axis
    plt.ylabel("Best Score (Distance)")  # Label Y-axis
    plt.grid(True) #Add grid lines behind the graph.

    #2.Graphic: Route Map (Cities and Roads)
    plt.subplot(1,2,2)  # Select the second subplot (index 2)

    #Get the coordinates of the cities.
    x_coords = [city.x for city in best_route]  # Extract all X coordinates from best route
    y_coords = [city.y for city in best_route]  # Extract all Y coordinates from best route

    #To close the road, add the starting city to the end.
    x_coords.append(best_route[0].x)  # Append start X to end of list
    y_coords.append(best_route[0].y)  # Append start Y to end of list

    plt.plot(x_coords,y_coords,'o-r') #'o-r' redline with dots - Plot the route
    plt.title(f"Best Route Found (score : {calculate_fitness(best_route):.2f})") # Set title with score

    for city in best_route:  # Loop through cities to label them
        plt.annotate(str(city.id), (city.x, city.y)) #Write a note/sticker on the dot (City ID).

    plt.tight_layout() #Set everything to automatic, ensure text doesn't overlap, and make sure margins are neat.
    plt.show()  # Display the plot window

# --- TASK 20
def run_parameter_comparison(cities):  # Function to compare different mutation rates
    if not HAS_MATPLOTLIB:  # Check for matplotlib
        print("Matplotlib is missing. Skipping graphs.")  # Exit if missing
        return

    scenarios = {"Low Mutation (1%)": 0.01, "Normal Mutation (10%)": 0.1, "High Mutation (50%)": 0.5} # Define scenarios dictionary
    plt.figure(figsize=(10, 6))  # Create figure
    print("\n--- COMPARISON STARTED ---")  # Print start message
    for name, m_rate in scenarios.items(): # name Low Mutation (1%)| m_rate 0.01 - Iterate over scenarios
        print(f"Running scenario: {name} ...")  # Print current scenario name

        _, final_score, history = solve_tsp_genetic(  # Run GA with specific parameters
            cities,
            pop_size=100,
            iterations=1500,
            greedy_count=10,
            mutation_rate=m_rate
        )
        # add Graf
        plt.plot(history, label=f"{name} (Score: {final_score:.0f})") # Plot history and add label

    # graffic settings
    plt.title("Impact of Mutation Rate on Convergence")  # Set chart title
    plt.xlabel("Epochs")  # Label X-axis
    plt.ylabel("Best Score (Distance)")  # Label Y-axis
    plt.legend() # It's the box in the corner of the graph that says "Which color line is which?".
    plt.grid(True)  # Enable grid

    print("Displaying Comparison Chart...")  # Print message
    plt.show()  # Show the chart


def generate_final_report_stats(cities):  # Function to generate comprehensive statistics report
    print("\n" + "=" * 60)  # Print separator line
    print("      PART 3: FINAL STATISTICAL REPORT GENERATION")  # Print header
    print("=" * 60)  # Print separator line
    print(f"Analyzing {len(cities)} cities...")  # Print number of cities

    # ---------------------------------------------------------
    # 1. RANDOM SEARCH (1000 RUNS)
    # ---------------------------------------------------------
    print("\n1. Running RANDOM SEARCH (1000 tests)...")  # Print section header
    random_scores = []  # Initialize list for random scores
    for _ in range(1000):  # Loop 1000 times
        sol = create_random_solution(cities)  # Create a random solution
        random_scores.append(calculate_fitness(sol))  # Calculate and save fitness

    r_best = min(random_scores)  # Find best score in random tests
    r_mean = statistics.mean(random_scores) # You made 1000 attempts. You add up all the scores and divide by 1000.
    r_stdev = statistics.stdev(random_scores) #Standard Deviation  It shows how much the data deviates from the average.
    r_variance = statistics.variance(random_scores) # It is the square of the standard deviation (s^2).

    print(f"   Done. Best Random: {r_best:.2f}")  # Print best random score

    # ---------------------------------------------------------
    # 2. GREEDY ALGORITHM (ALL POSSIBLE STARTS)
    # ---------------------------------------------------------
    print("\n2. Running GREEDY ALGORITHM (All start nodes)...")  # Print section header
    greedy_scores = []  # Initialize list for greedy scores
    for i in range(len(cities)):  # Loop through each city as a starting point
        # Her şehirden başlatıp sonucu kaydediyoruz
        sol = solve_greedy(cities, start_index=i)  # Run greedy from index 'i'
        greedy_scores.append(calculate_fitness(sol))  # Calculate and save fitness

    g_best_5 = sorted(greedy_scores)[:5]  # En iyi 5 sonucu al - Get top 5 best scores
    g_mean = statistics.mean(greedy_scores)  # Calculate mean of greedy scores
    g_stdev = statistics.stdev(greedy_scores)  # Calculate standard deviation
    g_variance = statistics.variance(greedy_scores)  # Calculate variance

    print(f"   Done. Best Greedy: {g_best_5[0]:.2f}")  # Print best greedy score

    # ---------------------------------------------------------
    # 3. GENETIC ALGORITHM (10 RUNS)
    # ---------------------------------------------------------
    print("\n3. Running GENETIC ALGORITHM (10 Runs - This may take time)...")  # Print section header
    ga_scores = []  # Initialize list for GA scores

    best_params = {"pop_size": 150, "iterations": 2000, "greedy_count": 20, "mutation_rate": 0.1, "crossover_rate": 0.85} # Set parameters

    for run in range(1, 11):  # Loop 10 times
        print(f"   Run {run}/10...", end="\r")  # Print current run number (updates same line)
        _, best_score, _ = solve_tsp_genetic(cities, **best_params)  # Run GA with params
        ga_scores.append(best_score)  # Save the best score from this run

    print(f"   Done. Best GA: {min(ga_scores):.2f}            ")  # Print global best GA score

    ga_mean = statistics.mean(ga_scores)  # Calculate mean of GA scores
    ga_stdev = statistics.stdev(ga_scores)  # Calculate standard deviation
    ga_variance = statistics.variance(ga_scores)  # Calculate variance

    # ---------------------------------------------------------
    # 4. PRINTING THE TABLE
    # ---------------------------------------------------------
    print("\n" + "=" * 70)  # Print table top border
    print(f"{'METRIC':<25} | {'RANDOM (1000)':<15} | {'GREEDY (All)':<15} | {'GENETIC (10)':<15}")  # Print table headers
    print("-" * 70)  # Print separator
    print(f"{'Best Score':<25} | {r_best:<15.2f} | {g_best_5[0]:<15.2f} | {min(ga_scores):<15.2f}") # Print best scores row
    print(f"{'Mean (Average)':<25} | {r_mean:<15.2f} | {g_mean:<15.2f} | {ga_mean:<15.2f}")  # Print mean row
    print(f"{'Standard Deviation':<25} | {r_stdev:<15.2f} | {g_stdev:<15.2f} | {ga_stdev:<15.2f}") # Print stdev row
    print(f"{'Variance':<25} | {r_variance:<15.2f} | {g_variance:<15.2f} | {ga_variance:<15.2f}") # Print variance row
    print("-" * 70)  # Print table bottom border

    print("\n>>> DETAILED RESULTS FOR REPORT:")  # Print details header
    print("Greedy Best 5 Results:", [f"{s:.2f}" for s in g_best_5])  # Print top 5 greedy results
    print("Genetic Algorithm 10 Runs:", [f"{s:.2f}" for s in ga_scores])  # Print all 10 GA results
    print("=" * 70)  # Print footer line


# --- MAIN BLOCK: TESTING REQUIREMENTS ---
if __name__ == "__main__":  # Main entry point check

    # Madde 1.c: Test on at least two different files
    files_to_test = ["berlin11_modified.tsp", "berlin52.tsp" ,"kroA100.tsp","kroA150.tsp"] # List of files to process
    current_cities = []  # Placeholder for loaded cities
    selected_filename = "none"  # Placeholder for filename
    while True:  # Infinite loop for main menu
        print("\n" + "=" * 50)  # Print menu header
        print("          Project Control MENU")
        print("=" * 50)
        print("1. Part 1")  # Option 1
        print("2. Part 2")  # Option 2
        print("3. Part 3")  # Option 3
        print("4. Part 4")  # Option 4
        print("5. Part 5")  # Option 5
        print("9. Part FINAL REPORT DATA GENERATOR")  # Option 9
        print("0. Exit")  # Option 0
        print("=" * 50)

        choice = input("please select an option: ")  # Get user input
        if choice =="1":  # If user chose 1
            for file_name in files_to_test:  # Iterate through test files
                print(f"\n{'-' * 10} TEST FILE: {file_name} {'-' * 10}")  # Print file header (ASCII compliant)

                # 1. PARSER TEST
                cities = parse_tsp_file(file_name)  # Test parsing function (changed variable name from 'sehirler' to 'cities')

                if not cities:  # If parsing failed
                    print(f"-> This file is being passed because {file_name} could not be loaded.")  # Skip
                    continue

                # 2. Distance Func TEST 2.article
                if len(cities) >= 2:  # Check if there are at least 2 cities
                    c1 = cities[0]  # Get first city
                    c2 = cities[1]  # Get second city
                    dist = calculate_distance(c1, c2)  # Calculate distance
                    print(f"Item 2 (Distance Test): Distance between City {c1.id} and City {c2.id} = {dist:.4f}") # Print result

                # 3. RANDOM SOLUTION AND REPEAT CHECK (article 4 ve 4.a)
                random_sol = create_random_solution(cities)  # Generate random solution
                ids = [c.id for c in random_sol][:5]  # Get IDs of first 5 cities
                print(f"Item 4 (Random Solution - Top 5): {ids}")  # Print sample IDs

                unique_check = set(city.id for city in random_sol)  # Create set of unique IDs
                if len(random_sol) == len(cities) and len(unique_check) == len(cities): # Verify solution validity
                    print(
                        "Item 4.a (Verification): PASSED. There is no repetition and all cities are available in the list.")
                else:
                    print("Item 4.a (Verification): ERROR! Missing city or repeat.")  # Report error

                # 4. FITNESS calculate
                fitness = calculate_fitness(random_sol)  # Calculate fitness of random solution
                print(f"Fitness: {fitness:.4f}")  # Print fitness

            print("\n=== ALL TESTS ARE COMPLETED ===")  # End of Part 1

        elif choice == "2":  # If user chose 2
            while True:  # Sub-menu loop
                print("1. (5. 6.)fitness and info functions")
                print("2. (7.)greedy vs random")
                print("3. (8.)Best Spawn point") # Corrected 'Spawm' to 'Spawn'
                print("4. (9.)POPULATION GENERATION ")
                print("0. Return to Main Menu")
                sub_choice = input("Part 2 please select an option: ")  # Get sub-menu input
                if sub_choice == "1":  # Test info/fitness
                    print("\n[TEST: INFO & FITNESS]")
                    for file_name in files_to_test:
                        print(f"\nFile: {file_name}") # Changed 'Dosya' to 'File'
                        cities = parse_tsp_file(file_name) # Changed 'sehirler' to 'cities'
                        if cities:
                            rand_sol = create_random_solution(cities)
                            # info func article 6
                            info(rand_sol)  # Call info function

                    # ---  GREEDY vs RANDOM (art 7) ---
                elif sub_choice == "2":  # Compare Greedy vs Random
                    print("\n[TEST: GREEDY vs RANDOM]")
                    for file_name in files_to_test:
                        print(f"\nFile: {file_name}")
                        cities = parse_tsp_file(file_name)
                        if cities:
                            # Random Score
                            r_sol = create_random_solution(cities)
                            r_score = calculate_fitness(r_sol)

                            #1 Greedy Score
                            g_sol = solve_greedy(cities, start_index=0)
                            g_score = calculate_fitness(g_sol)

                            print(f"Random Score: {r_score:.4f}") # Changed 'Skor' to 'Score'
                            print(f"Greedy Score: {g_score:.4f}")

                            if g_score < r_score:
                                print(f"-> Greedy algorithm {r_score - g_score:.2f} score is better!")

                    # --- ITERATIVE GREEDY (ITEM 8) ---
                elif sub_choice == "3":  # Find best start point for Greedy
                    print("\n[TEST: THE BEST STARTING POINT (ITEM 8)]")
                    for file_name in files_to_test:
                        print(f"\n>>> File scanned: {file_name}") # Fixed 'Fıle' to 'File'
                        cities = parse_tsp_file(file_name)
                        if not cities: continue

                        best_score = float('inf')
                        best_start_node = -1

                        for i in range(len(cities)):  # Loop through all nodes
                            current_greedy = solve_greedy(cities, start_index=i) #start_index = i It tries everything
                            # in turn as a starting point.
                            score = calculate_fitness(current_greedy)
                            print(f"\n--- Start Node Index: {i} ---")
                            info(current_greedy)

                            if score < best_score:
                                best_score = score
                                best_start_node = i

                        print(f"\n*** Result ({file_name}) ***")
                        print(f"Best Starting City Index: {best_start_node}")
                        print(f"Best Score: {best_score:.4f}")

                elif sub_choice =="4":  # Population generation test
                    print("\n Item 9: POPULATION GENERATION")
                    cities = parse_tsp_file("berlin52.tsp")

                    if cities:
                        greedy_reference = solve_greedy(cities, 0)
                        greed_score = calculate_fitness(greedy_reference)
                        print(f"Greedy Score: {greed_score:.4f}")
                        print("-" * 30)

                        #1. 100 random solutution creater
                        saved_population = [] #reset list
                        best_random_score = float('inf')
                        print("100 random solution created")
                        for i in range(100):
                            sol = create_random_solution(cities)
                            fit = calculate_fitness(sol)
                            saved_population.append(sol)
                            if i <5:
                                print(f"Random Sol{i+1}: Fitness = {fit:.4f}")

                            if fit < best_random_score:
                                best_random_score = fit

                        print ("\n(The other 95 are hidden.")
                        print("-"  *30)
                        print(f"best random score: {best_random_score:.4f}")
                        print(f"Greedy Score: {greed_score:.4f}")
                        diff = best_random_score - greed_score
                        print(f"difference: {diff:.4f}\n")

                elif sub_choice == "0":  # Back to main menu
                    print("Exiting the part 2")
                    break

        elif choice == "3":  # Part 3: GA Mechanics
            print("\n" + "-" * 50)
            print("     PART 3: POPULATION & GENETIC ALGORITHM")
            print("-" * 50)
            print("1. Run Full Test (Tasks 11, 12, 13, 14, 15)")
            print("0. Return to Main Menu")

            suf_choice = input("Please select an option: ")

            if suf_choice == "1":
                print("\n[PART 3 INTEGRATED TEST STARTING...]")

                for file_name in files_to_test:
                    cities = parse_tsp_file(file_name)
                    if not cities: continue

                    print(f"\n" + "=" * 40)
                    print(f" FILE: {file_name}")
                    print("=" * 40)

                    # --- ARTICLE 11 & 12: Population creater ---
                    print(">> [Task 11-12] Creating Initial Population...")
                    # 50 individuals, 5 greedy, 45 random
                    my_pop = create_initial_population(cities, pop_size=50, greedy_count=5)

                    # Article 11
                    print(f"   Population Type: {type(my_pop)} (Correct)")
                    print(f"   Population Size: {len(my_pop)}")

                    # --- Article 13: Statictic (Info) ---
                    print("\n>> [Task 13] Population Statistics:")
                    info_population(my_pop)  # Show stats
                    # Bu fonksiyon Best, Worst, Average, Median basacak.

                    # --- Article 14: ---
                    print("\n>> [Task 14] Tournament Selection Test:")
                    # show the tournament system working
                    print("   Running selection 5 times to see who wins...")

                    parent1 = tournament_selection(my_pop,5)  # Select parent 1
                    parent2 = tournament_selection(my_pop,5)  # Select parent 2

                    score_p1 =calculate_fitness(parent1)  # Score P1
                    score_p2 =calculate_fitness(parent2)  # Score P2

                    print(f"   Parent 1 Score: {score_p1:.2f}")
                    print(f"   Parent 2 Score: {score_p2:.2f}")

                    #--- Crossover 15. artc
                    print("\n>> [task15] crossover (Ordered Crossover)")
                    # We breed two parents and produce a child.
                    child = ordered_crossover(parent1, parent2)  # Create child
                    score_child = calculate_fitness(child)  # Score child
                    # Let's check the child (Is there a mistake? Is the number of cities correct?)
                    print(f"   {'Parent 1 Score':<20} : {score_p1:.4f}")
                    print(f"   {'Parent 2 Score':<20} : {score_p2:.4f}")
                    print(f"   {'Child Score':<20} : {score_child:.4f}")
                    print("-" * 45)

                    best_parent = min(score_p1, score_p2)  # Get best parent score
                    if score_child < best_parent:
                        diff = best_parent - score_child
                        print(f"   RESULT: Child scored {diff:.2f} points better than parents.!")
                    else:
                        diff = score_child - best_parent
                        print(f"   RESULT: Normal. Child is {diff:.2f} points worse than parents..")
                        print("    (This is an expected situation and can be corrected with mutation.)")

                    print("\n   Test Finished for this file.")


        elif choice == "4":  # Part 4: Mutation and Evolution
            print("\n" + "-" * 50)
            print("     PART 4: MUTATION")
            print("-" * 50)
            print("1. Test Mutation (Task 16)")
            print("2. Test One Epoch (Jump from Generation 0 to 1) (task 17)")
            print("3. Final Genetic Algorithm (Task 18)")
            print("0. Return to Main Menu")
            sub_choice = input("Select: ")
            if sub_choice == "1":  # Test mutation logic
                print("\n[ARTICLE 16: Mutation Test]")

                for file_name in files_to_test:
                    cities = parse_tsp_file(file_name)
                    if not cities: continue
                    print(f"\n>>> File: {file_name}")

                    pop = create_initial_population(cities, pop_size=50, greedy_count=5)
                    p1=tournament_selection(pop,5)
                    p2=tournament_selection(pop,5)
                    child=ordered_crossover(p1,p2)

                    score_before = calculate_fitness(child)  # Score before mutation
                    print(f"Child Score (Before Mutation): {score_before:.2f}")

                    #16.article
                    print("Applying Mutation(Rate: 1.0 -> %100 Definite Mutation) . . .")
                    mutated_child = inversion_mutation(child , mutation_rate=1.0) # Force mutation
                    score_after = calculate_fitness(mutated_child)  # Score after mutation
                    print(f"Child Score (after Mutation): {score_after:.2f}")

                    if child != mutated_child:  # Verify change
                        print("SUCCESS: Mutation changed the route structure! (Confirmed)")
                        if score_before != score_after:
                            diff = score_before - score_after
                            if diff > 0:
                                print(f"-> Result: Improved by {diff:.2f}")
                            else:
                                print(f"-> Result: Worse by {abs(diff):.2f}")
                        else:
                            print("-> Result: Score remained same (Coincidence), but route changed.")
                    else:
                        print("FAIL: The list did not change at all.")

            if sub_choice == "2":  # Test one full generation epoch
                print("\n[ARTICLE 17: One Epoch Test]")
                for file_name in files_to_test:
                    cities = parse_tsp_file(file_name)
                    if not cities: continue
                    print(f"\n>>> File: {file_name}")

                    #1. EPOCH 0: Initial Population
                    print("Creating Generation 0 (Initial) . . .")
                    pop_gen0 = create_initial_population(cities,50,5) #greedy 5 ,45 random

                    print("\n--- Generation 0 stats ---")
                    info_population(pop_gen0)

                    #Let's save the best score for comparison.
                    best_gen0 = min([calculate_fitness(s) for s in pop_gen0])

                    # 2. EPOCH 1: Evolutionary Transition (Function of Article 17)
                    print("\nRunning Evolution (creating Generation1). . . ")
                    # Let's assume the crossover rate is 80% and the mutation rate is 10%.
                    pop_gen1 = create_new_generation(pop_gen0 , mutation_rate=0.1, crossover_rate=0.8)

                    print("\n--- GENERATION 1 STATS ---")
                    info_population(pop_gen1)

                    best_gen1 = min([calculate_fitness(s) for s in pop_gen1])
                    #result
                    print("-" *40)
                    if len(pop_gen1) == len(pop_gen0):  # Check population stability
                        print("SUCCESS: Population size preserved.")
                    else:
                        print("FAIL: Population size changed!")

                    diff = best_gen0 - best_gen1
                    if diff > 0 :
                        print(f"EVOLUTION WORKING: Best score omproved by {diff:.2f} points.")
                    elif diff < 0:
                        print(f"WARNING: Best score got worse by {abs(diff):.2f} points.")
                        print("(This happens sometimes in early generations without Elitism. Keep going!)")
                    else:
                        print("STAGNATION: Best score remained exactly the same.")

            if sub_choice == "3":  # Run Full GA
                print("\n" + "=" * 50)
                print("     FINAL RUN AND GRAPHICS")
                print("=" * 50)

                for file_name in files_to_test:
                    cities = parse_tsp_file(file_name)
                    if not cities: continue
                    print(f"\n>>> File: {file_name}")

                    # Calculate your Greedy Score for reference
                    greedy_sol = solve_greedy(cities, 0)
                    greedy_score = calculate_fitness(greedy_sol)
                    print(f"Greedy Score to beat {greedy_score:.4f}")
                    print("Running Genetic Algorithm . . .")

                    #starting algorthm
                    final_route , final_score ,history = solve_tsp_genetic(cities,pop_size=100 , iterations=3000,greedy_count=10) # Run algo
                    print("-" * 30)
                    print(f"FINAL RESULT ({file_name}):")
                    print(f"Greedy Score : {greedy_score:.4f}")
                    print(f"Genetic Score: {final_score:.4f}")

                    diff = greedy_score - final_score
                    if  diff > 0 :
                        print(f"SUCCESS: Genetic Algorithm is better by {diff:.2f} points!")
                    else:
                        print(f"RESULT: Genetic is worse by {abs(diff):.2f} points. (Try increasing iterations)")
                    print("-" * 30)

                    print("Displaying Graphs")
                    plot_results(history , final_route)  # Show plot

        elif choice == "5":  # Part 5: Parameter Compare
            print("\n" + "=" * 50)
            print("     PART 5: COMPARE PARAMETERS")
            print("=" * 50)

            for file_name in files_to_test:
                print(f"Loading {file_name} for comparison...")
                cities = parse_tsp_file(file_name)

                if cities:
                    run_parameter_comparison(cities) # Run comparison

        elif choice == "9":  # Final Stats Report
            print("\n" + "#" * 60)
            print("     PART 3: FINAL STATISTICAL REPORT (ALL FILES)")
            print("#" * 60)

            for file_name in files_to_test:
                print(f"\n\n>>> PROCESSING FILE: {file_name} <<<")
                cities = parse_tsp_file(file_name)

                if not cities:
                    print(f"Skipping {file_name} (Could not load).")
                    continue

                generate_final_report_stats(cities)  # Generate report

                print(f">>> REPORT FOR {file_name} COMPLETED.")
                print("-" * 60)

        elif choice == "0":  # Exit option
            print("Exiting the program ... Have a nice day")
            break
        else:
            print("!!! Invalid selection")  # Invalid input handler