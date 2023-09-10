import random
import math
import time
import heapq
import numpy as np

class SchoolBusStop:
    def __init__(self, stop_id, x, y, num_students):
        self.stop_id = stop_id
        self.x = x
        self.y = y
        self.num_students = num_students


class School:
    def __init__(self):
        self.stops = []
        self.school_x = 0
        self.school_y = 0

    def add_bus_stop(self, stop_id, x, y, num_students):
        stop = SchoolBusStop(stop_id, x, y, num_students)
        self.stops.append(stop)

    def generate_random_bus_stops(self, num_stops):
        for stop_id in range(1, num_stops + 1):
            x = random.randint(-10, 10)  # Assuming the school is within a 20x20 coordinate grid
            y = random.randint(-10, 10)
            num_students = random.randint(1, 25)
            self.add_bus_stop(stop_id, x, y, num_students)


#def calculate_distance(stop1, stop2):
    #return abs(stop1.x - stop2.x) + abs(stop1.y - stop2.y)


def total_distance(solution, school):
    total_distance = 0
    for i in range(len(solution) - 1):
        stop1 = school.stops[solution[i] - 1]
        stop2 = school.stops[solution[i + 1] - 1]
        total_distance += calculate_distance(stop1, stop2)
    # Add distance from the last stop back to school
    last_stop = school.stops[solution[-1] - 1]
    school_location = SchoolBusStop(0, school.school_x, school.school_y, 0)
    total_distance += calculate_distance(last_stop, school_location)
    return total_distance


def simulate_annealing(school, initial_solution, temperature=10000, cooling_rate=0.003):
    current_solution = initial_solution
    best_solution = initial_solution
    current_distance = total_distance(initial_solution, school)
    best_distance = current_distance

    while temperature > 1:
        new_solution = current_solution.copy()
        index1, index2 = random.sample(range(1, len(new_solution) - 1), 2)
        new_solution[index1], new_solution[index2] = new_solution[index2], new_solution[index1]

        new_distance = total_distance(new_solution, school)
        delta_distance = new_distance - current_distance

        if delta_distance < 0 or random.random() < math.exp(-delta_distance / temperature):
            current_solution = new_solution
            current_distance = new_distance

        if new_distance < best_distance:
            best_solution = new_solution
            best_distance = new_distance

        temperature *= 1 - cooling_rate

    return best_solution, best_distance


def save_bus_stops_to_file(school, filename):
    with open(filename, "w") as file:
        for stop in school.stops:
            file.write(f"{stop.stop_id},{stop.x},{stop.y},{stop.num_students}\n")


def save_solution_to_file(solution, filename):
    with open(filename, "w") as file:
        for stop_id in solution:
            file.write(f"{stop_id}\n")


def load_bus_stops_from_file(filename):
    school = School()
    with open(filename, "r") as file:
        for line in file:
            stop_id, x, y, num_students = map(int, line.strip().split(","))
            school.add_bus_stop(stop_id, x, y, num_students)
    return school


def load_solution_from_file(filename):
    with open(filename, "r") as file:
        solution = [int(line.strip()) for line in file]
    return solution


def get_stop_info(stop_id, school):
    stop = school.stops[stop_id - 1]
    return f"Stop {stop_id}: (X={stop.x}, Y={stop.y}), Students: {stop.num_students}"


def generate_random_solution(school):
    stops = [stop.stop_id for stop in school.stops]
    random.shuffle(stops)
    return stops


def crossover(parent1, parent2):
    # Select a random segment of the parent solutions
    segment_start = random.randint(0, len(parent1) - 1)
    segment_end = random.randint(segment_start, len(parent1))

    # Create two child solutions
    child1 = [-1] * len(parent1)
    child2 = [-1] * len(parent1)

    # Copy the selected segment from parent1 to child1, preserving the order
    for i in range(segment_start, segment_end):
        child1[i] = parent1[i]

    # Copy the remaining stops from parent2 to child1, preserving the order
    index = 0
    for i in range(len(parent2)):
        stop = parent2[i]
        if stop not in child1:
            while child1[index] != -1:
                index += 1
            child1[index] = stop

    # Repeat the process for child2, swapping parent roles
    for i in range(segment_start, segment_end):
        child2[i] = parent2[i]

    index = 0
    for i in range(len(parent1)):
        stop = parent1[i]
        if stop not in child2:
            while child2[index] != -1:
                index += 1
            child2[index] = stop

    return child1, child2


def mutate(solution, mutation_rate=0.01):
    # Apply mutation to the solution with a certain probability (mutation_rate)
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            # Swap the current stop with a randomly selected stop
            j = random.randint(0, len(solution) - 1)
            solution[i], solution[j] = solution[j], solution[i]
    return solution


def genetic_algorithm(school, population_size=100, generations=1000):
    population = [generate_random_solution(school) for _ in range(population_size)]
    best_solution = None
    best_fitness = float('inf')

    for generation in range(generations):
        population.sort(key=lambda x: total_distance(x, school))
        parents = population[:population_size // 2]

        new_population = parents.copy()

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)

            mutate(child1)  # Mutate child1 directly, avoid creating a new list
            mutate(child2)  # Mutate child2 directly, avoid creating a new list

            new_population.extend([child1, child2])

        population = new_population

        # Calculate fitness only once for the best solution
        best_gen_fitness = total_distance(population[0], school)
        if best_gen_fitness < best_fitness:
            best_solution = population[0]
            best_fitness = best_gen_fitness

    return best_solution, best_fitness


def calculate_distance(stop1, stop2):
    # Assuming that stop1 and stop2 have attributes x and y representing their coordinates
    return math.sqrt((stop1.x - stop2.x) ** 2 + (stop1.y - stop2.y) ** 2)



def calculate_distance(coord1, coord2):
    if isinstance(coord1, SchoolBusStop):
        coord1 = (coord1.x, coord1.y)
    if isinstance(coord2, SchoolBusStop):
        coord2 = (coord2.x, coord2.y)

    distance = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
    return distance




def calculate_distance(coord1, coord2):
    distance = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
    return distance


def nearest_neighbor_solution(school):

    school_location = (0, 0)  # Assuming the school is the first stop

    best_route = None
    best_distance = float('inf')
    max_capacity = 25

    for start_stop_id in range(len(school.stops)):
        current_route = [start_stop_id]
        current_distance = calculate_distance(school_location, school.stops[start_stop_id])
        current_capacity = school.stops[start_stop_id].num_students

        unvisited_stops = set(range(len(school.stops)))
        unvisited_stops.remove(start_stop_id)

        while unvisited_stops:
            current_stop_id = current_route[-1]
            next_stop_id = None
            min_distance = float('inf')

            for neighbor_id in unvisited_stops:
                # If the capacity would be exceeded, skip it
                if current_capacity + school.stops[neighbor_id].num_students > max_capacity:
                    continue

                distance = calculate_distance(school.stops[current_stop_id], school.stops[neighbor_id])
                if distance < min_distance:
                    next_stop_id = neighbor_id
                    min_distance = distance

            if next_stop_id is not None:

                current_route.append(next_stop_id)
                current_distance += min_distance
                current_capacity += school.stops[next_stop_id].num_students
                unvisited_stops.remove(next_stop_id)

            else:


                # Calculate distance to return to the school
                return_to_school_distance = calculate_distance(school.stops[current_route[-1]], (0,0))
                current_distance += return_to_school_distance


                current_capacity=0
                # Clear out the route
                min_distance=float('inf')
                for neighbor_id in unvisited_stops:
                    # If the capacity would be exceeded, skip it
                    if school.stops[neighbor_id].num_students > max_capacity:
                        continue

                    distance = calculate_distance((0, 0), school.stops[neighbor_id])
                    if distance < min_distance:
                        next_start_id = neighbor_id
                        min_distance = distance

                current_route.append(next_start_id)
                #current_distance += min_distance
                current_capacity = school.stops[next_start_id].num_students
                unvisited_stops.remove(next_start_id)


        if current_distance < best_distance:
            best_distance = current_distance
            best_route = current_route.copy()
    return best_distance, best_route


def calculate_savings(coord1, coord2, school_location):
    if isinstance(coord1, SchoolBusStop):
        coord1 = (coord1.x, coord1.y)
    if isinstance(coord2, SchoolBusStop):
        coord2 = (coord2.x, coord2.y)

    savings = calculate_distance(coord1, school_location) + calculate_distance(coord2, school_location) - calculate_distance(coord1, coord2)
    return savings

def calculate_distance(coord1, coord2):
    if isinstance(coord1, SchoolBusStop):
        coord1 = (coord1.x, coord1.y)
    if isinstance(coord2, SchoolBusStop):
        coord2 = (coord2.x, coord2.y)

    distance = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
    return distance
def clarke_wright_savings_algorithm(school, capacity=25):
    print("\nExecuting Clarke-Wright Savings Algorithm...")
    school_location = (school.school_x, school.school_y)
    savings_list = []

    # Calculate savings for each pair of stops
    for i in range(len(school.stops)):
        for j in range(i + 1, len(school.stops)):
            stop1 = school.stops[i]
            stop2 = school.stops[j]
            savings = calculate_savings(stop1, stop2, school_location)
            savings_list.append((i, j, savings))

    # Sort the savings_list in descending order of savings
    savings_list.sort(key=lambda x: x[2], reverse=True)

    # Initialize all stops in separate routes
    routes = [[i] for i in range(len(school.stops))]

    while savings_list:
        stop1, stop2, _ = savings_list.pop(0)  # Pop the highest savings value

        route_index_1, route_index_2 = None, None

        # Find the routes to which stop1 and stop2 belong
        for i, route in enumerate(routes):
            if stop1 in route:
                route_index_1 = i
            if stop2 in route:
                route_index_2 = i

        # Check if the stops are not in the same route and if the capacity allows the merge
        if route_index_1 != route_index_2 and len(routes[route_index_1]) + len(routes[route_index_2]) <= capacity:
            # Merge route 2 into route 1
            merged_route = routes[route_index_1] + routes[route_index_2][::-1]
            if calculate_distance(school.stops[merged_route[0]], school.stops[merged_route[-1]]) <= capacity:
                routes[route_index_1] = merged_route
                del routes[route_index_2]

    # Combine the routes into a single list
    best_route = [stop_id for route in routes for stop_id in route]

    # Calculate total distance for the best route
    best_distance = 0
    for i in range(len(best_route) - 1):
        stop1 = school.stops[best_route[i]]
        stop2 = school.stops[best_route[i + 1]]
        best_distance += calculate_distance(stop1, stop2)

    # Print the best route with capacity constraints
    print("\nClarke-Wright Savings Algorithm - Best route:")
    current_students = 0
    route = []
    for stop_id in best_route:
        stop = school.stops[stop_id]
        current_students += stop.num_students

        if current_students > capacity:
            # If adding the stop exceeds capacity, print the current route and start a new route
            print_route(route, school)
            route = []
            current_students = stop.num_students

        route.append(stop)

    print_route(route, school)  # Print the final route
    print("Clarke-Wright Savings Algorithm - Total Distance:", best_distance)
    return best_distance, best_route




def print_route(route, school):
    for i, stop in enumerate(route):
        print_stop_info(stop, school, end=" -> " if i < len(route) - 1 else "\n")

def print_stop_info(stop, school, end="\n"):
    print(f"Stop {stop.stop_id}: (X={stop.x}, Y={stop.y}), Students: {stop.num_students}", end=end)

# Rest of the code remains the same as before





def main():
    school = School()
    school.generate_random_bus_stops(100)  # Generate bus stops

    print("Generated bus stops:")
    for stop in school.stops:
        print(f"Stop {stop.stop_id}: (X={stop.x}, Y={stop.y}), Students: {stop.num_students}")

    # Genetic Algorithm
    start_time = time.time()
    best_genetic_route, shortest_genetic_distance = genetic_algorithm(school, generations=1000)  # Reduced generations
    end_time = time.time()
    genetic_time = end_time - start_time

    print("\nGenetic Algorithm - Best route:")
    current_students = 0
    for i, stop_id in enumerate(best_genetic_route):
        if i > 0:
            print("->", end=" ")

        stop = school.stops[stop_id - 1]
        current_students += stop.num_students

        if current_students > 25:
            print("Back to School")
            current_students = stop.num_students
        print(get_stop_info(stop_id, school))

    print("Genetic Algorithm - Shortest Distance:", shortest_genetic_distance)
    print("Genetic Algorithm - Time taken:", genetic_time, "seconds")

    # Simulated Annealing
    initial_solution = [stop.stop_id for stop in school.stops]

    start_time = time.time()
    best_sa_route, shortest_sa_distance = simulate_annealing(school, initial_solution)
    end_time = time.time()
    sa_time = end_time - start_time

    print("\nSimulated Annealing - Best route:")
    current_students = 0
    for i, stop_id in enumerate(best_sa_route):
        if i > 0:
            print("->", end=" ")

        stop = school.stops[stop_id - 1]
        current_students += stop.num_students

        if current_students > 25:
            print("Back to School")
            current_students = stop.num_students
        print(get_stop_info(stop_id, school))

    print("Simulated Annealing - Shortest Distance:", shortest_sa_distance)
    print("Simulated Annealing - Time taken:", sa_time, "seconds")

    start_time = time.time()
    nn_distance, best_nn_route = nearest_neighbor_solution(school)
    end_time = time.time()
    nn_time = end_time - start_time

    print("\nNearest Neighbor (Graph Theory) - Best route:")
    current_students = 0
    current_students = 0
    for i, stop_id in enumerate(best_nn_route):
        if i > 0:
            print("->", end=" ")

        stop = school.stops[stop_id - 1]
        current_students += stop.num_students

        if current_students > 25:
            print("Back to School")
            current_students = stop.num_students
        print(get_stop_info(stop_id, school))
    print("Nearest Neighbor (Graph Theory) - Shortest Distance:", nn_distance)
    print("Nearest Neighbor (Graph Theory) - Time taken:", nn_time, "seconds")


    # Clarke-Wright Savings Algorithm
    start_time = time.time()
    cw_distance, cw_best_route = clarke_wright_savings_algorithm(school)
    end_time = time.time()
    cw_time = end_time - start_time

    print("Clarke-Wright Savings Algorithm - Time taken:", cw_time, "seconds")
    # Save solutions to files
    save_solution_to_file(best_genetic_route, "best_genetic_route.txt")
    save_solution_to_file(best_sa_route, "best_sa_route.txt")
    #save_solution_to_file(best_nn_route, "best_nn_route.txt")

    print("Genetic Algorithm - Shortest Distance:", shortest_genetic_distance)
    print("Genetic Algorithm - Time taken:", genetic_time, "seconds")
    print("Simulated Annealing - Shortest Distance:", shortest_sa_distance)
    print("Simulated Annealing - Time taken:", sa_time, "seconds")
    print("Nearest Neighbor (Graph Theory) - Shortest Distance:", nn_distance)
    print("Nearest Neighbor (Graph Theory) - Time taken:", nn_time, "seconds")

if __name__ == "__main__":
    main()