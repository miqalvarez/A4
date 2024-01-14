import random
import tsplib95
import numpy as np
import matplotlib.pyplot as plt

# Generate an initial population
def generate_population(size, num_nodes):
    return [random.sample(range(1, num_nodes + 1), num_nodes) for _ in range(size)]

# Evaluate the fitness of a chromosome (route)
def evaluate_fitness(route, problem):
    total_distance = 0
    # Compute the total distance
    for i in range(len(route) - 1):
        total_distance += tsplib95.distances.euclidean(problem.get_display(route[i]), problem.get_display(route[i + 1]))
        

    return total_distance



# Perform crossover on two parents based on the specified method
def crossover(parent1, parent2, method):
    crossover_point = random.randint(1, len(parent1) - 1)
    
    if method == 1:
        child1 = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [city for city in parent1 if city not in parent2[:crossover_point]]
        
        # Ensure the route is cyclic
        child1.append(child1[0])
        child2.append(child2[0])
    
    elif method == 2:
        # Two-point crossover
        crossover_point2 = random.randint(1, len(parent1) - 1)
        while crossover_point == crossover_point2:
            crossover_point2 = random.randint(1, len(parent1) - 1)

        if crossover_point > crossover_point2:
            crossover_point, crossover_point2 = crossover_point2, crossover_point

        child1 = parent1[:crossover_point] + parent2[crossover_point:crossover_point2] + parent1[crossover_point2:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:crossover_point2] + parent2[crossover_point2:]

        # Ensure unique cities
        child1 = list(dict.fromkeys(child1))
        child2 = list(dict.fromkeys(child2))
        
        # Ensure the route is cyclic
        child1.append(child1[0])
        child2.append(child2[0])

    return child1, child2
# Perform mutation on a chromosome based on the specified method
def mutate(route, method):
    mutation_point1 = random.randint(0, len(route) - 1)
    mutation_point2 = random.randint(0, len(route) - 1)
    
    if method == 1:
        route[mutation_point1], route[mutation_point2] = route[mutation_point2], route[mutation_point1]
    else:
        # Alternative mutation method
        route[mutation_point1], route[mutation_point2] = route[mutation_point2], route[mutation_point1]
        route = route[::-1]  # Reverse the route
    
    return route

# Select individuals for the next generation using the specified selection method
def selection(population, fitness_values, tournament_size, method):
    if method == 1:
        selected_indices = random.sample(range(len(population)), tournament_size)
        selected_individuals = [population[i] for i in selected_indices]
        selected_fitness_values = [fitness_values[i] for i in selected_indices]
        return selected_individuals[selected_fitness_values.index(min(selected_fitness_values))]
    else:
        sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k])
        return population[sorted_indices[0]]
    
# Genetic Algorithm
def genetic_algorithm(problem_file, population_size, generations, crossover_rate, mutation_rate, tournament_size, s_method, c_method, m_method):
    tsp_instance = tsplib95.load(problem_file)
    num_nodes = tsp_instance.dimension

    # Initialize population
    population = generate_population(population_size, num_nodes)

    fitness_history = []
    for generation in range(generations):
        # Evaluate fitness of the population
        fitness_values = [evaluate_fitness(chromosome, tsp_instance) for chromosome in population]

        # Select parents for crossover using tournament selection
        parents = [selection(population, fitness_values, tournament_size, s_method) for _ in range(population_size)]

        # Create offspring through crossover
        offspring = []
        for i in range(0, population_size, 2):
            if random.random() < crossover_rate:
                child1, child2 = crossover(parents[i], parents[i + 1], c_method)
            else:
                child1, child2 = parents[i], parents[i + 1]
            offspring.extend([child1, child2])

        # Mutate offspring
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = mutate(offspring[i], m_method)

        # Replace the old population with the new population
        population = offspring

        # Print the best route in each generation
        best_route = population[np.argmin(fitness_values)]
        best_fitness = min(fitness_values)
        fitness_history.append(best_fitness)
        print(f"Generation {generation + 1}: Best Route = {best_route}, Best Fitness = {best_fitness}")

    # Return the best route found
    best_route = population[np.argmin(fitness_values)]

    # Plot the best route using get_display along with the fitness history
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fitness_history)
    plt.title("Fitness History")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    x = [tsp_instance.get_display(x)[0] for x in best_route]
    y = [tsp_instance.get_display(x)[1] for x in best_route]
    plt.subplot(1, 2, 2)
        # Mark with the number of the city each point
    for i in range(len(x)):
        plt.text(x[i], y[i], str(i + 1))
    plt.plot(x, y, 'r-')
    plt.title(f"Best Route")

    plt.show()

    return best_route, min(fitness_values)

# Example usage
problem_file = "Problems/att48.tsp"
population_size = 400
generations = 800
crossover_rate = 0.5
mutation_rate = 0.5
tournament_size = 5

# PMX, 2PCX
c_method = 1

# Swap, Inverse
m_method = 1

# Tournament selection, elitism
s_method = 1

best_route, best_fitness = genetic_algorithm(problem_file, population_size, generations, crossover_rate, mutation_rate, tournament_size, s_method, c_method, m_method)

print(f"Best Route: {best_route}")
print(f"Best Fitness: {best_fitness}")