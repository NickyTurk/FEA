import numpy as np
import random

class GA:
    """
    Basic GA methods: tournament selection, mutation and crossover.
    To be reused in any MOO EA method.
    """
    def __init__(self, dimensions = 100, population_size=200, tournament_size=5, mutation_rate=0.1, crossover_rate=0.90,
                 ga_runs=100, mutation_type="swap", crossover_type="multi", offspring_size=100):
        self.name = 'genetic algorithm'
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.dimensions = dimensions
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.ga_runs = ga_runs
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
    
    def tournament_selection(self, population):
        """
        Entities in population is of namedTuple format: PopulationMember('variables', 'fitness')
        Picks a number of solutions and finds and returns the solution dict with the best score.
        """
        i = 0
        chosen_solution = None
        curr_fitness = None
        while i < self.tournament_size:
            rand = random.randint(0, len(population) - 1)
            temp = population[rand]
            if curr_fitness:
                if temp.fitness <= curr_fitness:
                    chosen_solution = temp
                    curr_fitness = temp.fitness
            else:
                curr_fitness = temp.fitness
            i += 1
        return chosen_solution

    def mutate(self, original_solution):
        """
        Type = Swap, scramble or inversion
        """
        rand = random.random()
        if rand < self.mutation_rate:
            _solution = [x for x in original_solution]
            if self.mutation_type == "swap":
                # for index, gene in enumerate(solution_dict):
                numbers = list(range(0, self.dimensions))
                index_1 = random.choice(numbers)
                numbers.remove(index_1)
                index_2 = random.choice(numbers)
                _solution[index_1] = original_solution[index_2]
                _solution[index_2] = original_solution[index_1]
            elif self.mutation_type == "scramble":
                index_1 = random.choice(list(range(0, self.dimensions)))
                numbers = list(range(0, self.dimensions))  # numbers to choose next index from
                numbers.remove(index_1)  # remove already chosen index from these numbers
                index_2 = random.choice(numbers)
                if index_1 > index_2:
                    max_index = index_1
                    min_index = index_2
                else:
                    max_index = index_2
                    min_index = index_1
                temp_list = original_solution[min_index:max_index]
                np.random.shuffle(temp_list)
                _solution[min_index:max_index] = temp_list
            return _solution
        else:
            return original_solution

    def crossover(self, first_solution, second_solution):
        """
        Picks two indices and selects everything between those points.
        Type = single, multi or uniform
        """
        if random.random() < self.crossover_rate:
            _first_solution = [x for x in first_solution]
            _second_solution = [x for x in second_solution]
            index_1 = random.randint(0, self.dimensions - 1)
            index_2 = random.randint(0, self.dimensions - 1)
            if index_1 > index_2:
                max_index = index_1
                min_index = index_2
            else:
                max_index = index_2
                min_index = index_1
            max_index = max_index + 1
            _first_solution[min_index:max_index] = second_solution[min_index:max_index]
            _second_solution[min_index:max_index] = first_solution[min_index:max_index]

            return [_first_solution, _second_solution]
        else:
            return []
    
    def create_offspring(self, curr_population):
        j = 0
        children = []
        population = [x for x in curr_population]
        while j < self.offspring_size:
            first_solution = self.tournament_selection(population)
            population.remove(first_solution)
            second_solution = self.tournament_selection(population)
            population.append(first_solution)
            first_solution = [x for x in first_solution.variables]
            second_solution = [x for x in second_solution.variables]
            crossed_over = self.crossover(first_solution, second_solution)
            if crossed_over:
                child1 = crossed_over[0]
                child2 = crossed_over[1]
            else:
                child1 = [x for x in first_solution]
                child2 = [x for x in second_solution]
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            children.append(child1)
            children.append(child2)
            j += 1
        return children
    
    def run(self):
        """
        run a basic GA
        """
        pass
