"""
Contains the GA class with the three basic methods: tournament_selection(population), mutation(original_individual)
and crossover(parent1, parent2).
These three methods combined form the generate_offspring() method.
Can be used a standalone algorithm using run(), or as the base-algorithm for MOO EA's or FEA's.
"""

import numpy as np
import random


class GA:
    def __init__(self, dimensions=100, population_size=200, tournament_size=5, mutation_rate=0.2, crossover_rate=0.95,
                 ga_runs=100, mutation_type="multi bitflip", crossover_type="single", offspring_size=100):
        """
        @param dimensions: Integer. Number of variables in a single individual.
        @param population_size: Integer. Number of individuals to form the population in a single generation.
        @param tournament_size: Integer. How many candidates to select a single parent from.
        @param mutation_rate: Float between 0.0 and 1.0. Chance of performing mutation on each individual.
        @param crossover_rate: Float between 0.0 and 1.0. Chance of performing crossover of the selected parent pair.
        @param ga_runs: Integer. How many generations to run the GA for.
        @param mutation_type: String 'swap', 'scramble', 'multi bitflip' or 'single bitflip'. Which type of mutation to perform.
        @param crossover_type: String 'single', or 'multi'. Which type of crossover to perform.
        @param offspring_size: Integer. How many children do you wish to create each generation.
        """
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
        idx = 0
        while i < self.tournament_size:
            rand = random.randint(0, len(population) - 1)
            temp = population[rand]
            if curr_fitness is not None:
                if temp.fitness <= curr_fitness:
                    chosen_solution = temp
                    curr_fitness = temp.fitness
                    idx = rand
            else:
                curr_fitness = temp.fitness
                chosen_solution = temp
                idx = rand
            i += 1
        return chosen_solution, idx

    def mutate(self, original_solution):
        """
        @param original_solution: The solution to be mutated.
        Type = Swap, scramble or bitflip.
        Swap = Choose two indices and swap the values.
        Scramble = Choose two indices and randomly scramble all the values between these indices.
        Single Bitflip = Randomly choose a single index and flip the value.
        Multi Bitflip = For each variable decide whether the bit will be flipped using the mutation rate.
        @return: The mutated child or a copy of the original solution if mutation was not performed.
        """
        _solution = [x for x in original_solution]
        rand = random.random()
        if rand < self.mutation_rate:
            if self.mutation_type == "swap":
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
            elif self.mutation_type == "single bitflip":
                numbers = list(range(0, self.dimensions))
                index_1 = random.choice(numbers)
                temp = _solution[index_1]
                if temp == 0:
                    _solution[index_1] = 1
                elif temp ==1:
                    _solution[index_1] = 0
            elif self.mutation_type == "multi bitflip":
                for i, x in enumerate(_solution):
                    rand = random.random()
                    if rand < self.mutation_rate:
                        if x == 0:
                            _solution[i] = 1
                        elif x == 1:
                            _solution[i] = 0

        return _solution

    def crossover(self, first_solution, second_solution):
        """
        @param first_solution: The first parent selected.
        @param second_solution: The second parent seleced.
        Crossover is performed between these parents to create offspring.
        Type = single, multi or uniform.
        Single = Single point crossover, a random index is selected and the parents are crossed based on this inex.
        Multi = Multi-point crossover, two indices are selected and the values between these indices are swapped between
                the parents.

        @return: The crossed over children, or an empty array if crossover was not performed.
        """
        if random.random() < self.crossover_rate:
            _first_solution = [x for x in first_solution]
            _second_solution = [x for x in second_solution]
            if self.crossover_type == 'multi':
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
            elif self.crossover_type == 'single':
                index_1 = random.randint(0, self.dimensions - 1)
                _first_solution[0:index_1] = second_solution[0:index_1]
                _second_solution[index_1:-1] = first_solution[index_1:-1]

            return [_first_solution, _second_solution]
        else:
            return []

    def create_offspring(self, curr_population):
        """
        @param curr_population:
        @return: The children created from the current population.
        Performs the necessary steps to create the set number of offspring.
        1. Tournament selection to select parents
        2. Crossover of the selected parents
        3. Mutation of the resulting offspring
        These steps are repeated until the length of the children set is equal to offspring_size.
        """
        j = 0
        children = []
        population = [x for x in curr_population]
        while len(children) < self.offspring_size:
            first_solution, idx1 = self.tournament_selection(population)
            population.pop(idx1)
            second_solution, idx2 = self.tournament_selection(population)
            population.insert(idx1, first_solution)
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
