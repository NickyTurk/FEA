import unittest

from refactoring.MOO.MOEA import *
from refactoring.utilities.util import add_method


class TestGeneral(unittest.TestCase):
    def setUp(self) -> None:
        self.objectives = 3
        population_size = 10
        dimensions = 10
        upper_value_limit = 50
        self.moea = MOEA(population_size, dimensions, combinatorial_options=[], value_range=[0, upper_value_limit])

    def test_initialize_population(self):
        curr_population, initial_solution = self.moea.initialize_population()
        self.assertEqual(self.moea.dimensions, len(initial_solution))
        self.assertEqual(self.moea.population_size, len(curr_population))
        self.assertEqual(curr_population[0].variables, initial_solution)

        random_member_to_check = random.randint(1, self.moea.population_size-1)
        self.assertNotEqual(curr_population[random_member_to_check].variables, initial_solution)

    def test_initialize_combinatorial_population(self):
        combinatorial_options = [20, 40, 60]
        self.moea.combinatorial_values = combinatorial_options
        curr_population, initial_solution = self.moea.initialize_population()

        random_variable_to_check = random.randint(0, self.moea.dimensions-1)
        self.assertIn(initial_solution[random_variable_to_check], combinatorial_options)

        random_member_to_check = random.randint(0, self.moea.population_size-1)
        random_variable_to_check = random.randint(0, self.moea.dimensions-1)
        self.assertIn(curr_population[random_member_to_check].variables[random_variable_to_check], combinatorial_options)

    def test_initialize_continuous_population(self):
        curr_population, initial_solution = self.moea.initialize_population()

        random_variable_to_check = random.randint(0, self.moea.dimensions-1)
        self.assertLessEqual(initial_solution[random_variable_to_check], self.moea.value_range[1])

        random_member_to_check = random.randint(0, self.moea.population_size-1)
        random_variable_to_check = random.randint(0, self.moea.dimensions-1)
        self.assertLessEqual(curr_population[random_member_to_check].variables[random_variable_to_check], self.moea.value_range[1])


class TestNSGA2(unittest.TestCase):
    def setUp(self) -> None:
        dimensions = 10
        objectives = 3

        @add_method(NSGA2)
        def calc_fitness(variables, gs=None, factor=None):
            fitnesses = []
            if gs is not None and factor is not None:
                full_solution = [x for x in gs.variables]
                for i, x in zip(factor, variables):
                    full_solution[i] = x
            else:
                full_solution = variables
            for i in range(objectives):
                adjusted_variables = full_solution[i:len(full_solution):objectives]
                fitnesses.append(sum(adjusted_variables))
            return fitnesses
        self.nsga = NSGA2(dimensions=dimensions, population_size=10, ea_runs=10)
        self.curr_population, self.initial_solution = self.nsga.initialize_population()

    def test_NSGA2_sorting(self):
        og_population = [x.fitness for x in self.curr_population]
        sorted_population = self.nsga.sorting_mechanism(self.curr_population)
        sorted_population = [x.fitness for x in sorted_population]
        self.assertNotEqual(og_population, sorted_population)

    def test_generation_selection(self):
        children = self.nsga.ea.create_offspring(self.curr_population)
        self.nsga.curr_population.extend(
            [PopulationMember(c, self.nsga.calc_fitness(c)) for c in children])
        self.nsga.select_new_generation()
        self.assertEqual(len(self.curr_population), self.nsga.population_size)

    def test_replace_worst(self):
        pass


class TestSPEA2(unittest.TestCase):
    def setUp(self) -> None:
        dim = 10
        n_obj = 3


if __name__ == '__main__':
    unittest.main()