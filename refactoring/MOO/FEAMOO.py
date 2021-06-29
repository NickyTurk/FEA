from ..FEA.factorevolution import FEA
from ..FEA.factorarchitecture import FactorArchitecture
import numpy as np
import itertools


class MOOSolution:
    def __init__(self):
        self.overall_fitness = np.inf
        self.variable_values = []


class FEAMOO:
    def __init__(self, function, fea_iterations, alg_iterations, pop_size, fa, base_alg, dimensions):
        self.function = function
        self.dim = dimensions
        self.nondom_archive = []
        self.pareto_set = []
        self.population = []
        self.factor_architecture = fa
        self.base_algorithm = base_alg
        self.fea_runs = fea_iterations
        self.base_alg_iterations = alg_iterations
        self.pop_size = pop_size
        self.current_iteration = 0
        self.worst_solution = []  # keep track to have a reference point for the HV indicator
        self.subpopulations = self.initialize_moo_subpopulations()

    def initialize_moo_subpopulations(self):
        fa = self.factor_architecture
        alg = self.base_algorithm
        return [alg(self.base_alg_iterations, self.pop_size, self.function, len(factor), factor, self.global_solution) for factor in fa.factors]

    def evaluate_pareto_dominance(self, population):
        """
        check whether the solutions are non-dominated
        """
        non_dominated_solutions = []
        for a, b in itertools.combinations(population, 2):
            if a > b:
                if b in non_dominated_solutions:
                    non_dominated_solutions.remove(b)
                non_dominated_solutions.append(a)
            elif b > a:
                if a in non_dominated_solutions:
                    non_dominated_solutions.remove(a)
                non_dominated_solutions.append(b)
            else:
                if b in non_dominated_solutions:
                    non_dominated_solutions.append(a)
                elif a in non_dominated_solutions:
                    non_dominated_solutions.append(b)
                else:
                    non_dominated_solutions.append(a)
                    non_dominated_solutions.append(b)
        return non_dominated_solutions

    def update_archive(self):
        to_check = set(self.nondom_archive.extend(self.pareto_set))
        self.nondom_archive = self.evaluate_pareto_dominance(to_check)

    def set_objectives(self, objectives):
        self.objectives = objectives

    def run(self):
        '''
        For each subpopulation:
            Optimize along all objectives
            Apply non-dominated sorting strategy
            Include diversity measure individuals
        Compete (based on non-domination of solution, i.e. overall fitness)
        -> or ignore compete and create set of solutions based on overlapping variables: improves diversity? Check this with diversity measure
        -> spread different non-domination solutions across different subpopulations, i.e., different subpopulations have different global solutions: this should also improve diversity along the PF?
            :return:
        '''
        for fea_run in range(self.fea_runs):
            pass

    def compete(self):
        """
        For each variable:
            - gather subpopulations with said variable
            - replace variable value in global solution with corresponding subpop value
            - check if it improves fitness for said solution
            - replace variable if fitness improves
        Set new global solution after all variables have been checked
        """
        sol = [x for x in self.global_solution]
        f = self.f
        curr_fitness = f.run(self.global_solution)
        for var_idx in range(self.dim):
            best_value_for_var = sol[var_idx]
            for pop_idx in self.factor_architecture.optimizers[var_idx]:
                curr_pop = self.subpopulations[pop_idx]
                pop_var_idx = np.where(curr_pop.factor == var_idx)
                var_candidate_value = curr_pop.gbest.lbest_position[pop_var_idx[0][0]]
                sol[var_idx] = var_candidate_value
                new_fitness = f.run(sol)
                if new_fitness < curr_fitness:
                    curr_fitness = new_fitness
                    best_value_for_var = var_candidate_value
            sol[var_idx] = best_value_for_var
        self.global_solution = sol
        self.global_fitness = curr_fitness
        self.solution_history.append(sol)


