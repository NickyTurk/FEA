from ..FEA.factorevolution import FEA
from ..FEA.factorarchitecture import FactorArchitecture
import numpy as np
import itertools
import random


class FEAMOO:
    def __init__(self, problem, fea_iterations, alg_iterations, pop_size, fa, base_alg, dimensions):
        self.function = problem
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
        self.global_solutions = []
        self.best_global_solution = []
        self.worst_solution = []  # keep track to have a reference point for the HV indicator
        self.subpopulations = self.initialize_moo_subpopulations()

    def initialize_moo_subpopulations(self):
        self.best_global_solution = random.choices(self.function.field.nitrogen_list, k=self.dim)
        fa = self.factor_architecture
        alg = self.base_algorithm
        return [alg(self.base_alg_iterations, self.pop_size, self.function, len(factor), factor, self.best_global_solution) for factor in fa.factors]

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
            for alg in self.subpopulations:
                alg.run()
            self.compete()
            self.share_solution()

    def compete(self):
        """
        For each variable:
            - gather subpopulations with said variable
            - replace variable value in global solution with corresponding subpop value
            - check if it improves fitness for said solution
            - replace variable if fitness improves
        Set new global solution after all variables have been checked
        """
        sol = [x for x in self.best_global_solution]
        f = self.function
        curr_fitnesses = f.run(self.best_global_solution, f.n_obj)
        for var_idx in range(self.dim):
            # Instead of checking just create several solutions?
            best_value_for_var = sol[var_idx]
            for pop_idx in self.factor_architecture.optimizers[var_idx]:
                curr_pop = self.subpopulations[pop_idx]
                pop_var_idx = np.where(curr_pop.factor == var_idx)
                var_candidate_value = curr_pop.gbest.lbest_position[pop_var_idx[0][0]]
                sol[var_idx] = var_candidate_value
                # This is what needs to change
                new_fitness = f.run(sol)
                if new_fitness < curr_fitness:
                    curr_fitness = new_fitness
                    best_value_for_var = var_candidate_value
            sol[var_idx] = best_value_for_var
        self.global_solutions.append(sol)

    def share_solution(self):
        """
        Construct new global solution based on best shared variables from all swarms
        """
        gs = self.global_solution
        print('global fitness found: ', self.global_fitness)
        print('===================================================')
        for alg in self.subpopulations:
            # update fitnesses
            alg.pop = [individual.update_individual_after_compete(individual.position, gs) for individual in alg.pop]
            # set best solution and replace worst solution with global solution across FEA
            alg.replace_worst_solution(gs)
            curr_best = alg.find_current_best()
            alg.gbest = min(curr_best, alg.gbest)
