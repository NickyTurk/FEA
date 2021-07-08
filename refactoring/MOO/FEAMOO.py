from ..FEA.factorevolution import FEA
from ..FEA.factorarchitecture import FactorArchitecture
from refactoring.optimizationproblems.prescription import Prescription
from .paretofront import *

import numpy as np
import random

class FEAMOO:
    def __init__(self, problem, fea_iterations, alg_iterations, pop_size, fa, base_alg, dimensions, combinatorial_options = []):
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
        self.worst_solution = []  # keep track to have a reference point for the HV indicator
        self.subpopulations = self.initialize_moo_subpopulations(combinatorial_options)
        self.pf = ParetoOptimization()

    def initialize_moo_subpopulations(self, combinatorial_options):
        random_global_solution = random.choices(combinatorial_options, k=self.dim)
        self.global_solutions.append(random_global_solution)
        fa = self.factor_architecture
        alg = self.base_algorithm
        return [alg(self.base_alg_iterations, self.pop_size, len(factor), factor, random_global_solution) for factor in fa.factors]

    def update_archive(self):
        to_check = set(self.nondom_archive.extend(self.pareto_set))
        self.nondom_archive = self.pf.evaluate_pareto_dominance(to_check, True)

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

        for var_idx in range(self.dim):
            # randomly pick one of the global solutions to perform competition for this variable
            chosen_global_solution = random.choice(self.global_solutions)
            self.global_solutions.remove(chosen_global_solution)
            sol = self.function(chosen_global_solution.variables)
            best_value_for_var = sol.variables[var_idx]
            # for each population with said variable perform competition on this single randomly chosen global solution
            for pop_idx in self.factor_architecture.optimizers[var_idx]:
                curr_pop = self.subpopulations[pop_idx]
                pop_var_idx = np.where(curr_pop.factor == var_idx)
                # randomly pick one of the nondominated solutions from this population
                var_candidate_value = random.choice(curr_pop.nondom_pop).variables[pop_var_idx[0][0]]
                sol.variables[var_idx] = var_candidate_value
                sol.set_fitness()
                # This is what needs to change
                if sol < chosen_global_solution:
                    best_value_for_var = var_candidate_value
            sol.variables[var_idx] = best_value_for_var
            self.global_solutions.append(sol)

    def share_solution(self):
        """
        Construct new global solution based on best shared variables from all swarms
        """
        to_pick = [s for s in self.global_solutions]
        for alg in self.subpopulations:
            if len(to_pick) > 1:
                gs = random.choice(to_pick)
                to_pick.remove(gs)
            elif len(to_pick) == 1:
                gs = to_pick[0]
                to_pick = [s for s in self.global_solutions]
            else:
                print('there are no elements in the global solutions list.')
                raise IndexError
            # update fitnesses
            alg.pop = [individual.update_individual_after_compete(individual.position, gs) for individual in alg.pop]
            # set best solution and replace worst solution with global solution across FEA
            alg.replace_worst_solution(gs)
            curr_best = alg.find_current_best()
            alg.gbest = min(curr_best, alg.gbest)
