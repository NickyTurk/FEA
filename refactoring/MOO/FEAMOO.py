from ..FEA.factorevolution import FEA
from ..FEA.factorarchitecture import FactorArchitecture
from refactoring.optimizationproblems.prescription import Prescription
from .paretofront import *

import numpy as np
import random


class FEAMOO:
    def __init__(self, problem, fea_iterations, alg_iterations, pop_size, fa, base_alg, dimensions,
                 combinatorial_options=None, field=None):
        if combinatorial_options is None:
            combinatorial_options = []
        self.field = field
        self.function = problem
        self.dim = dimensions
        self.nondom_archive = []
        self.population = []
        self.factor_architecture = fa
        self.base_algorithm = base_alg
        self.fea_runs = fea_iterations
        self.base_alg_iterations = alg_iterations
        self.pop_size = pop_size
        self.current_iteration = 0
        self.global_solutions = []
        self.worst_solution = self.function(field.assign_nitrogen_distribution(), field=self.field)
        # keep track to have a reference point for the HV indicator
        self.subpopulations = self.initialize_moo_subpopulations(combinatorial_options)
        self.pf = ParetoOptimization()

    def initialize_moo_subpopulations(self, combinatorial_options):
        random_global_solution = self.function(self.field.assign_nitrogen_distribution(), field=self.field)
        self.global_solutions.append(random_global_solution)
        fa = self.factor_architecture
        alg = self.base_algorithm
        return [alg(ga_runs=self.base_alg_iterations, population_size=self.pop_size, factor=factor,
                    global_solution=random_global_solution) for factor in fa.factors]

    def update_archive(self):
        to_check = [s for s in self.nondom_archive]
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
                alg.run(field=self.field)
                self.nondom_archive.extend(alg.nondom_pop)
            # print('Run ', fea_run, ' before compete/share: ')
            # [print(s.objective_values) for s in self.nondom_archive]
            self.compete()
            self.share_solution()
            self.update_archive()
            print(len(self.nondom_archive))
            # print('Run ', fea_run, ' after compete/share: ')
            # [print(s.objective_values) for s in self.nondom_archive]

    def compete(self):
        """
        For each variable:
            - gather subpopulations with said variable
            - replace variable value in global solution with corresponding subpop value
            - check if it improves fitness for said solution
            - replace variable if fitness improves
        Set new global solution after all variables have been checked
        """
        new_solutions = []
        for var_idx in range(self.dim):
            # randomly pick one of the global solutions to perform competition for this variable
            chosen_global_solution = random.choice(self.global_solutions)
            sol = self.function(chosen_global_solution.variables, self.field)
            best_value_for_var = sol.variables[var_idx]
            # for each population with said variable perform competition on this single randomly chosen global solution
            for pop_idx in self.factor_architecture.optimizers[var_idx]:
                curr_pop = self.subpopulations[pop_idx]
                pop_var_idx = np.where(np.array(curr_pop.factor) == var_idx)
                # randomly pick one of the nondominated solutions from this population
                if len(curr_pop.nondom_pop) != 0:
                    random_sol = random.choice(curr_pop.nondom_pop)
                else:
                    random_sol = random.choice(curr_pop.gbests)
                var_candidate_value = random_sol.variables[pop_var_idx[0][0]]
                sol.variables[var_idx] = var_candidate_value
                sol.set_fitness()
                # This is what needs to change
                if sol < chosen_global_solution:
                    best_value_for_var = var_candidate_value
            sol.variables[var_idx] = best_value_for_var
            new_solutions.append(sol)

        self.global_solutions = self.pf.evaluate_pareto_dominance(new_solutions)
        self.nondom_archive.extend(self.global_solutions)

    def share_solution(self):
        """
        Construct new global solution based on best shared variables from all swarms
        """
        if len(self.global_solutions) != 0:
            to_pick = [s for s in self.global_solutions]
        else:
            to_pick = [s for s in self.nondom_archive]
        for i, alg in enumerate(self.subpopulations):
            if len(to_pick) > 1:
                gs = random.choice(to_pick)
                to_pick.remove(gs)
            elif len(to_pick) == 1:
                gs = to_pick[0]
                # repopulate the list to restart if not at the last subpopulation
                if i < len(self.subpopulations) - 1:
                    if len(self.global_solutions) != 0:
                        to_pick = [s for s in self.global_solutions]
                    else:
                        to_pick = [s for s in self.nondom_archive]
            else:
                print('there are no elements in the global solutions list.')
                raise IndexError
            # update fitnesses
            alg.global_solution = gs
            alg.curr_population = [self.function(p.variables, self.field, gs, alg.factor) for p in alg.curr_population]
            # set best solution and replace worst solution with global solution across FEA
            temp_worst = alg.replace_worst_solution(gs)
            self.nondom_archive.extend(alg.nondom_pop)
            if temp_worst > self.worst_solution:
                self.worst_solution = temp_worst
