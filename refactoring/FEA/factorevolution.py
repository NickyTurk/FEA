import numpy as np


class FEA:
    def __init__(self, function, fea_runs, generations, pop_size, factor_architecture, base_algorithm):
        self.f = function
        self.fea_runs = fea_runs
        self.generations = generations
        self.pop = pop_size
        self.factor_architecture = factor_architecture
        self.dim = factor_architecture.dim
        self.base_algorithm = base_algorithm
        self.global_solution = np.random.uniform(function.lbound, function.ubound, size=factor_architecture.dim)
        self.global_fitness = function.run(self.global_solution)
        self.solution_history = [self.global_solution]
        self.subpopulations = self.initialize_factored_subpopulations()

    def run(self):
        for fea_run in range(self.fea_runs):
            for alg in self.subpopulations:
                alg.run()
            self.compete()
            self.share_solution()
            print('fea run ', fea_run, self.global_fitness)

    def initialize_factored_subpopulations(self):
        fa = self.factor_architecture
        alg = self.base_algorithm
        return [alg(self.generations, self.pop, self.f, len(factor), factor, self.global_solution) for factor in fa.factors]

    def share_solution(self):
        """
        Construct new global solution based on best shared variables from all swarms
        """
        gs = [x for x in self.global_solution]
        print('global fitness found: ', self.global_fitness)
        print('===================================================')
        for alg in self.subpopulations:
            # update fitnesses
            alg.pop = [individual.update_individual_after_compete(individual.position, gs) for individual in alg.pop]
            # set best solution and replace worst solution with global solution across FEA
            alg.replace_worst_solution(gs)
            curr_best = alg.find_current_best()
            alg.gbest = min(curr_best, alg.gbest)

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
            best_value_for_var = 0
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


if __name__ == '__main__':
    from refactoring.baseAlgorithms.pso import PSO
    from refactoring.optimizationProblems.function import Function
    from refactoring.FEA.factorarchitecture import FactorArchitecture

    fa = FactorArchitecture()
    fa.load_csv_architecture(file="../../results/factors/F1_overlapping_diff_grouping.csv", dim=50)
    func = Function(function_number=1, shift_data_file="f01_o.txt")
    fea = FEA(func, 10, 10, 3, fa, PSO)
    fea.run()