import itertools

from pymoo.factory import get_performance_indicator
from pymoo.algorithms.nsga2 import calc_crowding_distance
import numpy as np
import math
from operator import attrgetter, itemgetter


class ParetoOptimization:

    def __init__(self, obj_size=3):
        """
        :param obj_size: The number of objectives in the problem definition.
        """
        self.n_obj = obj_size
        self.approximate_pareto_front = []

    def evaluate_pareto_dominance(self, population, save=False):
        """
        check whether the solutions are non-dominated
        """
        pop = [x for x in population]
        non_dominated_solutions = []
        for comb in itertools.combinations(pop, 2):
            a, b = comb
            if a < b:
                if b in non_dominated_solutions:
                    non_dominated_solutions.remove(b)
                if a not in non_dominated_solutions:
                    non_dominated_solutions.append(a)
            elif b < a:
                if a in non_dominated_solutions:
                    non_dominated_solutions.remove(a)
                if b not in non_dominated_solutions:
                    non_dominated_solutions.append(b)
            # elif a == b:
            #     if b in non_dominated_solutions:
            #         non_dominated_solutions.append(a)
            #     elif a in non_dominated_solutions:
            #         non_dominated_solutions.append(b)
            #     else:
            #         non_dominated_solutions.append(a)
            #         non_dominated_solutions.append(b)
        self.approximate_pareto_front.extend([s.variables for s in non_dominated_solutions])
        return non_dominated_solutions

    def evaluate_solution(self, approx_pareto_set, reference_point):
        """
        :param approx_pareto_set: The discovered pareto set to calculate different evaluations for.
        :param reference_point: Which reference point to use to calculate HV
        """
        # reference_points = self.calculate_ref_points(h, self.obj_size)

        diversity = self.calculate_diversity(approx_pareto_set)

        updated_pareto_set = [np.array(sol.fitness) for sol in approx_pareto_set]
#        for sol in approx_pareto_set:
#            updated_pareto_set.append(np.array([x.nitrogen for x in sol.variables]))
        hv = get_performance_indicator("hv", ref_point=np.array(reference_point)) #hypervolume(np.array(updated_pareto_set))
        return {'hypervolume': hv.calc(np.array(updated_pareto_set)), 'diversity': diversity}

    def calculate_diversity(self, approx_pareto_set):
        """
        Spread/delta indicator:
        estimate the extent of the spread of the obtained Pareto front,
        supposedly created by Deb? But I cant find a copy of his 2001 book

        square root of the sum of the square of the difference between max and min solutions
        (determined based on overall fitness) for each objective
        Alternate wording: the sum of the width for each objective.
        Adra and Fleming, 2000. Coello Coello, Dhaenens, and Jordan, 2010. Ischubishi and shibata, 2004, GECCO
        """

        spread_indicator = 0
        for i in range(self.n_obj):
            to_sort = [x.fitness[i] for x in approx_pareto_set]
            sorted_set = [x for _,x in sorted(zip(to_sort, approx_pareto_set))]
            last = np.array([c for c in sorted_set[-1].variables])
            first = np.array([c for c in sorted_set[0].variables])
            spread_indicator += np.square(np.linalg.norm(last - first))
        return np.sqrt(spread_indicator)

    def calculate_ref_points(self, h: int):
        """A slightly worse point than the nadir point is usually used for hypervolume calculation in the EMO
        community.
         ...
         The basic idea is to specify the reference point so that a set of well-distributed solutions over the entire
         linear Pareto front has a large hypervolume and all solutions in such a solution set have similar
         hypervolume contributions.
         ...
         Our discussions and experimental results clearly show that a slightly worse point than the nadir point
         is not always appropriate for performance comparison of EMO algorithms.
         (Hisao Ishibuchi, Ryo Imada, Yu Setoguchi, Yusuke Nojima;
         How to Specify a Reference Point in Hypervolume Calculation for Fair Performance Comparison.
         Evol Comput 2018; 26 (3): 411–440.)

         OR

         use the solution with the worst found fitness values for each of the objectives: keep track of this while optimizing
         """
        ref_value = 1 + 1/h
        ref_points = np.full(self.n_obj, ref_value)
        return ref_points

