from pymoo.factory import get_performance_indicator
import numpy as np
import math
from operator import attrgetter


class ParetoOptimization:

    def __init__(self, obj_size):
        """
        :param obj_size: The number of objectives in the problem definition.
        """
        self.n_obj = obj_size

    def evaluate_solution(self, approx_pareto_set, reference_point):
        """
        :param approx_pareto_set: The discovered pareto set to calculate different evaluations for.
        :param reference_point: Which reference point to use to calculate HV
        """
        # reference_points = self.calculate_ref_points(h, self.obj_size)
        hv = get_performance_indicator("hv", ref_point=reference_point)
        diversity = self.calculate_diversity(approx_pareto_set)
        return {'hypervolume': hv.calc(approx_pareto_set), 'diversity': diversity}

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
        sorted_set = sorted(approx_pareto_set, key=attrgetter('overall_fitness'))

        for i in self.n_obj:
            spread_indicator += np.square(sorted_set[-1].variables - sorted_set[0].variables)
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

