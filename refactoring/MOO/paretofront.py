from pymoo.factory import get_performance_indicator
import numpy as np
import math


class ParetoOptimization:

    def __init__(self, obj_size):
        """
        :param obj_size: The number of objectives in the problem definition.
        """
        self.obj_size = obj_size

    def evaluate_solution(self, result, h=0):
        """
        :param result: Result to calculate different evaluations for.
        :param h: A user defined parameter used to define the population size;
                based on this population size, the reference point is calculated.
        """
        reference_points = self.calculate_ref_points(h, self.obj_size)
        hv = get_performance_indicator("hv", ref_point=reference_points)
        return {'hypervolume': hv.calc(result)}

    def calculate_ref_points(self, h: int, obj_size):
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
         """
        ref_value = 1 + 1/h
        ref_points = np.full((obj_size), ref_value)
        return ref_points

