from pymoo.factory import get_performance_indicator
import numpy as np


class ParetoOptimization:

    def __init__(self):
        pass

    def evaluate_paretofront(self, result):
        reference_points = self.calculate_ref_points()
        hv = get_performance_indicator("hv", ref_point=reference_points)
        return hv.calc(result)

    def calculate_ref_points(self):
        ref_points = []
        return np.array()