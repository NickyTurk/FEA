from pymoo.factory import get_problem, get_reference_directions, get_visualization
import numpy as np
import math


class MOOBenchmark:
    def __init__(self, fion_name, n_dimensions, n_objectives, h):
        self.fion_name = fion_name
        self.dimensions = n_dimensions
        self.n_obj = n_objectives
        self.problem = get_problem(fion_name, n_var=self.dimensions, n_obj=self.n_obj, elementwise_evaluation=True)
        self.reference_directions = get_reference_directions("das-dennis", self.n_obj, n_partitions=self.h)
        self.pareto_front = self.problem.pf(self.reference_directions)
        self.objectives = self.problem.obj_func()
        self.h = h  # how many partitions to split each objective in, also used to calculate reference points
        self.population_size = math.comp(h+n_objectives-1, n_objectives-1)

    def run(self, solution):
        return self.problem._evaluate(solution)

