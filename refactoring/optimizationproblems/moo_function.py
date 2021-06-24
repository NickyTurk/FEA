import pymoo
import numpy as np


class MOOFunction:
    def __init__(self, n_dimensions, n_objectives, h):
        self.dimensions = n_dimensions
        self.n_obj = n_objectives
        self.pareto_front = []
        self.reference_directions = []
        self.objectives = np.empty(n_objectives)
        self.h = h  # how many partitions to split each objective in, also used to calculate reference points
