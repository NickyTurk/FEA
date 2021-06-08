from pymoo.factory import get_problem, get_reference_directions, get_visualization

class MOOFunctions:
    def __init__(self, n_dimensions, n_objectives):
        self.dim = n_dimensions
        self.n_obj = n_objectives
        self.pareto_front = []
        self.reference_directions = []
        self.objectives = []

    def dtlz1(self):
        problem = get_problem('dtlz1', n_var = self.dim, n_obj = self.n_obj)
        self.reference_directions = get_reference_directions("das-dennis", self.n_obj, n_partitions=12)
        self.pareto_front = problem.pf(self.reference_directions)
        self.objectives = problem.obj_func()
