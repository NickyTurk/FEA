from ..FEA.factorevolution import FEA
from ..FEA.factorarchitecture import FactorArchitecture


class FEAMOO:
    def __init__(self, fea_iterations, alg_iterations, pop_size, fa, base_alg, dimensions, fitness_function=None):
        if fitness_function is not None:
            self.objectives = fitness_function.objectives
        else:
            self.objectives = []
        self.dim = dimensions
        self.nondom_archive = []
        self.current_nondom_global_solutions=[]
        self.factors = fa
        self.algorithm = base_alg
        self.iterations = fea_iterations
        self.base_alg_iterations = alg_iterations
        self.pop_size = pop_size
        self.current_iteration = 0

    def evaluate_pareto_dominance(self):
        """
        check whether the solutions are still non-dominated otherwise they must be replaced
        """
        for nondom in self.non_dominated_solutions:
            pass

    def update_archive(self):
        for sol in self.archive:
            pass

    def set_objectives(self, objectives):
        self.objectives = objectives

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
        fea = FEA(self.problem, 10, 10, 3, self.factors, self.algorithm)

