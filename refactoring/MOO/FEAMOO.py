from ..FEA.factorevolution import FEA
from ..FEA.factorarchitecture import FactorArchitecture


class FEAMOO:
    def __init__(self, problem, base_alg):
        self.problem = problem
        self.objectives = problem.objectives
        self.obj_size = problem.n_obj
        self.dim = problem.dimensions
        self.factors = self.create_objective_factors()
        self.algorithm = base_alg
        self.non_dominated_solutions = []
        self.archive = []

    def create_objective_factors(self, savefiles=True):
        """Create factors along different objective functions.
        For each objective, a FactorArchitecture object is created.
        :param savefiles: Boolean that determines whether the created factorArchitectures are saved in pickle files
        :returns FactorArchitecture object: with all the factors generated
        """
        all_factors = FactorArchitecture(self.dim)
        for i, obj in enumerate(self.objectives):
            fa = FactorArchitecture(self.dim)
            fa.diff_grouping(obj, 0.001)
            if savefiles:
                fa.save_architecture(
                    '../factor_architecture_files/MOO_' + fa.method + '_dim_' + str(self.dim) + '_obj_' + str(i))
            all_factors.factors.extend(fa.factors)
        all_factors.get_factor_topology_elements()
        return all_factors

    def evaluate_pareto_dominance(self):
        """
        check whether the solutions are still non-dominated otherwise they must be replaced
        """
        for nondom in self.non_dominated_solutions:
            pass

    def update_archive(self):
        for sol in self.archive:
            pass

    def run(self):

        fea = FEA(self.problem, 10, 10, 3, self.factors, self.algorithm)

