"""
    Benchmark functions as defined by:
     ------------------------------- Reference --------------------------------
    R. Cheng, Y. Jin, and M. Olhofer, Test problems for large-scale
    multiobjective and many-objective optimization, IEEE Transactions on
    Cybernetics, 2017, 47(12): 4108-4121.
     ------------------------------- Copyright --------------------------------
    Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
    research purposes. All publications which use this platform or any code
    in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    for evolutionary multi-objective optimization [educational forum], IEEE
    Computational Intelligence Magazine, 2017, 12(4): 73-87".
     --------------------------------------------------------------------------
"""

import pymoo
from pymoo.factory import get_problem, get_reference_directions, get_visualization
from pymoo.util.plotting import plot
from topology import *
from benchmarks import *
    

def test_moo(problem):
    for i in range(0, problem.n_obj):
        print('objective index: ', i)
        #generate_diff_grouping(eggholder, 50, 0.00000000001)
        generate_diff_grouping(problem, problem.n_var, 0.0000000001, MOO=True, moo_idx=i)


class TestProblem(pymoo.model.problem.Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,  # f = objectives notation
                         # n_constr=2,  # g = constraints notation
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]),
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]**2 + x[1]**2
        f2 = (x[0]-1)**2 + x[1]**2

        # g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        # g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        # out["G"] = [g1, g2]

if __name__ == "__main__":
    dtlz = get_problem("dtlz1", n_var=20, n_obj=6)
    objs = [
    lambda x: eggholder(x),
    lambda x: rosenbrock(x)
    ]
    functional_problem = pymoo.model.problem.FunctionalProblem(50,
                                       objs,
                                       xl=np.array([-100,-100]),
                                       xu=np.array([100,100]))
    test_moo(dtlz)
    
    