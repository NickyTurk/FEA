from refactoring.optimizationProblems.function import Function
from refactoring.utilities.varinteraction import MEE, RandomTree
from refactoring.FEA.factorevolution import FEA
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.baseAlgorithms.pso import PSO

# from stat_analysis import factor_graphing

if __name__ == '__main__':
    outputfile = open('results/MEET/MeetRandom/trial.txt', 'a')
    print("running")

    f = Function(17, shift_data_file="f17_op.txt")
    print(f.function_to_call)

    dim = 200
    outputfile.write("Dim: " + str(dim) + " Seed = 1 \t ODG and DG\n")
    random_iteration = [5, 15, 30, 50, 100, 200, 600, 1000]

    im = RandomTree(f, dim, 100, 0.001, 0.000001)
    # function, epsilon, m
    odg = FactorArchitecture(dim=dim)
    print('starting odg')
    odg.overlapping_diff_grouping(f, 0.001)
    odg.save_architecture('MeetRandom/odg')

    dg = FactorArchitecture(dim=dim)
    print('starting dg')
    dg.diff_grouping(f, 0.001)
    dg.save_architecture('MeetRandom/dg')

    summary  = {}

    fa = FactorArchitecture()
    print("FEA ODG")
    fa.load_architecture("MeetRandom/odg")
    fea = FEA(f, 10, 10, 3, fa, PSO, seed=1)
    fea.run()
    outputfile.write(f"ODG, \t\t{fea.global_fitness}\n")
    print(fea.global_fitness)
    summary['ODG'] = fea.global_fitness

    fa = FactorArchitecture()
    print("FEA DG")
    fa.load_architecture("MeetRandom/dg")
    fea = FEA(f, 10, 10, 3, fa, PSO, seed=1)
    fea.run()
    outputfile.write(f"DG, \t\t{fea.global_fitness}\n")
    print(fea.global_fitness)
    summary['DG'] = fea.global_fitness

    print(summary)


