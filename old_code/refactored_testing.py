from refactoring.optimizationproblems.function import Function
from refactoring.utilities.varinteraction import MEE, RandomTree
from refactoring.FEA.factorevolution import FEA
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.basealgorithms.pso import PSO


if __name__ == '__main__':
    outputfile = open('./MeetRandom/trial.txt', 'a')
    print("running")

    f = Function(17, shift_data_file="f17_op.txt")
    print(f.function_to_call)

    dim = 200
    outputfile.write("Dim: " + str(dim) + " Random Init\n")
    random_iteration = [50, 50, 100, 100, 200]

    print("Starting MEET IM")
    im = MEE(f, dim, 100, 0, 0.001, 0.000001, use_mic_value=True)
    IM = im.get_IM()
    print("finished IM")
    meet = FactorArchitecture(dim=dim)
    meet.MEET(IM)
    print("finished MEET")
    meet.save_architecture("MeetRandom/meet")

    total = 0
    for it in random_iteration:
        total += it

        print("Starting Random " + str(total))
        im = RandomTree(f, dim, 100, 0.001, 0.000001)
        T = im.run(20)
        print("finished Random " + str(total))
        meet = FactorArchitecture(dim=dim)
        meet.MEET(T)
        print("finished Random " + str(total))
        meet.save_architecture("MeetRandom/rand" + str(total))

    fa = FactorArchitecture()
    print("FEA MEET")
    fa.load_architecture("MeetRandom/meet")
    fea = FEA(f, 10, 10, 3, fa, PSO)
    fea.run()
    outputfile.write(f"MEET, \t\t{fea.global_fitness}\n")
    print(fea.global_fitness)

    total = 0
    for it in random_iteration:
        total += it
        fa = FactorArchitecture()
        print("FEA Rand " + str(total))
        fa.load_architecture("MeetRandom/rand" + str(total))
        fea = FEA(f, 10, 10, 3, fa, PSO)
        fea.run()
        outputfile.write(f"Rand {total}, \t{fea.global_fitness}\n")
        print(fea.global_fitness)


