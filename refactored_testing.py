from refactoring.optimizationProblems.function import Function
from refactoring.utilities.varinteraction import MEE, RandomTree
from refactoring.FEA.factorevolution import FEA
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.baseAlgorithms.pso import PSO


if __name__ == '__main__':
    print("running")

    f = Function(17, shift_data_file="f17_op.txt")
    print(f.function_to_call)

    dim = 50

    print("Starting MEET IM")
    im = MEE(f, dim, 100, 0, 0.001, 0.000001, use_mic_value=True)
    IM = im.get_IM()
    print("finished IM")
    meet = FactorArchitecture(dim=dim)
    meet.MEET(IM)
    print("finished MEET")
    meet.save_architecture("MeetRandom/meet")

    print("Starting Random 20")
    im = RandomTree(f, dim, 100, 0.001, 0.000001)
    T = im.run(20)
    print("finished Random 20")
    meet = FactorArchitecture(dim=dim)
    meet.MEET(T)
    print("finished Random 20")
    meet.save_architecture("MeetRandom/rand20")

    T = im.run(20)
    print("finished Random 40")
    meet = FactorArchitecture(dim=dim)
    meet.MEET(T)
    print("finished Random 40")
    meet.save_architecture("MeetRandom/rand40")

    T = im.run(60)
    print("finished Random 100")
    meet = FactorArchitecture(dim=dim)
    meet.MEET(T)
    print("finished Random 100")
    meet.save_architecture("MeetRandom/rand100")


    fa = FactorArchitecture()
    print("FEA MEET")
    fa.load_architecture("MeetRandom/meet")
    fea = FEA(f, 10, 10, 3, fa, PSO)
    fea.run()
    print()

    fa = FactorArchitecture()
    print("FEA Rand 20")
    fa.load_architecture("MeetRandom/rand20")
    fea = FEA(f, 10, 10, 3, fa, PSO)
    fea.run()
    print()

    fa = FactorArchitecture()
    print("FEA Rand40")
    fa.load_architecture("MeetRandom/rand40")
    fea = FEA(f, 10, 10, 3, fa, PSO)
    fea.run()
    print()

    fa = FactorArchitecture()
    print("FEA Rand100")
    fa.load_architecture("MeetRandom/rand100")
    fea = FEA(f, 10, 10, 3, fa, PSO)
    fea.run()
    print()

