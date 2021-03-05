from benchmarks import _benchmarks
from topology import *
from read_data import *
from pso import pso
from fea_pso import fea_pso
import argparse, csv
from datetime import datetime
from evaluation import *
from clustering import *
import numpy as np
from opfunu.cec.cec2010.function import *
# from cec2013lsgo.cec2013 import Benchmark
from functools import partial
from FunctionTesting import F11_E

from variable_interaction import MEE, MEET
from numpy import linalg as la


def harness(algorithm, iterations, repeats):
    summary = {}
    fitnesses = []
    for trial in range(0, iterations):
        result = algorithm()
        fitnesses.append(result[-1][2])
    bootstrap = create_bootstrap_function(repeats)
    replications = bootstrap(fitnesses)
    statistics = analyze_bootstrap_sample(replications)
    summary["statistics"] = statistics
    summary["bootstrap"] = replications
    summary["fitnesses"] = fitnesses
    return summary


def test_diff_grouping(function_names, m=0):
    shifted_function = ['F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F17', 'F18']
    shifted_error_function = ['F14', 'F15', 'F16']
    dimensions = np.arange(start=10, stop=101, step=10)
    epsilons = [1e-1, 1e-3, 1e-6, 1e-9]

    bench = Benchmark()

    for i, f_name in enumerate(function_names):
        with open('results/2013/' + f_name + '_2013_diff_grouping.csv', 'w') as csv_write:
            csv_writer = csv.writer(csv_write)
            csv_writer.writerow(['FUNCTION', 'DIMENSION', 'EPSILON', 'NR_GROUPS', 'FACTORS', 'SEPARATE VARS'])
            for d in dimensions:
                for e in epsilons:
                    f_int = int(''.join(list(filter(str.isdigit, f_name))))
                    f = bench.get_function(f_int)

                    factors, arbiters, optimizers, neighbors, separate_variables = generate_diff_grouping(
                        f, d, e)
                    
                    csv_writer.writerow([function_names[i], str(d), str(e), len(factors), factors, separate_variables])


def test_optimization(dimensions, function_names, cec2010_functions):
    file_extension = "m4_diff_grouping_small_epsilon"
    #file_extension = "overlapping_diff_grouping_small_epsilon"
    #file_extension = "2013_diff_grouping"
    filename_list = get_files_list("diff_grouping/F*_" + file_extension + ".csv")

    domain = [-50,50]

    for filename in filename_list:
        for dim in dimensions:
            no_m_param = ['F1', 'F2', 'F3', 'F19', 'F20']
            shifted_error_function = ['F14', 'F15', 'F16']
            m=4

            for i,function_name in enumerate(function_names):
                #test_var_int(function_name)

                if function_name in no_m_param:
                    f = cec2010_functions[i]
                elif function_name in shifted_error_function:
                    f = partial(cec2010_functions[i], m_group=dim)
                else:
                    f = partial(cec2010_functions[i], m_group=m) 
                factors, function_name = import_single_function_factors(filename, dim)
                if function_name in function_names:
                    print('current function ', function_name)
                    arbiters, optimizers, neighbors = get_factor_info(factors, dim)

                    # f_int = int(''.join(list(filter(str.isdigit, function_name))))
                    
                    # f = bench.get_function(f_int)
                    # info = bench.get_info(f_int)
                    # domain = (info['lower'], info['upper'])

                    pso_stop = lambda t, s: t == 10
                    p = 500
                    n = 10

                    algorithm = lambda: fea_pso(f, dim, domain, factors, optimizers, p, n, pso_stop)
                    summary = harness(algorithm, n, 1)

                    print("finished with one function, one dimension")
                    with open('results/FEA_PSO/' + str(function_name) + '_dim' + str(dim) + file_extension + ".csv", 'w') as write_to_csv:
                        csv_writer = csv.writer(write_to_csv)
                        csv_writer.writerow(summary["fitnesses"])
                        print('printed')

def test_pso(function_name, p, dim, t = 100, function = None):
    # if function is None:
    #     f_int = int(''.join(list(filter(str.isdigit, function_name))))
    #     f = function
    #     #f = bench.get_function(f_int)
    #     #info = bench.get_info(f_int)
    #     #domain = (info['lower'], info['upper'])
    # else:
    f = function
    domain = (-32,32)
    summary = {"name": function_name}
    fitnesses = []
    for i in range(10):
        result = pso(f, p, dim, domain, lambda t, f: t == t)
        fitnesses.append(result[-1].fitness)
    bootstrap = create_bootstrap_function(250)
    replications = bootstrap(fitnesses)
    statistics = analyze_bootstrap_sample(replications)
    summary["statistics"] = statistics
    summary["fitnesses"] = fitnesses
    return summary


def get_factor_info(factors, d):
    arbiters = nominate_arbiters(factors)
    optimizers = calculate_optimizers(d, factors)
    neighbors = determine_neighbors(factors)
    return arbiters, optimizers, neighbors


def test_var_int(f, name):
    matricies = ''

    d = 10

    interactions = []
    sizes = [50, 50, 20, 20, 10, 10, 5, 5]
    totals = []

    for s in sizes:
        mee = MEE(f, d, np.ones(d)*s, np.ones(d)*-s, 10000, 0.3, 0.0001, 0.000001)
        mee.direct_IM()
        matricies += '\n'
        matricies += 'Search: ' + str(s) + 'x' + str(s) + ' around origin\n'
        s, total = np_to_str(mee.IM)
        matricies += s  # I couldn't figure out how to get full representation of np array so made own function
        matricies += '\n'
        interactions.append(mee.IM)
        totals.append(total)
        print(np.array(mee.IM))
        # mee.strongly_connected_comps()
        print()

    norms = []
    data = ''
    data += 'Norms:\n'
    for var_int in interactions:
        diff = interactions[0]-var_int
        norms.append(la.norm(diff))
    data += str(norms)
    print(norms)

    data += '\nTotals:\n'
    data += str(totals)
    print(totals)

    data += '\n\n\n'
    data += matricies

    with open('SpaceSearch/' + name + '.txt', 'w') as f:
        f.write(data)


def MEET_factors(function, dim, de_thr = 0.001):
    ub = np.ones(dim) * 100
    lb = np.ones(dim) * -100
    delta = 0.000001  # account for variations
    sample_size = dim*10

    # caluclate MEE
    mee = MEET(function, dim, ub, lb, sample_size, de_thr, delta)
    mee.compute_interaction()
    mee.assign_factors()
    return mee.factors, mee.mic


def fuzzy_MEE_factors(function_name, function, dim, fuzzy_cluster_threshold, mic_thr = 0.1, de_thr = 0.001):
    ub = np.ones(dim) * 100
    lb = np.ones(dim) * -100
    a = mic_thr  # mic threshold
    b = de_thr  # de threshold
    delta = 0.000001  # account for variations
    sample_size = dim*4

    # calculate MEE
    mee = MEE(function, dim, ub, lb, sample_size, a, b, delta)
    mee.direct_IM()

    mee.strongly_connected_comps()

    # run spectral on adjacency/interaction matrix
    spectral = Spectral(np.array(mee.IM), 3)
    spectral.assign_clusters()
    spectral_factors = spectral.return_factors(spectral.soft_clusters, threshold= fuzzy_cluster_threshold)
    return spectral_factors


def np_to_str(x):
    s = ''
    total = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            total += x[i][j]
            s += str(x[i][j])
            s += '\t'
        s += '\n'
    return s,total

def test_runtime():
    f = F11_E()  # intialize

    # generate a point (use from MEE)
    dim = 50
    ub = np.ones(dim) * 100
    lb = np.ones(dim) * -100
    x = np.random.rand(dim) * (ub - lb) + lb

    m = 4

    y_1 = None
    y_2 = None

    trials = 1000

    start = time.time()
    for _ in range(trials):
        y_1 = f.F11o(x, m_group=m)  # original
    end = time.time()
    original_elapse = end - start
    print("Original: " + str(original_elapse))

    start = time.time()
    for _ in range(trials):
        y_2 = f.run(x, m_group=m)  # new and improved?
    end = time.time()
    new_elapse = end - start
    print("New: " + str(new_elapse))

    print(str(y_1 == y_2))  # make sure answer is the same
    print(str(original_elapse/new_elapse))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test out some algies")
    parser.add_argument("--benchmark", help="pick the name of a benchmark function", default="schwefel-1.2")
    parser.add_argument("--seed", help="the random seed to use")
    args = parser.parse_args()

    seed = args.seed
    if not seed:
        seed = int(datetime.now().strftime("%S"))

    benchmark = args.benchmark
    functions = [F3, F10, F12]

    function_names = ['F19', 'F20'] #'F6', 'F7', 'F10', 'F11',S
    cec2010_functions = [F11, F19, F20]



    # test_var_int('F6')

    dimensions = [50, 100]
    populations = [500,1000]
    iteration = 200

    no_m_param = ['F1', 'F2', 'F3', 'F19', 'F20']
    shifted_error_function = ['F14', 'F15', 'F16']
    m = 4

    test_runtime()
    exit(13)

    # MEET_factors(F3, 10)
    #
    # exit(1)

    for i,function_name in enumerate(function_names):
        #test_var_int(function_name)

        if function_name in no_m_param:
            f = cec2010_functions[i]
        elif function_name in shifted_error_function:
            f = partial(cec2010_functions[i], m_group=dim)
        else:
            f = partial(cec2010_functions[i], m_group=m)

        with open("results/meet_factors/" + function_name + "_meet.csv", "a") as write_to_csv:
            csv_writer = csv.writer(write_to_csv)
            # csv_writer.writerow(['function', 'dim', 'population', 'iterations', 'fitnesses', 'stats'])
            # for pop in populations:
            #     print(function_name, ' pop: ', str(pop))
            #     summary = test_pso(function_name, pop, dim, t=iteration, function=f)
            #     to_write = [function_name, str(dim), str(pop), str(iteration), summary["fitnesses"], summary["statistics"]]
            #     csv_writer.writerow(to_write)
            csv_writer.writerow(['function', 'dim', 'nr_groups', 'factors'])
            for dim in [20,50]:
                print('function ', function_name, 'dim: ', str(dim))
                factors, mic = MEET_factors(f, dim)
                to_write = [function_name, str(dim), str(len(factors)), str(factors)]
                csv_writer.writerow(to_write)
                print(to_write)
                np.savetxt("results/meet_factors/mic/" + function_name + "_" + str(dim) + ".csv", mic, delimiter=',')

    print('ALL DONE')
    exit(13)

    #test_diff_grouping(function_names)
    test_optimization(dimensions, function_names, cec2010_functions)


    # pso_stop = lambda t, s: t == 5
    # p = 100
    # n = 100
    # # width = d
    #
    # algorithm = lambda: pso(f, p * dim, dim, domain, lambda t, s: t == 5)
    # summary = harness(algorithm, n, 1)
    # print("G=", summary["statistics"])
    # print("G=", summary["fitnesses"])
    #
    # with open('results/f7_pso.csv', 'w') as write_to_csv:
    #     csv_writer = csv.writer(write_to_csv)
    #     csv_writer.writerows(summary["fitnesses"])
    #
    # random.seed(seed)
    # print("starting FEA")
    # algorithm = lambda: fea_pso(f, d, domain, factors, optimizers, p, n, pso_stop)
    # summary = harness(algorithm, n, 1)
    # print("G=", summary["statistics"])
    # print("G=", summary["fitnesses"])
    #
    # with open('results/f7_fea_pso_diff_group.csv', 'w') as write_to_csv:
    #     csv_writer = csv.writer(write_to_csv)
    #     csv_writer.writerows(summary["fitnesses"])

    # factors, arbiters, optimizers, neighbors = generate_linear_topology(d, k)
    #
    # random.seed(seed)
    # print("starting FEA")
    # algorithm = lambda: fea_pso(f, d, domain, factors, optimizers, p, n, pso_stop)
    # summary = harness(algorithm, n, 1)
    # print("G=", summary["statistics"])
    # print("G=", summary["fitnesses"])
    #
    # with open('results/f7_fea_pso_linear_topology.csv', 'w') as write_to_csv:
    #     csv_writer = csv.writer(write_to_csv)
    #     csv_writer.writerows(summary["fitnesses"])

    #FOR CEC 2010 OPFUNU
    #no_m_param = ['F1', 'F2', 'F3', 'F19', 'F20']
    #shifted_error_function = ['F14', 'F15', 'F16']

    # if function_name in no_m_param:
    #     f = functions[function_names.index(function_name)]
    # elif function_name in shifted_error_function:
    #     f = partial(functions[function_names.index(function_name)], m_group=dim)
    # else:
    # f = partial(functions[function_names.index(function_name)], m_group=m)  # retrieve appropriate function
