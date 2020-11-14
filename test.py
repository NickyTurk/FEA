from benchmarks import _benchmarks
from topology import *
from read_data import *
from pso import pso
from fea_pso import fea_pso
import argparse, csv
from datetime import datetime
from evaluation import *
import numpy as np
from opfunu.cec.cec2010.function import *
from cec2013lsgo.cec2013 import Benchmark
from functools import partial

from variable_interaction import MEE
from deap.benchmarks import *
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


def test_optimization(dimensions, function_names):
    #file_extension = "m4_diff_grouping_small_epsilon"
    #file_extension = "overlapping_diff_grouping_small_epsilon"
    file_extension = "2013_diff_grouping"
    filename_list = get_files_list("2013/F*_" + file_extension + ".csv")
    bench = Benchmark()

    for filename in filename_list:
        for dim in dimensions:
            factors, function_name = import_single_function_factors(filename, dim)
            if function_name in function_names:
                print('current function ', function_name)
                arbiters, optimizers, neighbors = get_factor_info(factors, dim)

                f_int = int(''.join(list(filter(str.isdigit, function_name))))
                
                f = bench.get_function(f_int)
                info = bench.get_info(f_int)
                domain = (info['lower'], info['upper'])
                print(domain)

                pso_stop = lambda t, s: t == 10
                p = 400
                n = 10

                algorithm = lambda: fea_pso(f, dim, domain, factors, optimizers, p, n, pso_stop)
                summary = harness(algorithm, n, 1)

                print("finished with one function, one dimension")
                with open('results/FEA_PSO/' + str(function_name) + '_dim' + str(dim) + file_extension + ".csv", 'w') as write_to_csv:
                    csv_writer = csv.writer(write_to_csv)
                    csv_writer.writerow(summary["fitnesses"])
                    print('printed')


def get_factor_info(factors, d):
    arbiters = nominate_arbiters(factors)
    optimizers = calculate_optimizers(d, factors)
    neighbors = determine_neighbors(factors)
    return arbiters, optimizers, neighbors


def test_var_int(function_name):
    matricies = ''

    bench = Benchmark()
    f_int = int(''.join(filter(str.isdigit, function_name)))
    print(f_int)

    f = bench.get_function(f_int)
    info = bench.get_info(f_int)
    domain = (info['lower'], info['upper'])
    print(domain)

    d = 50

    interactions = []
    sizes = [100, 100, 50, 50, 10, 10]
    totals = []

    for s in sizes:
        mee = MEE(f, d, np.ones(d)*s, np.ones(d)*-s, 50, 0.3, 0.0001, 0.000001)
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

    with open('SpaceSearch/' + function_name + '.txt', 'w') as f:
        f.write(data)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test out some algies")
    parser.add_argument("--benchmark", help="pick the name of a benchmark function", default="schwefel-1.2")
    parser.add_argument("--seed", help="the random seed to use")
    args = parser.parse_args()

    seed = args.seed
    if not seed:
        seed = int(datetime.now().strftime("%S"))

    benchmark = args.benchmark
    functions = [F6, F10, F12]

    function_names = ['F12'] #'F6', 'F7', 'F10', 'F11',

    k = 2
    m = 4

    # test_var_int('F6')

    solution = np.array([54.17907981066105, -81.29932796440022, -54.83959174207751, -74.26855949545245, 67.3197270995056, 93.00900939377559,
     -45.50648254930236, 47.179106977536094, -84.77226689955808, -53.331812351900695, -6.47687692796886,
     25.705557128737325, 36.86484832305888, 71.76418298810756, -23.321800501231877, -82.65536763451573,
     54.21870582334469, 94.29977725867164, 52.546584440881105, 21.82192675552521])

    print(solution)

    bench = Benchmark()

    from numpy.random import rand
    info = bench.get_info(1)
    sol = info['lower'] + rand(200) * (info['upper'] - info['lower'])
    f = bench.get_function(12)
    print(sol)
    print(f(sol))

    '''
    for function_name in function_names:
        #test_var_int(function_name)

        #test_diff_grouping(function_names)

        dimensions = [20]
        test_optimization(dimensions, function_names)
    '''

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
