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
from functools import partial


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


def test_diff_grouping(m, functions, function_names):
    shifted_function = ['F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F17', 'F18']
    shifted_error_function = ['F14', 'F15', 'F16']
    dimensions = np.arange(start=10, stop=101, step=10)
    epsilons = [1e-1, 1e-3, 1e-6, 1e-9]

    for i, f in enumerate(functions):
        with open('results/' + function_names[i] + '_overlapping_diff_grouping_small_epsilon.csv', 'w') as csv_write:
            csv_writer = csv.writer(csv_write)
            csv_writer.writerow(['FUNCTION', 'DIMENSION', 'EPSILON', 'NR_GROUPS', 'FACTORS', 'SEPARATE VARS'])
            for d in dimensions:
                for e in epsilons:
                    # try:
                    if function_names[i] in shifted_function:
                        factors, arbiters, optimizers, neighbors, separate_variables = generate_overlapping_diff_grouping(
                            f, d, e, m=m)  # generate_linear_topology(d, k)  generate_diff_grouping(f, d, 1)

                    elif function_names[i] in shifted_error_function:
                        factors, arbiters, optimizers, neighbors, separate_variables = generate_overlapping_diff_grouping(
                            f, d, e, m=d)  # generate_linear_topology(d, k)  generate_diff_grouping(f, d, 1)

                    else:
                        factors, arbiters, optimizers, neighbors, separate_variables = generate_overlapping_diff_grouping(
                            f, d, e)
                    print(len(factors), factors)
                    csv_writer.writerow([function_names[i], str(d), str(e), len(factors), factors, separate_variables])


def get_factor_info(factors, d):
    arbiters = nominate_arbiters(factors)
    optimizers = calculate_optimizers(d, factors)
    neighbors = determine_neighbors(factors)
    return arbiters, optimizers, neighbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test out some algies")
    parser.add_argument("--benchmark", help="pick the name of a benchmark function", default="schwefel-1.2")
    parser.add_argument("--seed", help="the random seed to use")
    args = parser.parse_args()

    seed = args.seed
    if not seed:
        seed = int(datetime.now().strftime("%S"))
        
    benchmark = args.benchmark

    functions = [F3, F7, F12, F16, F19]  # [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19,F20] _benchmarks[benchmark]["function"]
    function_names = ['F3', 'F7', 'F12', 'F16', 'F19'] #['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20']
    
    no_m_param = ['F1', 'F2', 'F3', 'F19', 'F20']

    k = 2
    m = 4
    domain = _benchmarks[benchmark]["interval"]

    # test_diff_grouping(4, functions, function_names)

    dimensions = [50, 100]
    #file_extension = "m4_diff_grouping"
    file_extension = "overlapping_diff_grouping"
    filename_list = get_files_list("F*_" + file_extension + "_small_epsilon.csv")

    for filename in filename_list:
        print(filename)
        for dim in dimensions:
            factors, function_name = import_single_function_factors(filename, dim)
            if function_name in function_names:
                print('current function ', function_name)
                arbiters, optimizers, neighbors = get_factor_info(factors, dim)

                if function_name in no_m_param:
                    f = functions[function_names.index(function_name)]
                else:
                    f = partial(functions[function_names.index(function_name)], m_group = m)  # retrieve appropriate function

                pso_stop = lambda t, s: t == 5
                p = 200
                n = 10

                algorithm = lambda: fea_pso(f, dim, domain, factors, optimizers, p, n, pso_stop)
                summary = harness(algorithm, n, 1)
                #print("G=", summary["statistics"])
                #print("G=", summary["fitnesses"])

                with open('results/FEA_PSO/' + str(function_name) + '_dim' + str(dim) + file_extension + ".csv", 'w') as write_to_csv:
                    csv_writer = csv.writer(write_to_csv)
                    csv_writer.writerow(summary["fitnesses"])
                    print('printed')

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
