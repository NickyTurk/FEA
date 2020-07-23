from benchmarks import _benchmarks
from topology import *
from pso import pso
from fea_pso import fea_pso
import argparse
import random
from datetime import datetime
from evaluation import *
import numpy as np
import pandas, csv
from opfunu.cec.cec2010.function import *

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test out some algies")
    parser.add_argument("--benchmark", help="pick the name of a benchmark function", default="schwefel-1.2")
    parser.add_argument("--seed", help="the random seed to use")
    args = parser.parse_args()

    seed = args.seed
    if not seed:
        seed = int(datetime.now().strftime("%S"))

    benchmark = args.benchmark

    functions = [F3, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20] #_benchmarks[benchmark]["function"]
    function_names = ['F3', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20']
    dimensions = np.arange(start=100, stop=1000, step=100)
    k = 2
    domain = _benchmarks[benchmark]["interval"]
    for i,f in enumerate(functions):
        with open('results/' + function_names[i] + '_100_diff_grouping.csv', 'w') as csv_write:

            csv_writer = csv.writer(csv_write)
            csv_writer.writerow(['DIMENSION', 'NR_GROUPS', 'FACTORS', 'SEPARATE VARS'])
            for d in dimensions:
                print(d)
                factors, arbiters, optimizers, neighbors, separate_variables = generate_diff_grouping(f, d, 1e-3)   #  generate_linear_topology(d, k)  generate_diff_grouping(f, d, 1)
                csv_writer.writerow([str(d), len(factors), factors, separate_variables])
                print(len(factors))

    '''
    pso_stop = lambda t, s: t == 5
    p = 100
    n = 100
    width = d

    print("benchmark", benchmark)
    print( "seed", seed)

    # random.seed(seed)
    print("starting PSO")
    algorithm = lambda: pso(f, p*d, d, domain, lambda t, s: t == 5)
    summary = harness(algorithm, n, 1)
    print("G=", summary["statistics"])
    print("G=", summary["fitnesses"])

    with open('results/f7_pso.csv', 'w') as write_to_csv:
        csv_writer = csv.writer(write_to_csv)
        csv_writer.writerows(summary["fitnesses"])

    random.seed(seed)
    print("starting FEA")
    algorithm = lambda: fea_pso(f, d, domain, factors, optimizers, p, n, pso_stop)
    summary = harness(algorithm, n, 1)
    print("G=", summary["statistics"])
    print("G=", summary["fitnesses"])

    with open('results/f7_fea_pso_diff_group.csv', 'w') as write_to_csv:
        csv_writer = csv.writer(write_to_csv)
        csv_writer.writerows(summary["fitnesses"])

    factors, arbiters, optimizers, neighbors = generate_linear_topology(d, k)

    random.seed(seed)
    print("starting FEA")
    algorithm = lambda: fea_pso(f, d, domain, factors, optimizers, p, n, pso_stop)
    summary = harness(algorithm, n, 1)
    print("G=", summary["statistics"])
    print("G=", summary["fitnesses"])

    with open('results/f7_fea_pso_linear topology.csv', 'w') as write_to_csv:
        csv_writer = csv.writer(write_to_csv)
        csv_writer.writerows(summary["fitnesses"])
    '''