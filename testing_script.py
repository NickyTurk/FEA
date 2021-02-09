from topology import *
from read_data import *
from pso import pso
from fea_pso import fea_pso
from variable_interaction import MEE
from evaluation import *
from clustering import *

from datetime import datetime
import numpy as np
from opfunu.cec.cec2010.function import *
from functools import partial
import csv


def get_function(function_name, function_idx, dim):
    cec2010_functions = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20]
    no_m_param = ['F1', 'F2', 'F3', 'F19', 'F20']
    shifted_error_function = ['F14', 'F15', 'F16']
    m = 4

    if function_name in no_m_param:
        f = cec2010_functions[function_idx]
    elif function_name in shifted_error_function:
        f = partial(cec2010_functions[function_idx], m_group=dim)
    else:
        f = partial(cec2010_functions[function_idx], m_group=m)

    return f


class TestDecomposition():

    def __init__(self, dim, function_number):
        self.dim = dim
        self.function_name = 'F' + str(function_number)
        self.function_idx = function_number - 1

        self.f = get_function(self.function_name, self.function_idx, self.dim)

    def test_fuzzy_mee(self, mic_thr=0.1, de_thr=0.1, fuzzy_cluster_threshold=0.2):
        ub = np.ones(self.dim) * 100
        lb = np.ones(self.dim) * -100
        a = mic_thr  # mic threshold
        b = de_thr  # de threshold
        delta = 0.000001  # account for variations
        sample_size = self.dim * 4

        # caluclate MEE
        mee = MEE(self.f, self.dim, ub, lb, sample_size, a, b, delta)
        mee.direct_IM()

        mee.strongly_connected_comps()

        # run spectral on adjacency/interaction matrix
        spectral = Spectral(np.array(mee.IM), 4)
        spectral.assign_clusters()
        spectral_factors = spectral.return_factors(spectral.soft_clusters, threshold=fuzzy_cluster_threshold)
        return spectral_factors


class TestOptimization():

    def __init__(self, dim, function_number, factor_topology='DG', DG_epsilon=0):
        self.dim = dim
        self.function_name = 'F' + str(function_number)
        self.function_idx = function_number - 1
        self.DG_epsilon = DG_epsilon
        if factor_topology == 'DG':
            self.file_extension = "m4_diff_grouping_small_epsilon"
            self.filename = "results/factors/" + self.function_name + "_" + self.file_extension + ".csv"
        elif factor_topology == 'ODG':
            self.file_extension = "overlapping_diff_grouping_small_epsilon"
            self.filename = "results/factors/" + self.function_name + "_" + self.file_extension + ".csv"
        elif factor_topology == 'spectral':
            self.DG_epsilon = 0
            self.file_extension = "spectral"
            self.filename = "results/spectral_factors/" + self.function_name + "_" + self.file_extension + ".csv"
        elif factor_topology == 'fuzzy_spectral':
            self.DG_epsilon = 0
            self.file_extension = "fuzzy_spectral"
            self.filename = "results/factors/" + self.function_name + "_" + self.file_extension + ".csv"

        self.f = get_function(self.function_name, self.function_idx, self.dim)

        self.domain = [-50, 50]

    def harness(self, algorithm, iterations=10, repeats=1):
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

    def get_factor_info(self, factors, d):
        arbiters = nominate_arbiters(factors)
        optimizers = calculate_optimizers(d, factors)
        neighbors = determine_neighbors(factors)
        return arbiters, optimizers, neighbors

    def test_fea(self, pso_iterations, pop, fea_iterations):
        factors, function_name = import_single_function_factors(self.filename, self.dim, epsilon=self.DG_epsilon)
        arbiters, optimizers, neighbors = self.get_factor_info(factors, self.dim)
        algorithm = lambda: fea_pso(self.f, self.dim, self.domain, factors, optimizers, pop, fea_iterations,
                                    lambda t, s: t == pso_iterations)
        summary = self.harness(algorithm)
        return summary

    def test_pso(self, pop, iterations):
        algorithm = lambda: pso(self.f, pop, self.dim, self.domain, lambda t, f: t == iterations)
        summary = self.harness(algorithm, iterations=5)
        return summary


if __name__ == '__main__':
    function_nrs = [19]
    dim = [50]  # dim
    thr = 0.2 #fuzzy threshold
    for nr in function_nrs:
        # with open('results/factors/' + 'F'+str(nr) + '_fuzzy_spectral.csv', 'w') as csv_write:
        #     for d in dim:
        #         test_decomp = TestDecomposition(dim=d, function_number=nr)
        #         csv_writer = csv.writer(csv_write)
        #         csv_writer.writerow(['FUNCTION', 'DIMENSION', 'THRESHOLD', 'NR_GROUPS', 'FACTORS', 'SEPARATE VARS'])
        #         factors, sep_vars = test_decomp.test_fuzzy_mee(fuzzy_cluster_threshold=thr)
        #         csv_writer.writerow([test_decomp.function_name, str(d), str(thr), len(factors), factors, sep_vars])
        for d in dim:
            test_opt = TestOptimization(dim=d, function_number=nr, factor_topology='fuzzy_spectral', DG_epsilon=0.001)
            # with open('results/pso_20/' + str(test_opt.function_name) + '_pso_param.csv', 'a') as write_to_csv:
            #     csv_writer = csv.writer(write_to_csv)
            #     csv_writer.writerow(['function', 'dim', 'population', 'iterations', 'fitnesses', 'stats'])
            #     for pop in [500]:
            #         print('function nr: ', nr)
            #         summary = test_opt.test_pso(pop, 200)
            #         to_write = [str(test_opt.function_name), str(test_opt.dim), str(pop), str(200), summary["fitnesses"], summary["statistics"]]
            #         csv_writer.writerow(to_write)

            with open('results/FEA_PSO/' + str(test_opt.function_name) + '_dim' + str(
                    test_opt.dim) + test_opt.file_extension + ".csv", 'a') as write_to_csv:
                print('function nr: ', nr)
                summary = test_opt.test_fea(pso_iterations=10, pop=500, fea_iterations=10)
                csv_writer = csv.writer(write_to_csv)
                csv_writer.writerow(summary["fitnesses"])
