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

from variable_interaction import MEE
from numpy import linalg as la

class TestOptimization():

    def __init__(self, dim, function_number, factor_topology = 'DG', DG_epsilon = 0):
        self.dim = dim
        self.cec2010_functions = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, 15, F16, F17, F18, F19, F20]
        self.function_name = 'F' + str(function_number)
        self.function_idx = function_number-1
        self.DG_epsilon = DG_epsilon
        if factor_topology == 'DG':
            self.file_extension = "m4_diff_grouping_small_epsilon"
            self.filename = "results/factors/" + self.function_name + "_" + self.file_extension + ".csv"
        elif factor_topology == 'ODG': 
            self.file_extension = "overlapping_diff_grouping_small_epsilon"
            self.filename = "results/factors/" + self.function_name + "_" + self.file_extension + ".csv"
        elif factor_topology == 'spectral':
            self.file_extension = "spectral"
            self.filename = "results/spectral_factors/" + self.function_name + "_" + self.file_extension + ".csv"

        no_m_param = ['F1', 'F2', 'F3', 'F19', 'F20']
        shifted_error_function = ['F14', 'F15', 'F16']
        m=4

        if self.function_name in no_m_param:
            self.f = self.cec2010_functions[self.function_idx]
        elif self.function_name in shifted_error_function:
            self.f = partial(self.cec2010_functions[self.function_idx], m_group=self.dim)
        else:
            self.f = partial(self.cec2010_functions[self.function_idx], m_group=m) 

        self.domain = [-50, 50]
    
    def harness(self,algorithm, iterations, repeats=2):
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
        algorithm = lambda: fea_pso(self.f, self.dim, self.domain, factors, optimizers, pop, fea_iterations, lambda t, s: t == pso_iterations)
        summary = self.harness(algorithm, fea_iterations, 10)
        return summary
    
    def test_pso(self, pop, iterations):
        algorithm = lambda: pso(self.f, pop, self.dim, self.domain, lambda t, f: t == iterations)
        summary = self.harness(algorithm, iterations, 10)
        return summary

if __name__ == '__main__':
    function_nrs = [5,11,17,19]
    for nr in function_nrs:
        test_opt = TestOptimization(dim=20, function_number=nr, factor_topology='ODG', DG_epsilon=0.001) 
        
        with open('results/pso_20/' + str(test_opt.function_name) + '_pso_param.csv', 'a') as write_to_csv:
            csv_writer = csv.writer(write_to_csv)
            csv_writer.writerow(['function', 'dim', 'population', 'iterations', 'fitnesses', 'stats'])
            for pop in [500]:
                print('function nr: ', nr)
                summary = test_opt.test_pso(pop, 200)
                to_write = [str(test_opt.function_name), str(test_opt.dim), str(pop), str(200), summary["fitnesses"], summary["statistics"]]
                csv_writer.writerow(to_write)
        
        """
        with open('results/FEA_PSO/' + str(test_opt.function_name) + '_dim' + str(test_opt.dim) + test_opt.file_extension + ".csv", 'a') as write_to_csv:
            print('function nr: ', nr)
            summary = test_opt.test_fea(pso_iterations=10, pop=500, fea_iterations=10)
            csv_writer = csv.writer(write_to_csv)
            csv_writer.writerow(summary["fitnesses"])
        """
