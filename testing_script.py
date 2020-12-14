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

    def __init__(self, dim, function_number, factor_topology = 'DG'):
        self.dim = dim
        self.cec2010_functions = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, 15, F16, F17, F18, F19, F20]
        self.function_name = 'F' + str(function_number)
        self.function_idx = function_number-1
        if factor_topology == 'DG':
            file_extension = "m4_diff_grouping_small_epsilon"
        elif factor_topology == 'ODG': 
            file_extension = "overlapping_diff_grouping_small_epsilon"
        self.filename = "diff_grouping/" + self.function_name + "_" + file_extension + ".csv"

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
    
    def harness(self,algorithm, iterations, repeats):
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
        factors, function_name = import_single_function_factors(self.filename, self.dim)
        arbiters, optimizers, neighbors = self.get_factor_info(factors, self.dim)
        algorithm = lambda: fea_pso(self.f, self.dim, self.domain, factors, optimizers, pop, fea_iterations, lambda t, s: t == pso_iterations)
        summary = self.harness(algorithm, fea_iterations, 1)
        return summary
    
    def test_pso(self, pop, iterations):
        summary = {"name": self.function_name}
        fitnesses = []
        for i in range(10):
            result = pso(self.f, pop, self.dim, self.domain, lambda t, f: t == iterations)
            fitnesses.append(result[-1].fitness)
        bootstrap = create_bootstrap_function(250)
        replications = bootstrap(fitnesses)
        statistics = analyze_bootstrap_sample(replications)
        summary["statistics"] = statistics
        summary["fitnesses"] = fitnesses
        return summary

if __name__ == '__main__':
    test_opt = TestOptimization(dim=50, function_number=5, factor_topology='ODG') 
    summary = test_opt.test_fea(pso_iterations=10, pop=500, fea_iterations=10)