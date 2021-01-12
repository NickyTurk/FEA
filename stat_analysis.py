import networkx as nx
import pandas as pd
import re, os
import matplotlib.pyplot as plt
import read_data
import numpy as np
from scipy.stats import ttest_ind
from itertools import chain
import collections
from ast import literal_eval

import colorsys
import itertools

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class OptimizationAnalysis():

    def __init__(self, dim= 20, functions = ['F3', 'F5', 'F11']):
        self.dim = dim
        self.functions = functions

    def avg_fitness(self):
        # pso = read_data.transform_files_to_df("F*_pso_param.csv", subfolder = "pso_"+str(self.dim))
        # fea = read_data.transform_files_to_df("F*_dim" + str(self.dim) + "overlapping_diff_grouping_small_epsilon.csv", subfolder = "FEA_PSO", header = False)
        # ccea = read_data.transform_files_to_df("F*_dim" + str(self.dim) + "m4_diff_grouping_small_epsilon.csv", subfolder = "FEA_PSO", header = False)

        for f in self.functions:
            pso = read_data.transform_files_to_df(f + "_pso_param.csv", subfolder = "pso_"+str(self.dim))
            fea = read_data.transform_files_to_df(f + "_dim" + str(self.dim) + "overlapping_diff_grouping_small_epsilon.csv", subfolder = "FEA_PSO", header = False)
            ccea = read_data.transform_files_to_df(f + "_dim" + str(self.dim) + "m4_diff_grouping_small_epsilon.csv", subfolder = "FEA_PSO", header = False)
            # spectral = read_data.transform_files_to_df(f + "_dim" + str(self.dim) + "spectral.csv", subfolder="FEA_PSO", header=False)

            print(f)
            pso_values = pso.loc[pso['function'] == f]
            avg_fitness =  []
            for fitness in pso_values['fitnesses']:
                avg_fitness.append(np.mean(fitness))
            min_pso_idx = avg_fitness.index(min(avg_fitness))
            min_pso_avg = min(avg_fitness)
            pso_row = pso_values.iloc[[min_pso_idx]]
            pso_fitnesses = pso_row['fitnesses'].to_numpy()[0]
            pso_std = np.std(pso_fitnesses)
            print('pso: ', min_pso_avg, 'pso std: ', pso_std)

            fea_values = fea.loc[fea['function'] == f]
            fea_fitnesses = np.array(fea_values.iloc[0,0:9])
            fea_avg = np.mean(fea_fitnesses)
            fea_std = np.std(fea_fitnesses)
            print('fea odg: ', fea_avg, 'fea odg std: ', fea_std)

            ccea_values = ccea.loc[ccea['function'] == f]
            ccea_fitnesses = np.array(ccea_values.iloc[0,0:9])
            ccea_avg = np.mean(ccea_fitnesses)
            ccea_std = np.std(ccea_fitnesses)
            print('ccea dg: ', ccea_avg, 'ccea dg std: ', ccea_std)

            # spectral_values = spectral.loc[ccea['function'] == f]
            # spectral_fitnesses = np.array(spectral_values.iloc[0, 0:9])
            # spectral_avg = np.mean(spectral_fitnesses)
            # spectral_std = np.std(spectral_fitnesses)
            # print('spectral dg: ', spectral_avg, 'spectral std: ', spectral_std)

            print('pso vs fea: ', ttest_ind(pso_fitnesses, fea_fitnesses))
            print('pso vs ccea: ', ttest_ind(pso_fitnesses, ccea_fitnesses))
            print('ccea vs fea: ', ttest_ind(ccea_fitnesses, fea_fitnesses))

            # print('spectral vs pso: ',  ttest_ind(pso_fitnesses, spectral_fitnesses))
            # print('spectral vs fea: ', ttest_ind(fea_fitnesses, spectral_fitnesses))
            # print('spectral vs ccea: ', ttest_ind(ccea_fitnesses, spectral_fitnesses))



class FactorAnalysis():

    def __init__(self, filename):
        self.filename = filename
        self.functions_run = ['F3', 'F5', 'F11', 'F17', 'F19']
        pass

    def get_frame_and_attributes(self):
        frame = read_data.transform_files_to_df(self.filename)

        epsilons = frame.EPSILON.unique()
        functions = frame.FUNCTION.unique()
        dimensions = frame.DIMENSION.unique()

        return frame, epsilons, functions, dimensions

    def factor_stats_per_function(self):
        frame, epsilons, functions, dimensions = self.get_frame_and_attributes()
        for f in functions:
            if f in self.functions_run:
                print('function: ', f)
                fion_frame = frame.loc[frame['function'] == f]
                print(np.mean(fion_frame['NR_GROUPS'].values.tolist()))
                print(np.std(fion_frame['NR_GROUPS'].values.tolist()))

    def overlap_in_factors(self, dimension, epsilon):
        factors, fion_name = read_data.import_single_function_factors(self.filename, dimension, epsilon)
        for i, factor in enumerate(factors):
            print('factor: ', i)
            for j, fac2 in enumerate(factors[i+1:]):
                overlap = list(set(factor) & set(fac2))
                if overlap:
                    print('overlapping factor: ', i+j, 'overlapping variables: ', overlap)

    def overlap_element_count(self, dimension, epsilon):
        factors, fion_name = read_data.import_single_function_factors(self.filename, dimension, epsilon)
        chained_factors = list(chain(*factors))
        counts = collections.Counter(chained_factors)
        print(counts)


    # Fac is string of format [(0,1,2),(3,4),(4,5,6,7)]
    # Turn this into list of tuples
    def parse_factors(self, fac):
        tups = re.findall("(?:\([^)]+\)\s*)+", fac)  # gets the tuples copied from internet https://stackoverflow.com/questions/51965798/python-regular-expressions-extract-a-list-of-tuples-after-a-keyword-from-a-text
        # tups is list of form ['(a,b,c)', '(d,e,f)']
        l = [] # list of tuples
        for t in tups:
            tup = map(int, t.strip('(),').split(',')) # get rid of parenthesis and split, and applies int function
            tup = tuple(tup)
            l.append(tup)
        return l


    def parse_file(self, file):
        filename = "results\\" + file + "_m4_diff_grouping_small_epsilon.csv"

        dataframe = pd.read_csv(filename)
        # print(dataframe[2])
        # print(dataframe['DIMENSION'])

        small_ep = dataframe.loc[dataframe['EPSILON'] == min(dataframe['EPSILON'])]
        for indx, row in small_ep.iterrows():
            factors = row['FACTORS']
            groups = self.parse_factors(factors)
            num_factors = int(row['DIMENSION'])

            group_nums = [[] for _ in range(num_factors)]  # make 2d list for each variable
            for var in range(num_factors): # go through variables
                for i in range(len(groups)):  # go through factors
                    if var in groups[i]:  # check if the variable is in the factor
                        group_nums[var].append(i + 1)

            # now need to flatten into (x, y) points
            x = []
            y = []
            for var in range(len(group_nums)):
                for g in group_nums[var]:
                    x.append(var)
                    y.append(g)

            plt.scatter(x, y)
            plt.title("Group Structure")
            plt.xlabel('Variable')
            plt.ylabel('Group')

            dir = "results\\plots\\" + file
            mkdir(dir)
            plt.savefig(dir + "\\dim_" + str(num_factors) + ".png")

            plt.show()


            # save as csv as well
            merge = [(x[i], y[i]) for i in range(len(x))]
            csv_out = "Variable, Group"
            for x, y in merge:
                csv_out += str(x) + "," + str(y) + '\n'

            file_out = open(dir + "\\dim_" + str(num_factors) + ".csv", 'w')
            file_out.write(csv_out)
            file_out.close()

    # df is the dataframe with the factors
    def graph_factors(self, df):

        for indx, row in df.iterrows():
            factors = row['FACTORS']
            groups = self.parse_factors(factors)
            dims = int(row['DIMENSION'])

            # make different colors for each of our factors
            HSV_tuples = [(x*1.0/len(groups), 0.5, 0.5) for x in range(len(groups))]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

            G = nx.Graph()
            f_edges = []
            for f in groups:
                print(f)
                G.add_nodes_from(f)
                G.add_edges_from(itertools.combinations(f, 2))  # Fully connect the factor

                # store the edges for each factor so can color them later
                f_edges.append(list(itertools.combinations(f, 2)))


            # DRAW!!
            # plt.figure(1, figsize=(50,50))

            pos = nx.random_layout(G)

            options = {"node_size": 500, "alpha": 0.8}
            for f in groups:  # draw nodes
                print(f)
                nx.draw_networkx_nodes(G, pos, nodelist=f, **options)

            for i in range(len(groups)):  # draw edges
                e = f_edges[i]
                c = [colors[i]]
                nx.draw_networkx_edges(G, pos, edgelist=e, width=4, alpha=0.8, edge_color=c)

            labels = {}
            for d in range(dims):
                labels[d] = str(d)

            nx.draw_networkx_labels(G, pos, labels, font_size=16)


            plt.axis("off")
            plt.show()
            print()
            # break



    # for i in range(20):
    #     s = "F" + str(i + 1)
    #     print(s)
    #     parse_file(s)


    """
    Factor analysis:
    Number of Factors
    Variables per Factor (size) min max avg sd
    num factors var belongs to (min max avg sd)
    size of overlap (pairwise)
    """

if __name__ == '__main__':
    # optimization = OptimizationAnalysis()
    # optimization.avg_fitness()

    fctAnl = FactorAnalysis("factors/F11_overlapping_diff_grouping_small_epsilon.csv")
    #fctAnl.factor_stats_per_function()
    #fctAnl.overlap_in_factors(50, 0.001)
    fctAnl.overlap_element_count(20, 0.001)

    """
    filename = "results\\factors\\" + "F1" + "_m4_diff_grouping_small_epsilon.csv"

    dataframe = pd.read_csv(filename)
    # print(dataframe[2])
    # print(dataframe['DIMENSION'])

    small_ep = dataframe.loc[dataframe['EPSILON'] == min(dataframe['EPSILON'])]
    fa = FactorAnalysis()
    fa.graph_factors(small_ep)
    """