import networkx as nx
import pandas as pd
import re, os
import matplotlib.pyplot as plt
import read_data
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from itertools import chain
import collections

import colorsys
import itertools

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class OptimizationAnalysis():

    def __init__(self, dim= 50, functions = ['F3', 'F5', 'F11', 'F17','F19']): #'F3', 'F5', 'F11', 'F17'
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
            spectral = read_data.transform_files_to_df(f + "_dim" + str(self.dim) + "spectral.csv", subfolder="FEA_PSO", header=False)

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

            ccea_values = ccea.loc[ccea['function'] == f]
            ccea_fitnesses = np.array(ccea_values.iloc[0,0:9])
            ccea_avg = np.mean(ccea_fitnesses)
            ccea_std = np.std(ccea_fitnesses)
            print('ccea dg: ', ccea_avg, 'ccea dg std: ', ccea_std)

            fea_values = fea.loc[fea['function'] == f]
            fea_fitnesses = np.array(fea_values.iloc[0,0:9])
            fea_avg = np.mean(fea_fitnesses)
            fea_std = np.std(fea_fitnesses)
            print('fea odg: ', fea_avg, 'fea odg std: ', fea_std)

            spectral_values = spectral.loc[ccea['function'] == f]
            spectral_fitnesses = np.array(spectral_values.iloc[0, 0:9])
            spectral_avg = np.mean(spectral_fitnesses)
            spectral_std = np.std(spectral_fitnesses)

            print('ANOVA: \n', f_oneway(pso_fitnesses, ccea_fitnesses, fea_fitnesses, spectral_fitnesses))

            print('spectral dg: ', spectral_avg, 'spectral std: ', spectral_std)

            print('pso vs fea: ', ttest_ind(pso_fitnesses, fea_fitnesses))
            print('pso vs ccea: ', ttest_ind(pso_fitnesses, ccea_fitnesses))
            print('ccea vs fea: ', ttest_ind(ccea_fitnesses, fea_fitnesses))

            print('spectral vs pso: ',  ttest_ind(pso_fitnesses, spectral_fitnesses))
            print('spectral vs fea: ', ttest_ind(fea_fitnesses, spectral_fitnesses))
            print('spectral vs ccea: ', ttest_ind(ccea_fitnesses, spectral_fitnesses))

class FactorAnalysis():

    def __init__(self):
        self.functions_run = ['F3', 'F5', 'F11', 'F17', 'F19']
        self.means_per_method = []
        self.group_sizes_per_method = []
        self.overlap_sizes_per_method = []

    def boxplot_sizes(self, sizes, title):
        '''
        https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
        :return:
        '''

        ticks = self.functions_run

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)

        plt.figure()

        data_b = sizes[0]
        bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * len(sizes), sym='', widths=.8)
        set_box_color(bpr, '#d95f02')
        # plt.plot([], c='#d95f02', label='ODG')

        # data_c = sizes[1]
        # bpm = plt.boxplot(data_c, positions=np.array(range(len(data_c))) * len(sizes), sym='', widths=0.5)
        # set_box_color(bpm, '#7570b3')
        # plt.plot([], c='#7570b3', label='M-SD')

        # if len(sizes) > 2:
        #     data_a = sizes[2]
        #     bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a))) * len(sizes) + .8, sym='', widths=0.5)
        #     set_box_color(bpl, '#1b9e77')  # colors are from http://colorbrewer2.org/
        #     plt.plot([], c='#1b9e77', label='DG')

        # plt.legend(loc='upper center')
        plt.title(title)

        plt.xticks(range(0, len(ticks) * len(sizes), len(sizes)), ticks)
        plt.xlim(-2, len(ticks) * len(sizes))
        plt.tight_layout()
        plt.show()

    def barplot_avgs(self):

        x = np.arange(len(self.functions_run))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, self.means_per_method[0], width, label='CCEA DG')
        rects2 = ax.bar(x + width / 2, self.means_per_method[1], width, label='FEA ODG')
        #rects3 = ax.bar(x + width / 2, self.means_per_method[2], width, label='FEA Spectral')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Mean')
        ax.set_title('Mean by function and methods')
        ax.set_xticks(x)
        ax.set_xticklabels(self.functions_run)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        #autolabel(rects3)

        fig.tight_layout()

        plt.show()

    def get_frame_and_attributes(self, filename_extension):
        frame = read_data.transform_files_to_df('factors/F*'+filename_extension)

        epsilons = frame.EPSILON.unique()
        functions = frame.FUNCTION.unique()
        dimensions = frame.DIMENSION.unique()

        return frame, epsilons, functions, dimensions

    def factor_nr_groups_per_function(self, filename_extension):
        frame, epsilons, functions, dimensions = self.get_frame_and_attributes(filename_extension)
        for f in functions:
            if f in self.functions_run:
                print('\n---------------------------------------------------\n')
                print('function: ', f)
                print('\n---------------------------------------------------\n')
                fion_frame = frame.loc[frame['function'] == f]
                print('across all epsilons and dimensions\n--------------------------------------\n')
                print('avg nr groups: ', np.mean(fion_frame['NR_GROUPS'].values.tolist()))
                print('stdev of nr groups: ', np.std(fion_frame['NR_GROUPS'].values.tolist()))
                print('\n---------------------------------------------------\n')

    def run_factor_stats_per_function(self, filename_extension, dimension=20, epsilon=0):
        group_avgs = []
        group_sizes = []
        overlap_sizes = []
        for function in self.functions_run:
            stats_dict = self.run_factor_stats_one_function(filename_extension, function, dimension, epsilon)
            group_avgs.append(stats_dict.get('group_size_avg'))
            group_sizes.append(stats_dict.get('group_sizes'))
            overlap_sizes.append(stats_dict.get('overlap_sizes'))
        self.means_per_method.append(group_avgs)
        self.group_sizes_per_method.append(group_sizes)
        self.overlap_sizes_per_method.append(overlap_sizes)

    def run_factor_stats_one_function(self, filename_extension, function, dimension=50, epsilon=0):
        print('stats for function ', function, ' with dimension ', str(dimension), ' and epsilon ', str(epsilon),
              '\n---------------------------------------------------\n')
        factors, fion_name = read_data.import_single_function_factors('results/factors/' + function + filename_extension, dimension, epsilon)
        group_size_avg, group_size_std, group_sizes = self.factor_group_size(factors)
        overlap_size_avg = 0
        overlap_sizes = []
        if 'overlapping' in filename_extension:
            print('check overlap for', filename_extension)
            overlap_size_avg, overlap_size_std, overlap_sizes = self.overlap_in_factors(factors)
            overlap_count_by_element = self.overlap_element_count(factors)
        return {'group_size_avg': round(group_size_avg, 2), 'group_sizes': group_sizes, 'overlap_size_avg': overlap_size_avg, 'overlap_sizes': overlap_sizes}

    def factor_group_size(self, factors):
        group_sizes = []
        for f in factors:
            group_sizes.append(len(f))
        avg = np.mean(group_sizes)
        std = np.std(group_sizes)
        print('avg group size: ', avg)
        print('stdev of group size: ', std)
        return avg, std, group_sizes


    def overlap_in_factors(self, factors):
        overlap_sizes = []
        for i, factor in enumerate(factors):
            # print('factor: ', i)
            for j, fac2 in enumerate(factors[i+1:]):
                overlap = list(set(factor) & set(fac2))
                if overlap:
                    # print('overlapping factor: ', i+j, 'overlapping variables: ', overlap)
                    overlap_sizes.append(len(overlap))
        return np.mean(overlap_sizes), np.std(overlap_sizes), overlap_sizes

    def overlap_element_count(self, factors):
        chained_factors = list(chain(*factors))
        counts = collections.Counter(chained_factors)
        print(counts)
        return counts


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
    # df is the dataframe with the factors
    def factor_stats(self, df):
        output = []

        num_factors = []
        variables_p_factor = []
        num_factors_var_in = []
        size_overlap_vars = []
        size_overlap_factors = []

        for indx, row in df.iterrows():
            groups = row['FACTORS']
            factors = self.parse_factors(groups)
            dims = int(row['DIMENSION'])

            num_factors.append(len(factors))

            factor_sizes = [len(f) for f in factors]
            variables_p_factor.append((min(factor_sizes), max(factor_sizes), sum(factor_sizes)/len(factor_sizes), np.std(factor_sizes)))

            var_in = []

            for var in range(dims):
                var_in.append(0)
                for f in factors:
                    if var in f:
                        var_in[var] += 1

            num_factors_var_in.append((min(var_in), max(var_in), sum(var_in)/len(var_in), np.std(var_in)))

            overlaps = []
            for i,f in enumerate(factors):
                overlaps.append((0,0))
                for g in factors:
                    if g == f:
                        continue
                    set_f = set(f)
                    set_g = set(g)
                    intersect = set_f.intersection(set_g)
                    if len(intersect) == 0:
                        continue
                    t = overlaps[i]
                    t[0] += len(intersect)  # number overlapping variables
                    t[1] += 1  # number of factors overlap with
                    overlaps[i] = t

            num_overlap_vars = [x[0] for x in overlaps]
            num_overlap_factors = [x[1] for x in overlaps]

            size_overlap_vars.append((min(num_overlap_vars), max(num_overlap_vars), sum(num_overlap_vars)/len(num_overlap_vars), np.std(num_overlap_vars)))
            size_overlap_factors.append((min(num_overlap_factors), max(num_overlap_factors), sum(num_overlap_factors)/len(num_overlap_factors), np.std(num_overlap_factors)))


            # Num Factors, Vars/fac min, max, avg, sd, num factors var min, max, avg, sd, size overlap vars min, max, avg, sd, size overlap factors min, max, avg, sd
            out = [num_overlap_factors,
                      variables_p_factor[0],variables_p_factor[1], variables_p_factor[2], variables_p_factor[3],
                      num_factors_var_in[0],num_factors_var_in[1], num_factors_var_in[2], num_factors_var_in[3],
                      size_overlap_vars[0],size_overlap_vars[1], size_overlap_vars[2], size_overlap_vars[3],
                      size_overlap_factors[0],size_overlap_factors[1], size_overlap_factors[2], size_overlap_factors[3]]
            output.append(out)

        # Output results into a dataframe

        # Can copy df if want so not modifying, but is not used elsewhere so we can modify directly
        
        df['Number Factors'] = [x[0] for x in output]
        df['Min Vars per Factor'] = [x[1] for x in output]
        df['Max Vars per Factor'] = [x[2] for x in output]
        df['Avg Vars per Factor'] = [x[3] for x in output]
        df['Std Vars per Factor'] = [x[4] for x in output]
        df['Min Num Factors Var is in'] = [x[5] for x in output]
        df['Max Num Factors Var is in'] = [x[6] for x in output]
        df['Avg Num Factors Var is in'] = [x[7] for x in output]
        df['Std Num Factors Var is in'] = [x[8] for x in output]
        df['Min Num Overlapping Vars per Factor'] = [x[9] for x in output]
        df['Max Num Overlapping Vars per Factor'] = [x[10] for x in output]
        df['Avg Num Overlapping Vars per Factor'] = [x[11] for x in output]
        df['Std Num Overlapping Vars per Factor'] = [x[12] for x in output]
        df['Min Num Overlapping Factors'] = [x[13] for x in output]
        df['Max Num Overlapping Factors'] = [x[13] for x in output]
        df['Avg Num Overlapping Factors'] = [x[14] for x in output]
        df['Std Num Overlapping Factors'] = [x[15] for x in output]

        return df


if __name__ == '__main__':
    optimization = OptimizationAnalysis(dim=20)
    optimization.avg_fitness()

    # fctAnl = FactorAnalysis()
    # fctAnl.run_factor_stats_per_function("_overlapping_diff_grouping_small_epsilon.csv", dimension=50, epsilon=0)
    # fctAnl.run_factor_stats_per_function("_spectral.csv", dimension=20, epsilon=0)
    # fctAnl.run_factor_stats_per_function("_m4_diff_grouping_small_epsilon.csv", dimension=20, epsilon=0.001)
    # fctAnl.boxplot_sizes(fctAnl.overlap_sizes_per_method, 'Overlap Size for 50 Dimensions')

    """
    filename = "results\\factors\\" + "F1" + "_m4_diff_grouping_small_epsilon.csv"

    dataframe = pd.read_csv(filename)
    # print(dataframe[2])
    # print(dataframe['DIMENSION'])

    small_ep = dataframe.loc[dataframe['EPSILON'] == min(dataframe['EPSILON'])]
    fa = FactorAnalysis()
    fa.graph_factors(small_ep)
    """