import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

import read_data
from read_data import *
from ast import literal_eval


def get_frame_and_attributes(filename):
    frame = transform_files_to_df(filename, subfolder = 'factors')

    epsilons = frame.EPSILON.unique()
    functions = frame.FUNCTION.unique()
    dimensions = frame.DIMENSION.unique()

    return frame, epsilons, functions, dimensions


def sum_nr_groups(frame, epsilons, functions, dimensions):
    sum_nr_group_epsilon_dimension = np.zeros([len(epsilons), len(dimensions)])

    for i, row in frame.iterrows():
        e = np.where(epsilons == row['EPSILON'])
        d = np.where(dimensions == row['DIMENSION'])
        sum_nr_group_epsilon_dimension[e, d] += row['NR_GROUPS']


def boxplot_group_size_dimension_per_epsilon(frame, epsilons, functions, dimensions):
    """
    PLOT OUT AVERAGE GROUP SIZE FOR EACH DIMENSION ACROSS 20 FUNCTION, FOR EACH EPSILON VALUE.
    """

    group_size_epsilon_dimension = np.zeros([len(epsilons), len(dimensions), len(functions)])
    for i, row in frame.iterrows():
        e = np.where(epsilons == row['EPSILON'])
        d = np.where(dimensions == row['DIMENSION'])
        f = np.where(functions == row['FUNCTION'])
        factors = literal_eval(row['FACTORS'])
        sum_len = 0
        for fac in factors:
            sum_len += len(fac)
        group_size_epsilon_dimension[e, d, f] = (sum_len / len(factors))

    # Create a figure instance
    fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True)
    a = plt.gca()
    a.axes.yaxis.set_ticks([])
    a.axes.xaxis.set_ticks([])
    for i, eps in enumerate(group_size_epsilon_dimension):
        # Create an axes instance
        ax = fig.add_subplot(2, 2, i + 1)
        ax.set_title('Epsilon: ' + str(epsilons[i]))
        # Create the boxplot
        bp = ax.boxplot(np.transpose(eps))
    plt.gcf().text(0.45, 0.01, '10s of Dimensions', fontsize=12)
    plt.gcf().text(0.01, 0.45, 'Group Size', fontsize=12, rotation=90)
    fig.suptitle('Group size across 20 functions')
    plt.show()


def boxplot_group_size_per_function_type(frame, epsilons, functions, dimensions):
    """
    PLOT PER FUNCTION TYPE
    """
    epsilons = epsilons[2:]

    function_types = [[0, 1, 2],  # separable_functions
                      [3, 4, 5, 6, 7],  # single_group_non_separable_functions
                      [8, 9, 10, 11, 12],  # D2m_group_non_separable_functions
                      [13, 14, 15, 16, 17],  # Dm_group_non_separable_functions
                      [18, 19]]  # non_separable_functions
    functiontype_names = ['separable_function', 'single_group_non_separable_functions',
                          'D2m_group_non_separable_functions',
                          'Dm_group_non_separable_functions', 'non_separable_functions']

    group_size_epsilon_dimension = np.zeros([len(epsilons), len(dimensions), len(functions)])

    for i, row in frame.iterrows():
        e = np.where(epsilons == row['EPSILON'])
        d = np.where(dimensions == row['DIMENSION'])
        f = np.where(functions == row['FUNCTION'])
        factors = literal_eval(row['FACTORS'])
        sum_len = 0
        for fac in factors:
            sum_len += len(fac)
        group_size_epsilon_dimension[e, d, f] = (sum_len / len(factors))

    for i, type in enumerate(function_types):
        for j, eps in enumerate(group_size_epsilon_dimension):
            group_size_function = []
            for fionidx, fion in enumerate(np.transpose(eps)):
                if fionidx in type:
                    group_size_function.append(fion)
            print(np.transpose(np.transpose(group_size_function)))
            fig, ax = plt.subplots()
            ax.set_title('Type: ' + functiontype_names[i] + ', Epsilon: ' + str(epsilons[j]))
            # Create the boxplot
            ax.boxplot(np.transpose(np.transpose(group_size_function)))
            plt.savefig('Type_' + functiontype_names[i] + '_Epsilonvalue_' + str(epsilons[j]))


def get_fits(functions, itr):
    dim = 50
    tail = '_' + str(itr) + 'itr.csv'
    ret = {}
    for f in functions:
        fea = read_data.transform_files_to_df(
            f + "_dim" + str(dim) + "overlapping_diff_grouping_small_epsilon" + tail, subfolder="FEA_PSO/40_itr",
            header=False)
        # fea = read_data.transform_files_to_df(f + "_dim" + str(self.dim) + "overlapping_diff_grouping_small_epsilon.csv", subfolder = "FEA_PSO", header = False)
        ccea = read_data.transform_files_to_df(f + "_dim" + str(dim) + "m4_diff_grouping_small_epsilon" + tail,
                                               subfolder="FEA_PSO/40_itr", header=False)
        # ccea = read_data.transform_files_to_df(f + "_dim" + str(self.dim) + "m4_diff_grouping_small_epsilon.csv", subfolder = "FEA_PSO", header = False)
        spectral = read_data.transform_files_to_df(f + "_dim" + str(dim) + "spectral" + tail,
                                                   subfolder="FEA_PSO/40_itr", header=False)
        meet = read_data.transform_files_to_df(f + "_dim" + str(dim) + "meet" + tail, subfolder="FEA_PSO/40_itr",
                                               header=False)

        ccea_values = ccea.loc[ccea['function'] == f]
        ccea_fitnesses = np.array(ccea_values.iloc[0, 0:10])
        ccea_avg = np.mean(ccea_fitnesses)
        ccea_std = np.std(ccea_fitnesses)

        fea_values = fea.loc[fea['function'] == f]
        fea_fitnesses = np.array(fea_values.iloc[0, 0:10])
        fea_avg = np.mean(fea_fitnesses)
        fea_std = np.std(fea_fitnesses)

        spectral_values = spectral.loc[ccea['function'] == f]
        spectral_fitnesses = np.array(spectral_values.iloc[0, 0:10])
        spectral_avg = np.mean(spectral_fitnesses)
        spectral_std = np.std(spectral_fitnesses)

        meet_values = meet.loc[ccea['function'] == f]
        meet_fitnesses = np.array(meet_values.iloc[0, 0:10])
        meet_avg = np.mean(meet_fitnesses)
        meet_std = np.std(meet_fitnesses)

        ret[f] = (ccea_fitnesses, fea_fitnesses, spectral_fitnesses, meet_fitnesses)
    return ret


def boxplot_fitness():
    functions = ["F19"]
    topologies = ['CCEA', 'ODG', 'spectral', 'MEET']
    # topologies = ['CCEA', 'ODG', 'spectral', 'MEET']
    iterations = [10, 20, 30, 40]
    fits = [get_fits(functions, i) for i in iterations]  # array of dicts

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    for f in functions:
        fig, axs = plt.subplots(1,2, figsize=(15,10))
        # plt.setp(axs, xticklabels=['10', '20', '30', '40'])
        max_scale = 0
        min_scale = 0
        all_points = []
        for i in range(len(topologies)):
            for itr in range(len(topologies)):
                all_points += fits[itr][f][i].tolist()
        max_scale = max(all_points)
        min_scale = min(all_points)
        scale_factor = 0.05  # adds some whitespace above max point

        # data = [fits[3][f][i] for i in range(len(topologies))]
        # axs.boxplot(data)
        # axs.set_title(f, fontsize=30)
        # axs.set_xticklabels(['CCEA', 'ODG', 'Spectral', 'MEET'])
        # fig.tight_layout()

        for i, topo in enumerate(topologies):
            if i == 0 or i == 2:
                continue
            # continue
            fig.suptitle(f, fontsize=30)
            data = [fits[itr][f][i] for itr in range(len(iterations))]
            coord = (0 if i == 1 else 1, int(i/2))
            axs[coord[0]].boxplot(data)
            axs[coord[0]].set_title(topo, fontsize=24)
            axs[coord[0]].set_xticklabels([10, 20, 30, 40])
            # axs[coord[0]].set_xticklabels(['CCEA', 'ODG', 'Spectral', 'MEET'])
            axs[coord[0]].set_ylim(min_scale - scale_factor * (max_scale - min_scale),
                                             max_scale + scale_factor * (max_scale - min_scale))

            fig.subplots_adjust(right=0.98, left=0.1)

        path = 'results/plots/40_itr/'
        plt.savefig(path + f + '_over_time.png')
        plt.show()
        # exit(1)
    pass


def turn_func_to_csv(f, top):
    functions = ["F3", "F5", "F11", "F17", "F19"]
    iterations = [10, 20, 30, 40]
    out = ''
    fits = [get_fits(functions, i) for i in iterations]
    for i in range(len(iterations)):
        data = fits[i][f][top]
        for d in data:
            out += str(d) + ','

        out += '\n'
    print(out)


if __name__ == '__main__':
    boxplot_fitness()
    # turn_func_to_csv('F17', 1)
    exit(13)

    filename = "F*_m4_diff_grouping_small_epsilon.csv"
    frame, epsilons, functions, dimensions = get_frame_and_attributes(filename)
    # boxplot_group_size_per_function_type(frame, epsilons, functions, dimensions)
    boxplot_group_size_dimension_per_epsilon(frame, epsilons, functions, dimensions)