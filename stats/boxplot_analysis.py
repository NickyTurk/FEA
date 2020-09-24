import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from stats.read_data import *
from ast import literal_eval


def get_frame_and_attributes(filename):
    frame = transform_files_to_df(filename)

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
        ax.set_title('Epsilon value: ' + str(epsilons[i]))
        # Create the boxplot
        bp = ax.boxplot(np.transpose(eps))
    plt.gcf().text(0.45, 0.01, 'Dimensions', fontsize=12)
    plt.gcf().text(0.01, 0.45, 'Group Size', fontsize=12, rotation=90)
    fig.suptitle('Boxplots of group size across 20 functions')
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
            ax.set_title('Type: ' + functiontype_names[i] + ', Epsilon value: ' + str(epsilons[j]))
            # Create the boxplot
            ax.boxplot(np.transpose(np.transpose(group_size_function)))
            plt.savefig('Type_' + functiontype_names[i] + '_Epsilonvalue_' + str(epsilons[j]))


if __name__ == '__main__':
    filename = "F*_m4_diff_grouping_small_epsilon.csv"
    frame, epsilons, functions, dimensions = get_frame_and_attributes(filename)
    boxplot_group_size_per_function_type(frame, epsilons, functions, dimensions)
