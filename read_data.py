import pandas as pd
import numpy as np
import glob
from ast import literal_eval


def transform_files_to_df(file_regex):
    all_files = get_files_list(file_regex)

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)


def get_files_list(file_regex):
    path = r'./results'
    return glob.glob(path + '/' + file_regex)



def import_single_function_factors(file_name, dim=50, epsilon=0):

    frame = pd.read_csv(file_name, header=0)
    dim_frame = frame.loc[frame['DIMENSION'] == dim]
    fion_name = frame['FUNCTION'].unique()
    dim_array = np.array(dim_frame['FACTORS'])

    if epsilon ==0:
        index = dim_frame['NR_GROUPS'].argmax()
        factors = literal_eval(dim_array[index])
    else:
        epsilon_row = dim_frame.loc[dim_frame['EPSILON'] == epsilon]
        factors = literal_eval(np.array(epsilon_row['FACTORS'])[0])

    return factors, fion_name[0]


if __name__ == '__main__':
    transform_files_to_df("F*_m4_diff_grouping_small_epsilon.csv")
    # import_single_function_factors("F1_m4_diff_grouping_small_epsilon.csv", 50)

