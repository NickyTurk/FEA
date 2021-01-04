import pandas as pd
import numpy as np
import glob, re
from ast import literal_eval


def transform_files_to_df(file_regex, subfolder = '', header = True):
    all_files = get_files_list(file_regex, subfolder)

    li = []
    print(all_files)

    for filename in all_files:
        if header:
            df = pd.read_csv(filename, index_col=None, header=0, converters={'fitnesses': eval})
            if 'function' not in df.columns:
                function_nr = re.findall(r"F([0-9]+?)(?=\_)", filename) 
                f_int = ''.join(list(filter(str.isdigit, function_nr[0])))
                df['function'] = 'F' + f_int
            li.append(df)
        else:
            df = pd.read_csv(filename, index_col=None, header=None)
            function_nr = re.findall(r"F([0-9]+?)(?=\_)", filename) 
            f_int = ''.join(list(filter(str.isdigit, function_nr[0])))
            df['function'] = 'F' + f_int
            li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)


def get_files_list(file_regex, subfolder = ''):
    if not subfolder:
        path = r'./results'
    else:
        path = './results/' + subfolder
    return glob.glob(path + '/' + file_regex)


def import_single_function_factors(file_name, dim=50, epsilon=0):

    frame = pd.read_csv(file_name, header=0)
    frame.columns = map(str.upper, frame.columns)
    frame = frame.rename(columns = {"DIM": "DIMENSION"}, errors="raise")
    dim_frame = frame.loc[frame['DIMENSION'] == dim]
    fion_name = frame['FUNCTION'].unique()
    dim_array = np.array(dim_frame['FACTORS'])

    if epsilon == 0:
        index = dim_frame['NR_GROUPS'].argmax()
        factors = literal_eval(dim_array[index])
    else:
        epsilon_row = dim_frame.loc[dim_frame['EPSILON'] == epsilon]
        factors = literal_eval(np.array(epsilon_row['FACTORS'])[0])

    return factors, fion_name[0]


if __name__ == '__main__':
    #transform_files_to_df("F*_m4_diff_grouping_small_epsilon.csv")
    # import_single_function_factors("F1_m4_diff_grouping_small_epsilon.csv", 50)
    print(transform_files_to_df("F*_pso_param.csv", subfolder = "pso_50"))

