import os.path

import pandas as pd
import numpy as np
import glob, re
from ast import literal_eval


class DataReader:
    def __init__(self, file_regex):
        self.file_regex = file_regex
        self.path_to_files = self.get_files_list()  # array of all files and their path that match the regex or string

    def transform_files_to_df(self, header=True):
        li = []

        for filename in self.path_to_files:
            if header:
                df = pd.read_csv(filename, index_col=None, header=0, converters={'fitnesses': eval, 'fitness': eval})
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

    def get_files_list(self):
        result = []
        search_path = os.path.dirname(os.path.abspath('.gitignore'))
        for root, dir, files in os.walk(search_path):
            if self.file_regex in files:
                result.append(os.path.join(root, self.file_regex))
        return result

    def import_factors(self, dim=50, epsilon=0):
        frame = pd.read_csv(self.filepath, header=0)
        frame.columns = map(str.upper, frame.columns)
        frame = frame.rename(columns={"DIM": "DIMENSION"}, errors="ignore")
        dim_frame = frame.loc[frame['DIMENSION'] == int(dim)]
        fion_name = frame['FUNCTION'].unique()
        dim_array = np.array(dim_frame['FACTORS'])

        if epsilon == 0:
            index = dim_frame['NR_GROUPS'].argmax()
            factors = literal_eval(dim_array[index])
        else:
            epsilon_row = dim_frame.loc[dim_frame['EPSILON'] == epsilon]
            factors = literal_eval(np.array(epsilon_row['FACTORS'])[0])

        return factors, fion_name[0]