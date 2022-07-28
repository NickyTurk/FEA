import os.path
from setup import ROOT_DIR

import pandas as pd
import numpy as np
import re
from ast import literal_eval


class MultiFileReader(object):
    def __init__(self, file_regex="", dir=""):
        self.file_regex = file_regex
        self.path_to_files = self.get_files_list(dir)  # array of all files and their path that match the regex or string

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

    def get_files_list(self, dir=""):
        result = []
        if dir:
            search_path = dir
        else:
            search_path = os.path.dirname(ROOT_DIR)#os.path.abspath('FunctionTesting.py'))
        regex = r'(.*)' + self.file_regex + r'(.*)'
        r = re.compile(regex)
        for root, dir, files in os.walk(search_path):
            for x in files:
                if r.match(x):
                    result.append(os.path.join(root, x))
        return result

    def import_factors(self, dim, epsilon=0):
        frame = pd.read_csv(self.file_regex, header=0)
        frame.columns = map(str.upper, frame.columns)
        frame = frame.rename(columns={"DIM": "DIMENSION"}, errors="ignore")
        dim_frame = frame.loc[frame['DIMENSION'] == int(dim)]
        fion_name = frame['FUNCTION'].unique()
        dim_array = np.array(dim_frame['FACTORS'])

        if epsilon == 0:
            home = dim_frame['NR_GROUPS'].argmax()
            factors = literal_eval(dim_array[home])
        else:
            epsilon_row = dim_frame.loc[dim_frame['EPSILON'] == epsilon]
            factors = literal_eval(np.array(epsilon_row['FACTORS'])[0])

        return factors, fion_name[0]