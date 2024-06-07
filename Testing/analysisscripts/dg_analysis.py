import numpy as np

from FEA.factorarchitecture import FactorArchitecture
from utilities.multifilereader import MultiFileReader

"""
Summarize grouping information for the variable groupings as found by MO-DG.
"""

problems = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6"]
nr_objs = [10]
for problem in problems:
    for n_obj in nr_objs:
        file_regex = "DG_" + problem + "_" + str(n_obj) + r"_eps(.*)"
        stored_files = MultiFileReader(
            file_regex=file_regex, dir="D:\\factor_architecture_files\\DG_MOO\\"
        )
        for file in stored_files.path_to_files:
            print(file)
            fa = FactorArchitecture(1000)
            fa.load_architecture(file)
            print("number of factors: ", len(fa.factors))
            print("length of each factor: ", [len(fac) for fac in fa.factors])
            overlap_sizes = []
            for i, fac in enumerate(fa.factors):
                for j in range(i + 1, len(fa.factors)):
                    overlap = np.intersect1d(fac, fa.factors[j])
                    if len(overlap) > 0:
                        overlap_sizes.append(len(overlap))
            print(
                "Number of overlaps: ", len(overlap_sizes), ". Length of overlaps: ", overlap_sizes
            )
            print("average overlap size: ", np.mean(overlap_sizes))
            print("average length of factors: ", np.mean([len(fac) for fac in fa.factors]))
