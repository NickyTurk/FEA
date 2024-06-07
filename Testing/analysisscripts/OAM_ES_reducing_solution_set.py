import pickle

import numpy as np
from hvwfg import wfg
from matplotlib import pyplot as plt
from pymoo.core.result import Result
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

from MOO.MOEA import MOEA
from MOO.archivemanagement import *
from MOO.paretofrontevaluation import ParetoOptimization
from utilities.multifilereader import MultiFileReader

problems = ["DTLZ5", "DTLZ6", "WFG3", "WFG7"]  #  'DTLZ6', 'WFG3', 'WFG7'
ks = [0.4, 0.5, 0.25, 0.5]
ls = [0.3, 0.4, 0.4, 0.2]
#
# ks = [.4,.5,.5,.5]
# ls = [.5,.2,.4,.4]
comparing = []
for k, l in zip(ks, ls):
    comparing.append(["NSGA2", "k_" + str(k).replace(".", "") + "_l_" + str(l).replace(".", "")])
obj = 5
archive_overlap = 3
po = ParetoOptimization(obj_size=obj)

for i, problem in enumerate(problems):
    # comparing.append(['NSGA2', 'k_' + str(ks[i]).replace('.', '') + '_l_' + str(ls[i]).replace('.', '')])
    print("******************\n", problem, str(obj), comparing[i], "\n***********************\n")
    reference_point = pickle.load(
        open("E:\\" + problem + "_" + str(obj) + "_reference_point.pickle", "rb")
    )
    file_regex = r"NSGA2(.*)" + problem + r"_(.*)" + str(obj) + r"_objectives_"
    stored_files = MultiFileReader(
        file_regex,
        dir="C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\full_solution\\",
    )
    experiment_filenames = stored_files.path_to_files
    experiments = [
        x
        for x in experiment_filenames
        if comparing[i][0] in x and comparing[i][1] in x and problem in x
    ]
    avg_hv = {"nsga2": [], "EFAM": [], "ESE": [], "SFAM": [], "ESS": []}
    avg_div = {"nsga2": [], "EFAM": [], "ESE": [], "SFAM": [], "ESS": []}
    for experiment in experiments:
        try:
            results = pickle.load(open(experiment, "rb"))
        except EOFError:
            print("issues with file: ", experiment)
            continue
        if isinstance(results, MOEA):
            overlapping_archive = results.nondom_archive.find_archive_overlap(
                nr_archives_overlapping=archive_overlap
            )
            sfam = ObjectiveArchive(obj, 100, percent_best=ks[i], percent_diversity=ls[i])
            sfam.update_archive(results.nondom_pop)
            sfam_overlap = sfam.find_archive_overlap(nr_archives_overlapping=archive_overlap)
            ese_sol_set = environmental_solution_selection_nsga2(
                results.nondom_archive.flatten_archive(), sol_size=len(overlapping_archive)
            )
            ess_sol_set = environmental_solution_selection_nsga2(
                results.nondom_pop, sol_size=len(sfam_overlap)
            )
        # elif isinstance(results, FactorArchive):
        #     overlapping_archive = results.find_archive_overlap(nr_archives_overlapping=archive_overlap)
        else:
            continue

        nsga2_fitness = np.array(
            [np.array(sol.fitness) / reference_point for sol in results.nondom_pop]
        )
        nsga2_hv = wfg(nsga2_fitness, np.ones(obj))
        nsga2_div = po.calculate_diversity(nsga2_fitness, normalized=True)

        if len(sfam_overlap) > 2:
            sfam_fitness = np.array(
                [np.array(sol.fitness) / reference_point for sol in sfam_overlap]
            )
            sfam_hv = wfg(sfam_fitness, np.ones(obj))
            sfam_div = po.calculate_diversity(sfam_fitness, normalized=True)

            ess_fitness = np.array([np.array(sol.fitness) / reference_point for sol in ess_sol_set])
            ess_hv = wfg(ess_fitness, np.ones(obj))
            ess_div = po.calculate_diversity(ess_fitness, normalized=True)
        else:
            sfam_div = 0
            sfam_hv = 0

            ess_hv = 0
            ess_div = 0

        if len(overlapping_archive) > 2:
            overlapping_fitness = np.array(
                [np.array(sol.fitness) / reference_point for sol in overlapping_archive]
            )
            efam_hv = wfg(overlapping_fitness, np.ones(obj))
            efam_div = po.calculate_diversity(overlapping_fitness, normalized=True)
            ese_fitness = np.array([np.array(sol.fitness) / reference_point for sol in ese_sol_set])
            ese_hv = wfg(ese_fitness, np.ones(obj))
            ese_div = po.calculate_diversity(ese_fitness, normalized=True)
        else:
            efam_div = 0
            efam_hv = 0

            ese_hv = 0
            ese_div = 0

        avg_hv["nsga2"].append(nsga2_hv)
        avg_hv["EFAM"].append(efam_hv)
        avg_hv["ESE"].append(ese_hv)
        avg_hv["SFAM"].append(sfam_hv)
        avg_hv["ESS"].append(ess_hv)
        avg_div["nsga2"].append(nsga2_div)
        avg_div["EFAM"].append(efam_div)
        avg_div["ESE"].append(ese_div)
        avg_div["SFAM"].append(sfam_div)
        avg_div["ESS"].append(ess_div)
    print(
        "HV:\n",
        np.mean(avg_hv["nsga2"]),
        "\n",
        np.mean(avg_hv["EFAM"]),
        "\n",
        np.mean(avg_hv["ESE"]),
        "\n",
        np.mean(avg_hv["SFAM"]),
        "\n",
        np.mean(avg_hv["ESS"]),
    )
    print(
        "spread:\n",
        np.mean(avg_div["nsga2"]),
        "\n",
        np.mean(avg_div["EFAM"]),
        "\n",
        np.mean(avg_div["ESE"]),
        "\n",
        np.mean(avg_div["SFAM"]),
        "\n",
        np.mean(avg_div["ESS"]),
    )
