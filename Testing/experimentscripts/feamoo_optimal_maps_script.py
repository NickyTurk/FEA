from sklearn.ensemble import RandomForestRegressor
from predictionalgorithms.yieldprediction import YieldPredictor
from optimizationproblems.prescription import Prescription
from FEA.factorarchitecture import FactorArchitecture
from utilities.util import *
from MOO.MOEA import *
from MOO.MOFEA import MOFEA
import pandas as pd
from datetime import timedelta
import pickle, random, re, os, time

import warnings

warnings.filterwarnings("ignore")

fea_runs = 10
ga_run = 50
population_size = 100
upper_bound = 150

overlap_bool = True

if overlap_bool:
    alg_name = "FNSGA2"
else:
    alg_name = "CCNSGA2"

field_names = ["henrys"]  #'Henrys',
current_working_dir = os.getcwd()
path = re.search(r"^(.*?[\\/]FEA)", current_working_dir)
path = path.group()
print("path: ", path)

field_1 = pickle.load(
    open(path + "/utilities/saved_fields/Henrys.pickle", "rb")
)  # /home/alinck/FEA
# field_2 = pickle.load(open(path + '/utilities/saved_fields/sec35mid.pickle', 'rb'))
# field_3 = pickle.load(open(path + '/utilities/saved_fields/sec35west.pickle', 'rb'))
fields_to_test = [field_1]  # [field_1, field_2, field_3]
agg_files = [
    "C:/Users/f24n127/Documents/Work/Ag/Data/aggregated_data/wood_henrys_yl18_aggreg_20181203.csv"
]  # wood_henrys_yl18_aggreg_20181203.csv broyles_sec35mid_2016_yl_aggreg_20181112.csv
reduced_agg_files = [
    "C:/Users/f24n127/Documents/Work/Ag/Data/reduced_wood_henrys_aggregate.csv"
]  # reduced_wood_henrys_aggregate.csv  reduced_broyles_sec35mid_aggregate

for i, field in enumerate(fields_to_test):
    df = pd.read_csv(agg_files[i])
    y_labels = df["yl18_bu_ac"]  # yl18_bu_ac #yl_2016
    data_to_use = [
        "x",
        "y",
        "n_lbs_ac",
        "elev_m",
        "slope_deg",
        "ndvi15",
        "ndvi16",
        "ndvi17",
        "yl16_nn_bu_ac",
        "n16_lbs_ac",
    ]
    # Henrys ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi15', 'ndvi16', 'ndvi17', 'yl16_nn_bu_ac','n16_lbs_ac']
    # Sec35mid ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac', 'n15_lbs_ac', 'n14_lbs_ac']
    x_data = df[data_to_use]
    rf = RandomForestRegressor()
    rf.fit(x_data, y_labels)

    print(field_names[i], alg_name)
    field.fixed_costs = 1000
    field.fertilizer_list_1 = [0, 150]
    field.max_fertilizer_rate = max(field.fertilizer_list_1) * len(field.cell_list)
    field.n_dict = {st: idx for idx, st in enumerate(field.fertilizer_list_1)}
    field.total_ylpro_bins = field.num_pro_bins * field.num_yield_bins
    random_global_variables = random.choices(
        [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], k=len(field.cell_list)
    )
    # random_global_variables = [rate for x in range(len(field.cell_list))] #random.randrange(0, upper_bound)
    pr = Prescription(variables=random_global_variables, field=field, optimized=True)
    yp = YieldPredictor(
        prescription=pr,
        field=field,
        agg_data_file=reduced_agg_files[i],
        trained_model=rf,
        data_headers=data_to_use,
    )
    pr.yield_predictor = yp
    pr.set_fitness(cont_bool=True)
    print(pr.objective_values)
    FA = FactorArchitecture(len(field.cell_list))
    print("cell list ", len(field.cell_list))
    FA.factors = field.create_strip_groups(overlap=overlap_bool)
    print("factors ", FA.factors)
    FA.get_factor_topology_elements()
    # nsga = partial(NSGA2, population_size=population_size, ea_runs=ga_run)

    @add_method(NSGA2)
    def calc_fitness(variables, gs=None, factor=None):
        pres = Prescription(
            variables=variables, field=field, factor=factor, optimized=True, yield_predictor=yp
        )
        if gs:
            # global_solution = Prescription(variables=gs.variables, field=field)
            pres.set_fitness(global_solution=gs.variables, cont_bool=True)
        else:
            pres.set_fitness(cont_bool=True)
        return pres.objective_values

    for j in range(5):
        start = time.time()
        filename = (
            path
            + "/results/prescriptions/RF_optimized/SNSGA2_"
            + field_names[i]
            + "_strip_trial_3_objectives_ga_runs_"
            + str(ga_run)
            + "_population_"
            + str(population_size)
            + time.strftime("_%d%m%H%M%S")
            + ".pickle"
        )
        # feamoo = MOFEA(fea_iterations= fea_runs, factor_architecture= FA, base_alg= nsga, dimensions=len(field.cell_list),
        #                value_range=[0,upper_bound], ref_point=[1, 1, 1]) #, combinatorial_options=field.nitrogen_list)
        # feamoo.run()
        nsga = NSGA2(
            population_size=population_size,
            ea_runs=ga_run,
            dimensions=len(field.cell_list),
            value_range=[0, upper_bound],
            reference_point=[1, 1, 1],
        )
        nsga.run()
        end = time.time()
        pickle.dump(nsga, open(filename, "wb"))
        elapsed = end - start
        print(
            "NSGA with ga runs %d and population %d took %s"
            % (ga_run, population_size, str(timedelta(seconds=elapsed)))
        )
