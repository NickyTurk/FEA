from copy import deepcopy

from sklearn.ensemble import RandomForestRegressor
from predictionalgorithms.yieldprediction import YieldPredictor
from optimizationproblems.prescription import Prescription
from FEA.factorarchitecture import FactorArchitecture
from utilities import fileIO
from utilities.util import *
from MOO.MOEA import *
from MOO.MOFEA import MOFEA
import pandas as pd
from datetime import timedelta
import pickle, random, re, os, time

import warnings

warnings.filterwarnings("ignore")


"""
Parameters
"""
fea_runs = 10
ga_run = 20
population_size = 50
upper_bound = 150

overlap_bool = False

if overlap_bool:
    alg_name = "FNSGA2"
else:
    alg_name = "CCNSGA2"

field_names = ["sec35mid"]  #'Henrys',
current_working_dir = os.getcwd()
path = re.search(r"^(.*?[\\/]FEA)", current_working_dir)
path = path.group()

"""
Set up farming field information
"""
field_1 = pickle.load(
    open(path + "/utilities/saved_fields/Henrys.pickle", "rb")
)  # /home/alinck/FEA
field_2 = pickle.load(open(path + "/utilities/saved_fields/sec35mid.pickle", "rb"))
fields_to_test = [field_1, field_2]

agg_files = [
    "C:/Users/f24n127/Documents/Work/Ag/Data/aggregated_data/wood_henrys_yl18_aggreg_20181203.csv",
    "C:/Users/f24n127/Documents/Work/Ag/Data/aggregated_data/broyles_sec35mid_2016_yl_aggreg_20181112.csv",
]
reduced_agg_files = [
    "C:/Users/f24n127/Documents/Work/Ag/Data/aggregated_data/reduced_henrys_agg.csv",
    "C:/Users/f24n127/Documents/Work/Ag/Data/aggregated_data/reduced_broyles_sec35mid_agg.csv",
]

labels = ["yl18_bu_ac", "yl_2016"]
data_to_use = [
    [
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
    ],  # Henrys
    [
        "x",
        "y",
        "n_lbs_ac",
        "elev_m",
        "slope_deg",
        "ndvi_2012",
        "ndvi_2014",
        "ndvi_2015",
        "yl14_nn_bu_ac",  # Sec35Mid
        "n15_lbs_ac",
        "n14_lbs_ac",
    ],
]

for i, field in enumerate(fields_to_test):
    df = pd.read_csv(agg_files[i])
    y_labels = df[labels[i]]
    x_data = df[data_to_use[i]]

    # Train model
    rf = RandomForestRegressor()
    rf.fit(x_data, y_labels)

    # Add extra field information, done here because it is directly related to the Field object
    field.fixed_costs = 1000
    field.fertilizer_list_1 = [0, 150]
    field.max_fertilizer_rate = max(field.fertilizer_list_1) * len(field.cell_list)
    field.n_dict = {st: idx for idx, st in enumerate(field.fertilizer_list_1)}
    field.total_ylpro_bins = field.num_pro_bins * field.num_yield_bins
    random_global_variables = random.choices(
        [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], k=len(field.cell_list)
    )

    # Set up prescription and yieldpredictor objects based on field information to calculate fitness for prescriptions
    pr = Prescription(variables=random_global_variables, field=field, optimized=True)
    yp = YieldPredictor(
        prescription=pr,
        field=field,
        agg_data_file=agg_files[i],
        trained_model=rf,
        data_headers=data_to_use,
    )
    pr.yield_predictor = yp
    pr.set_fitness(cont_bool=True)

    # Get factor architecture based on field size (i.e. number of plots on the field is the number of variables)
    FA = FactorArchitecture(len(field.cell_list))
    FA.factors = field.create_strip_groups(
        overlap=overlap_bool
    )  # group sizes are based on strips on the field
    FA.get_factor_topology_elements()

    # Set up base-algorithm, NSGA2 has the best results for this problem.
    # "partial" is used to send through the algorithm specific parameters.
    nsga = partial(NSGA2, population_size=population_size, ea_runs=ga_run)

    """
    Add fitness evaluation to the base algorithm.
    """

    @add_method(NSGA2)
    def calc_fitness(variables, gs=None, factor=None):
        pres = Prescription(
            variables=variables, field=field, factor=factor, optimized=True, yield_predictor=yp
        )
        if gs:
            pres.set_fitness(global_solution=gs.variables, cont_bool=True)
        else:
            pres.set_fitness(cont_bool=True)
        return pres.objective_values

    """
    Get optimized map from algorithm
    """
    # File to save to
    filename = (
        path
        + "/results/prescriptions/CNN_optimized/"
        + alg_name
        + "_"
        + field_names[i]
        + "_strip_trial_3_objectives_ga_runs_"
        + str(ga_run)
        + "_population_"
        + str(population_size)
        + time.strftime("_%d%m%H%M%S")
        + ".pickle"
    )

    # Set up algorithm and run it
    feamoo = MOFEA(
        fea_iterations=fea_runs,
        factor_architecture=FA,
        base_alg=nsga,
        dimensions=len(field.cell_list),
        value_range=[0, upper_bound],
        ref_point=[1, 1, 1],
    )  # , combinatorial_options=field.nitrogen_list)
    feamoo.run()

    """
    Create final prescriptions with experimental plots
    """
    # Get prescription maps with highest net return and lowest fertilizer
    net_return_fitnesses = np.array([np.array(x.fitness[-1]) for x in feamoo.nondom_archive])
    net_return_sol = [x for y, x in sorted(zip(net_return_fitnesses, feamoo.nondom_archive))][0]

    fertilizer_fitnesses = np.array([np.array(x.fitness[0]) for x in feamoo.nondom_archive])
    fertilizer_sol = [x for y, x in sorted(zip(fertilizer_fitnesses, feamoo.nondom_archive))][0]

    # create Prescription objects
    prescription_NR_max = Prescription(variables=net_return_sol, field=field)
    prescription_fert_min = Prescription(variables=fertilizer_sol, field=field)

    # Overlay experimental plots
    prescription_NR_max.overlay_experimental_plots(experiment_percentage=0.25, method="stdev")
    prescription_fert_min.overlay_experimental_plots(experiment_percentage=0.25, method="stdev")

    """
    Save prescriptions to file
    """
    fert_cells = []
    nr_cells = []
    # create list of nitrogen values to write to file
    for i, cell in enumerate(field.cell_list):
        fert_cell = deepcopy(
            cell
        )  # needs to be a deepcopy since we are assigning different values to this cell object
        cell.nitrogen = prescription_fert_min.complete_variables[i].nitrogen
        fert_cells.append(fert_cell)
        nr_cell = deepcopy(cell)
        nr_cell.nitrogen = prescription_NR_max.complete_variables[i].nitrogen
        nr_cells.append(nr_cell)

    # Shapefile schema
    schema = {
        "ID": "int",
        "AvgYield": "float:9.6",
        "AvgProtein": "float:9.6",
        "Nitrogen": "float:9.6",
    }

    fileIO.ShapeFiles.write_field_to_shape_file(
        nr_cells,
        field,
        shapefile_schema=schema,
        filename=path + "/results/optimal_maps_with_exp_plots/FNSGA_net_return",
    )
    fileIO.ShapeFiles.write_field_to_shape_file(
        fert_cells,
        field,
        shapefile_schema=schema,
        filename=path + "/results/optimal_maps_with_exp_plots/FNSGA_fertilizer",
    )
