from predictionalgorithms.CNN_yieldpredictor.Predictor.YieldMapPredictor import *
from predictionalgorithms.yieldprediction import *
from optimizationproblems.prescription import Prescription
from FEA.factorarchitecture import FactorArchitecture
from utilities.util import *
from MOO.MOEA import *
from MOO.MOFEA import MOFEA
import pandas as pd
import numpy as np
from datetime import timedelta
import pickle, random, re, os, time

np.set_printoptions(suppress=True)

"""
PARAMETERS
"""
fea_runs = 10
ga_run = 50
population_size = 100
upper_bound = 150 # max lbs of nitrogen / acre to apply
field_names = ['henrys'] # 'sec35middle', 'Sec35West'] #'Henrys'

# using single population (i.e. NSGA2) or subpopulations?
single_pop = False

# If single pop is False, using FEA (overlap=True) or CCEA?
overlap_bool = False
if overlap_bool:
    alg_name = 'FNSGA2'
else:
    alg_name = 'CCNSGA2'

# get path to current directory
current_working_dir = os.getcwd()
path_ = re.search(r'^(.*?[\\/]FEA)',current_working_dir)
path = path_.group()

# field_3 = pickle.load(open(path + '/utilities/saved_fields/sec35mid.pickle', 'rb')) # /home/alinck/FEA
#field_2 = pickle.load(open(path_ + '/utilities/saved_fields/sec35mid.pickle', 'rb'))
field_3 = pickle.load(open(path + '/utilities/saved_fields/Henrys.pickle', 'rb'))
fields_to_test = [field_3] #[field_1, field_2, field_3]

for i, field in enumerate(fields_to_test):
    field.field_name = field_names[i]
    agg_files = ["C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\henrys\\wood_10m_yldDat_with_sentinel.csv"] #henrys\\wood_10m_yldDat_with_sentinel.csv"] #broyles_10m_yldDat_with_sentinel.csv"]
    reduced_agg_files = ["C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\reduced_broyles_sec35mid_cnn_aggregate.csv"] # henrys\\reduced_wood_10m_yldDat_with_sentinel_aggregate.csv"] #reduced_broyles_10m_yldDat_with_sentinel_aggregate.csv"]
    df = pd.read_csv(agg_files[i])

    """
    Load CNN model
    """
    cnn = YieldMapPredictor(filename="C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\henrys\\wood_10m_yldDat_with_sentinel.csv", field=field.field_name, pred_year=2020, training_years=[2016, 2018] )
    # Load prediction data (it will be saved in cnn.data)
    cnn.load_pred_data(objective='yld')
    # Load model weights
    cnn.model = cnn.init_model(modelType='Hyper3DNet')
    path_weights = 'C:\\Users\\f24n127\\Documents\\Work\\OFPETool-master\\static\\uploads\\Hyper3DNet-'+field_names[i]+'--Objective-yld\\Hyper3DNet-'+field_names[i]+'--Objective-yld'
    cnn.model.loadModel(path=path_weights)
    patches, centers = cnn.extract2DPatches()

    """
    Adjust patches and centers for use in MOO
    """
    lat_lon = [utm.to_latlon(cnn.coords[x, y][0], cnn.coords[x, y][1], 12, northern=True) for (x, y) in centers] #utm.to_latlon(cnn.coords[x, y][0], cnn.coords[x, y][1], 12, northern=True
    # add cell ids to centers dataframe
    centers = create_indexed_dataframe(dps=pd.DataFrame(lat_lon), field=field) # epsg_string='epsg:2263'

    # creates list of indeces of cnn patches that are in the field cells
    indeces = [center[2] for center in np.array(centers)]
    points_in_cells_indeces = np.array(indeces).astype(int)

    # Reduce number of patches to predict on through random selection
    cell_ids = set(centers['cell_index'])
    new_centers = pd.DataFrame(columns=centers.columns)
    for cell_id in cell_ids:
        cell_rows = centers.loc[centers['cell_index'] == cell_id]
        singlerow = cell_rows.sample()
        new_centers = new_centers.append(singlerow)
    cnn.centers = new_centers

    new_patches = []
    for idx, center in cnn.centers.iterrows():
        new_patches.append(patches[center['point_index']])
        # if idx not in points_in_cells_indeces:
        #     nitrogen_rate = patches[idx, 0, 0, :]
        #     nitrogen_rate[nitrogen_rate != 0] = 120
        #     patches[idx, 0, 0, :] = nitrogen_rate

    cnn.patches = np.array(new_patches)
    field.fixed_costs = 1000
    field.fertilizer_list_1 = [0,150]
    field.max_fertilizer_rate = max(field.fertilizer_list_1) * len(field.cell_list)
    field.n_dict = {st: idx for idx, st in enumerate(field.fertilizer_list_1)}
    field.total_ylpro_bins = field.num_pro_bins * field.num_yield_bins

    """
    Initialize global solution prescription and yieldpredictor object
    """
    random_global_variables = [random.randrange(0, upper_bound) for x in range(len(field.cell_list))] # random.randrange(0, upper_bound)
    pr = Prescription(variables=random_global_variables, field=field, optimized=True)
    yp = YieldPredictor(prescription=pr, field=field, agg_data_file=reduced_agg_files[i], trained_model=cnn, cnn_bool=True)

    """
    Load architecture for subpopulations
    """
    if not single_pop:
        FA = FactorArchitecture(len(field.cell_list))
        FA.factors = field.create_strip_groups(overlap=overlap_bool)
        FA.get_factor_topology_elements()
        nsga = partial(NSGA2, population_size=population_size, ea_runs=ga_run)

    """
    This creates the appropriate fitness function.
    @add_method is a decorator function that allows you to overwrite the fitness function.
    """
    @add_method(NSGA2)
    def calc_fitness(variables, gs=None, factor=None):
        pres = Prescription(variables=variables, field=field, factor=factor, optimized=True, yield_predictor=yp)
        if gs:
            #global_solution = Prescription(variables=gs.variables, field=field)
            pres.set_fitness(global_solution=gs.variables, cont_bool=True)
        else:
            pres.set_fitness(cont_bool=True)
        return pres.objective_values

    """
    Start actual experiments
    """
    for j in range(5):
        start = time.time()
        filename = path + '/results/prescriptions/CNN_optimized/'+alg_name+'_' + field_names[i] + '_strip_trial_3_objectives_ga_runs_' + str(ga_run) + '_population_' + str(population_size) + time.strftime('_%d%m%H%M%S') + '.pickle'
        if not single_pop:
            algorithm = MOFEA(fea_iterations=fea_runs, factor_architecture=FA, base_alg=nsga, dimensions=len(field.cell_list),
                           value_range=[0,upper_bound], ref_point=[1, 1, 1]) #, combinatorial_options=field.nitrogen_list)
            algorithm.run()
        else:
            algorithm = NSGA2(population_size=population_size, ea_runs=ga_run, dimensions=len(field.cell_list),
                         value_range=[0, upper_bound], reference_point=[1, 1, 1])
            algorithm.run()
        end = time.time()
        pickle.dump(algorithm, open(filename, "wb"))
        elapsed = end-start
        print("CCNSGA with ga runs %d and population %d took %s"%(ga_run, population_size, str(timedelta(seconds=elapsed))))



