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

fea_runs = 10
ga_run = 20
population_size = 50
upper_bound = 150

field_names = ['henrys'] # 'sec35middle', 'Sec35West'] #'Henrys',
current_working_dir = os.getcwd()
path_ = re.search(r'^(.*?[\\/]FEA)',current_working_dir)
path = path_.group()

overlap_bool = False

if overlap_bool:
    alg_name = 'FNSGA2'
else:
    alg_name = 'CCNSGA2'

# field_1 = pickle.load(open(path + '/utilities/saved_fields/sec35mid.pickle', 'rb')) # /home/alinck/FEA
#field_2 = pickle.load(open(path_ + '/utilities/saved_fields/sec35mid.pickle', 'rb'))
field_3 = pickle.load(open(path + '/utilities/saved_fields/Henrys.pickle', 'rb'))
fields_to_test = [field_3] #[field_1, field_2, field_3]

for i, field in enumerate(fields_to_test):
    field.field_name = field_names[i]
    agg_files = ["C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\henrys\\wood_10m_yldDat_with_sentinel.csv"] #broyles_10m_yldDat_with_sentinel.csv"]
    reduced_agg_files = ["C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\henrys\\reduced_wood_10m_yldDat_with_sentinel_aggregate.csv"] #reduced_broyles_10m_yldDat_with_sentinel_aggregate.csv"]
    df = pd.read_csv(agg_files[i])
    cnn = YieldMapPredictor(filename="C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\henrys\\wood_10m_yldDat_with_sentinel.csv", field=field.field_name, pred_year=2020, training_years=[2016, 2018] )
    # Load prediction data (it will be saved in cnn.data)
    cnn.load_pred_data(objective='yld')
    # Load model weights
    cnn.model = cnn.init_model(modelType='Hyper3DNet')
    path_weights = 'C:\\Users\\f24n127\\Documents\\Work\\OFPETool-master\\static\\uploads\\Hyper3DNet-'+field_names[i]+'--Objective-yld\\Hyper3DNet-'+field_names[i]+'--Objective-yld'
    cnn.model.loadModel(path=path_weights)
    cnn.patches, centers = cnn.extract2DPatches()

    #adjust patches and centers for use in MOO
    lat_lon = [utm.to_latlon(cnn.coords[x, y][0], cnn.coords[x, y][1], 12, northern=True) for (x, y) in centers] #utm.to_latlon(cnn.coords[x, y][0], cnn.coords[x, y][1], 12, northern=True
    # add cell ids to centers dataframe
    cnn.centers = create_indexed_dataframe(dps=pd.DataFrame(lat_lon), field=field) # epsg_string='epsg:2263'
    indeces = [center[2] for center in np.array(cnn.centers)]
    points_in_cells_indeces = np.array(indeces).astype(int)
    for idx, patch in enumerate(cnn.patches):
        if idx not in points_in_cells_indeces:
            nr = cnn.patches[idx, 0, 0, :]
            nr[nr != 0] = 120#field.base_rate
            cnn.patches[idx, 0, 0, :] = nr

    field.fixed_costs = 1000
    field.fertilizer_list_1 = [0,150]
    field.max_fertilizer_rate = max(field.fertilizer_list_1) * len(field.cell_list)
    field.n_dict = {st: idx for idx, st in enumerate(field.fertilizer_list_1)}
    field.total_ylpro_bins = field.num_pro_bins * field.num_yield_bins
    # rates = [0, 20, 40, 60, 80, 100]
    # for rate in rates:
    #     print(rate)
    random_global_variables = [random.randrange(0, upper_bound) for x in range(len(field.cell_list))] # random.randrange(0, upper_bound)
    pr = Prescription(variables=random_global_variables, field=field, optimized=True)
    yp = YieldPredictor(prescription=pr, field=field, agg_data_file=agg_files[i], trained_model=cnn, cnn_bool=True)
    #     pr.yield_predictor = yp
    #     pr.set_fitness()
    #     print(pr.objective_values)
    # yp.calculate_yield(cnn=True)
    FA = FactorArchitecture(len(field.cell_list))
    print(len(field.cell_list))
    FA.factors = field.create_strip_groups(overlap=overlap_bool)
    print(FA.factors)
    FA.get_factor_topology_elements()
    nsga = partial(NSGA2, population_size=population_size, ea_runs=ga_run)

    @add_method(NSGA2)
    def calc_fitness(variables, gs=None, factor=None):
        pres = Prescription(variables=variables, field=field, factor=factor, optimized=True, yield_predictor=yp)
        if gs:
            #global_solution = Prescription(variables=gs.variables, field=field)
            pres.set_fitness(global_solution=gs.variables, cont_bool=True)
        else:
            pres.set_fitness(cont_bool=True)
        return pres.objective_values

    for j in range(5):
        start = time.time()
        filename = path + '/results/prescriptions/CNN_optimized/'+ alg_name + '_' + field_names[i] + '_strip_trial_3_objectives_ga_runs_' + str(ga_run) + '_population_' + str(population_size) + time.strftime('_%d%m%H%M%S') + '.pickle'
        feamoo = MOFEA(fea_iterations=fea_runs, factor_architecture=FA, base_alg=nsga, dimensions=len(field.cell_list),
                       value_range=[0,upper_bound], ref_point=[1, 1, 1]) #, combinatorial_options=field.nitrogen_list)
        feamoo.run()
        # nsga = NSGA2(population_size=population, ea_runs=ga_run, dimensions=len(field.cell_list),
        #              value_range=[0, upper_bound], reference_point=[1, 1, 1])
        # nsga.run()
        end = time.time()
        pickle.dump(feamoo, open(filename, "wb"))
        elapsed = end-start
        print("CCNSGA with ga runs %d and population %d took %s"%(ga_run, population_size, str(timedelta(seconds=elapsed))))



