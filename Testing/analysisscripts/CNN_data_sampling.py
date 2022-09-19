from predictionalgorithms.CNN_yieldpredictor.Predictor.YieldMapPredictor import *
from predictionalgorithms.yieldprediction import *
from optimizationproblems.prescription import Prescription
from utilities.util import *
import pandas as pd
import numpy as np
import pickle, random, utm, re

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path_ = path.group()

np.set_printoptions(suppress=True)
field = pickle.load(open(path_ + '/utilities/saved_fields/sec35mid.pickle', 'rb'))

agg_files = ["/home/amy/Documents/Work/OFPE/Data/Sec35Mid/broyles_sec35mid_2016_yl_aggreg_20181112.csv"]
reduced_agg_files = ["/home/amy/Documents/Work/OFPE/Data/Sec35Mid/reduced_broyles_sec35mid_cnn_random.csv",
"/home/amy/Documents/Work/OFPE/Data/Sec35Mid/reduced_broyles_sec35mid_cnn_spatial.csv",
"/home/amy/Documents/Work/OFPE/Data/Sec35Mid/reduced_broyles_sec35mid_cnn_aggregate.csv"]

df = pd.read_csv(agg_files[i])
y_labels = df['yl_2016'] #yl18_bu_ac
data_to_use = ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac', 'n15_lbs_ac', 'n14_lbs_ac']
#HENRYS ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi15', 'ndvi16', 'ndvi17', 'yl16_nn_bu_ac','n16_lbs_ac']
#SEC35MID ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac', 'n15_lbs_ac', 'n14_lbs_ac']
cnn = YieldMapPredictor(filename='/home/amy/Documents/Work/OFPE/Data/broyles_10m_yldDat_with_sentinel.csv', field=field.field_name, pred_year=2020, training_years=[2016, 2018] )
# Load prediction data (it will be saved in cnn.data)
cnn.load_pred_data(objective='yld')
# Load model weights
cnn.model = cnn.init_model(modelType='Hyper3DNet')
path_weights = '/home/amy/projects/OFPETool-master/static/uploads/Model-Hyper3DNet-sec35middle--Objective-yld/Hyper3DNet' + "-" + field.field_name+ "--Objective-yld"
cnn.model.loadModel(path=path_weights)
cnn.patches, centers = cnn.extract2DPatches()

#adjust patches and centers for use in MOO
lat_lon = [utm.to_latlon(cnn.coords[x, y][0], cnn.coords[x, y][1], 12, northern=True) for (x, y) in centers]
cnn.centers = create_indexed_dataframe(dps=pd.DataFrame(lat_lon), field=field)
indeces = [center[2] for center in np.array(cnn.centers)]
points_in_cells_indeces = np.array(indeces).astype(int)
for idx, patch in enumerate(cnn.patches):
    if idx not in points_in_cells_indeces:
        nr = cnn.patches[idx, 0, 0, :]
        nr[nr != 0] = 120#field.base_rate
        cnn.patches[idx, 0, 0, :] = nr

field.fixed_costs = 1000
random_global_variables = [random.randrange(0, 150) for x in range(len(field.cell_list))]
pr = Prescription(variables=random_global_variables, field=field)
yp = YieldPredictor(prescription=pr, field=field, agg_data_file=reduced_agg_files[i], trained_model=cnn, data_headers=data_to_use, cnn_bool=True)

yp.calculate_yield(prescription=pr, cnn=True)