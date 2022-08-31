import pandas as pd
import pickle, random, re, os, time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

field = pickle.load(open(path + '/utilities/saved_fields/millview.pickle', 'rb'))
agg_file = "C:/Users/amypeerlinck/Documents/work/OFPE/Data/millview/mv20.7.csv"

k_fold = KFold(10)

df = pd.read_csv(agg_file)
y_label = df['yld'].to_numpy()
yld_data_to_use = ['x', 'y', 'prev_yld', 'aa_sr', 'elev', 'slope', 'ndvi_py_s', 'ndvi_2py_s', 'ndvi_cy_s', 'ndwi_py_s', 'bm_wd', 'carboncontent10cm', 'phw10cm', 'watercontent10cm', 'sandcontent10cm', 'claycontent10cm']
x_data = df[yld_data_to_use].to_numpy()
rf_yld = RandomForestRegressor()

y_weedslabel = df['bm_wd'].to_numpy()
wds_data_to_use = ['x', 'y', 'yld', 'prev_yld', 'aa_sr', 'elev', 'slope', 'ndvi_py_s', 'ndvi_2py_s', 'ndvi_cy_s', 'ndwi_py_s', 'carboncontent10cm', 'phw10cm', 'watercontent10cm', 'sandcontent10cm', 'claycontent10cm']
x_weedsdata = df[wds_data_to_use].to_numpy()
rf_weeds = RandomForestRegressor()

for k, (train, test) in enumerate(k_fold.split(x_data, y_label)):
    rf_yld.fit(x_data[train], y_label[train])
    rf_weeds.fit(x_weedsdata[train], y_weedslabel[train])

    print(
        "[fold {0}] Yld score: {1:.5f}, yld rmse: {2:.5f}, weeds score: {3:.5f}, weeds rmse: {4:.5f}".format(
            k, rf_yld.score(x_data[test], y_label[test]), mean_squared_error(rf_yld.predict(x_data[test]), y_label[test]),
            rf_weeds.score(x_weedsdata[test], y_weedslabel[test]), mean_squared_error(rf_weeds.predict(x_weedsdata[test]), y_weedslabel[test])
        )
    )
