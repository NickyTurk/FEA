import csv
from operator import attrgetter

try:
    import _pickle as pickle
except:
    import pickle
import re
import pandas as pd
import numpy as np

experiment_filenames = ["../../results/FEAMOO/FEAMOO_Henrys_trial_3_objectives_linear_topo_10_5_ga_runs_50_population_100_1507091719.pickle"]
aggregated_data_files = ["../../../Documents/Work/OFPE/Data/Henrys/wood_henrys_10m_yld_2016-2020_UPDATE.csv"]
field_files= ["../utilities/saved_fields/Henrys.pickle"]
for agg_file, experiment in zip(aggregated_data_files, experiment_filenames):
    feamoo = pickle.load(open(experiment, 'rb'))
    field = pickle.load(open("../utilities/saved_fields/Henrys.pickle", 'rb'))
    field_name = re.search(r'[^FEAMOO_]+(?=_)',experiment)
    objectives = ['jumps', 'strat', 'fertilizer_rate']

    df = pd.read_csv(agg_file)
    print(df)
    if field.aa_crs != field.latlong_crs:
        from pyproj import Transformer
        project_to_latlong = Transformer.from_crs(field.aa_crs, field.latlong_crs)
        xy = np.array([np.array(project_to_latlong.transform(x, y)) for x, y in zip(df['x'], df['y'])])
        df['x'] = xy[:, 0]
        df['y'] = xy[:, 1]
    headers = {c: i for i, c in enumerate(df.columns)}
    header_index = {i:c for i,c in enumerate(df.columns)}
    header_index[len(df.columns)] = 'N'
    dps = df.to_numpy()
    project_from_latlong = Transformer.from_crs(field.latlong_crs, field.aa_crs)
    for obj in objectives:
        filename_to_write = '../../MOO_prescriptions/' + field_name.group(0) + '_prescription_' + obj + '_objective_runs_' + str(
            feamoo.base_alg_iterations) + '_pop_' + str(feamoo.pop_size) + '_UPDATED.csv'
        prescription = min(feamoo.nondom_archive, key=attrgetter(obj))
        prescribed_n_datapoints = []
        # [print(dp) for dp in feamoo.field.yield_points.datapoints]
        all_points_df = pd.DataFrame()
        for i, gridcell in enumerate(prescription.variables):
            x_int = headers['x']
            y_int = headers['y']
            bl_x, bl_y = gridcell.bottomleft_x, gridcell.bottomleft_y
            ur_x, ur_y = gridcell.upperright_x, gridcell.upperright_y
            points_in_cell = dps[(dps[:,y_int] >= bl_x) &
                                             (dps[:,y_int] <= ur_x) &
                                             (dps[:,x_int] <= ur_y) &
                                             (dps[:,x_int] >= bl_y)]
            cell_df = pd.DataFrame(points_in_cell)
            cell_df['N'] = gridcell.nitrogen
            if len(points_in_cell) > 0:
                all_points_df = pd.concat([cell_df, all_points_df])
        all_points_df = all_points_df.rename(columns=header_index)
        all_points_df.to_csv(filename_to_write)





