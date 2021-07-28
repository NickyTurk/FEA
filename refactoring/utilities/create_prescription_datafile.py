import csv
from operator import attrgetter

try:
    import _pickle as pickle
except:
    import pickle
import re
import pandas as pd
import numpy as np


experiment_filenames = ["../../results/FEAMOO/NSGA2_Henrys_trial_3_objectives_ga_runs_200_population_500_1907215351.pickle", "../../results/FEAMOO/NSGA2_Sec35Mid_trial_3_objectives_ga_runs_200_population_500_1907215548.pickle", "../../results/FEAMOO/NSGA2_Sec35West_trial_3_objectives_ga_runs_200_population_500_1907215723.pickle"] #, "../../results/FEAMOO/FEAMOO_Sec35West_trial_3_objectives_linear_topo_ga_runs_100_population_200_1707230035.pickle"]
aggregated_data_files = ["../../../Documents/Work/OFPE/Data/Henrys/wood_henrys_10m_yld_2016-2020_UPDATE.csv", "../../../Documents/Work/OFPE/Data/Sec35Mid/broyles_sec35mid_10m_yld_2016-2020_UPDATE.csv", "../../../Documents/Work/OFPE/Data/Sec35West/broyles_sec35west_10m_yld_2016-2020_UPDATE.csv"]
field_files= ["../utilities/saved_fields/Henrys.pickle", "../utilities/saved_fields/sec35mid.pickle", "../utilities/saved_fields/sec35west.pickle"]

def create_pd_dataframe(prescription, field, headers, dps):
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
    if field.aa_crs != field.latlong_crs:
        xy = np.array([np.array(project_from_latlong.transform(x, y)) for x, y in zip(all_points_df['x'], all_points_df['y'])])
        all_points_df['x'] = xy[:, 0]
        all_points_df['y'] = xy[:, 1]
    return all_points_df

for agg_file, experiment, fieldfile in zip(aggregated_data_files, experiment_filenames, field_files):
    feamoo = pickle.load(open(experiment, 'rb'))
    print(feamoo.iteration_stats[-1]['diversity'])
    field = pickle.load(open(fieldfile, 'rb'))
    field_name = re.search(r'.*\/FEAMOO\/(.*)_trial',experiment)
    objectives = ['jumps', 'strat', 'fertilizer_rate']

    df = pd.read_csv(agg_file)
    # print(field.latlong_crs, field.aa_crs, field.field_crs)
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
    find_center_obj = []
    for obj in objectives:
        print(obj)
        filename_to_write = '../../MOO_prescriptions/' + field_name.group(1) + '_prescription_' + obj + '_objective_runs_' + str(
            feamoo.ga_runs) + '_pop_' + str(feamoo.population_size) + '_UPDATED.csv'
        feamoo.nondom_archive.sort(key=attrgetter(obj))
        prescription = feamoo.nondom_archive[0]
        find_center_obj.append(np.array(prescription.objective_values))
        print(prescription.objective_values)   
        print([x.nitrogen for x in prescription.variables])
        all_points_df = create_pd_dataframe(prescription, field, headers, dps)
        all_points_df.to_csv(filename_to_write)
    length = np.array(find_center_obj).shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    point = np.array([sum_x/length, sum_y/length, sum_z/length])
    objectives = np.array([np.array(sol.objective_values) for sol in nondom_archive])
    dist = np.sum((objectives-point)**2, axis=1)
    idx = np.argmin(dist)
    prescription = nondom_archive[idx]
    print(point, prescription.objective_values)
    filename_to_write = '../../MOO_prescriptions/' + field_name.group(1) + '_prescription_center_objective_runs_' + str(
            feamoo.ga_runs) + '_pop_' + str(feamoo.population_size) + '_UPDATED.csv'
    all_points_df = create_pd_dataframe(prescription, field, headers, dps)
    all_points_df.to_csv(filename_to_write)
