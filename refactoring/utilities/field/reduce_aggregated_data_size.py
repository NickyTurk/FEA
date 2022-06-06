import pandas as pd
from pyproj import Transformer
import pickle, re, os, random
import numpy as np


def reduce_dataframe(field, data_to_use, agg_file, transform_to_latlon=False, transform_from_latlon=False):
    df = pd.read_csv(agg_file)
    points_df = pd.DataFrame()
    project_from_latlong = Transformer.from_crs(field.latlong_crs, 'epsg:32612')
    project_to_latlong = Transformer.from_crs( 'epsg:32612', field.latlong_crs)# 'epsg:32612')
    x_int = 0
    y_int = 1
    dps = df[data_to_use]
    new_df = pd.DataFrame()

    if transform_to_latlon:
        xy = np.array(
            [np.array(project_to_latlong.transform(x, y)) for x, y in zip(np.array(dps['x']), np.array(dps['y']))])
        dps['x'] = xy[:, 0]
        dps['y'] = xy[:, 1]
    for i, gridcell in enumerate(field.cell_list):
        bl_x, bl_y = gridcell.bottomleft_x, gridcell.bottomleft_y
        ur_x, ur_y = gridcell.upperright_x, gridcell.upperright_y
        # Get all points in the cell
        points_in_cell = dps[(dps['y'] >= bl_x) &
                             (dps['y'] <= ur_x) &
                             (dps['x'] <= ur_y) &
                             (dps['x'] >= bl_y)].values.tolist()
        if len(points_in_cell) > 0:
            print(len(points_in_cell))
            if len(points_in_cell) > 10:
                n_keep = 10
            else:
                n_keep = len(points_in_cell)
            to_keep = random.sample(points_in_cell, n_keep)
            cell_df = pd.DataFrame(to_keep)
            new_df = pd.concat([cell_df, new_df])
    if transform_from_latlon:
        xy = np.array(
            [np.array(project_from_latlong.transform(x, y)) for x, y in
             zip(new_df[x_int], new_df[y_int])])
        new_df.loc[:, x_int] = xy[:, 0]
        new_df.loc[:, y_int] = xy[:, 1]
    new_df.columns = data_to_use
    return new_df


if __name__ == '__main__':
    #current_working_dir = os.getcwd()
    #path = re.search(r'^(.*?\\FEA)', current_working_dir)
    #path = path.group()

    data_to_use = ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi15', 'ndvi16', 'ndvi17', 'yl16_nn_bu_ac',
     'n16_lbs_ac', 'yl18_bu_ac']
    agg_file = "/home/amy/Documents/Work/OFPE/Data/Henrys/wood_henrys_yl18_aggreg_20181203.csv"
    field = pickle.load(open('/home/amy/projects/FEA/refactoring/utilities/saved_fields/Henrys.pickle', 'rb'))
    reduced = reduce_dataframe(field, data_to_use, agg_file, transform_to_latlon=True)
    reduced.to_csv("/home/amy/Documents/Work/OFPE/Data/Henrys/reduced_henrys_agg.csv")