import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, MultiPolygon
from shapely.ops import split
import pickle, re, os, random
import numpy as np

'''
Cell ID for 2D patches and points -> enables direct mapping to reduce calculation cost
For extract 2D patches as well: uses double for loop with a kind of step size to determine how many points to skip, step could be increased
'''


def reduce_dataframe(field, data_to_use, agg_file, transform_to_latlon=False, transform_from_latlon=False, spatial_sampling=False):
    df = pd.read_csv(agg_file)
    project_from_latlong = Transformer.from_crs(field.latlong_crs, 'epsg:32612')
    project_to_latlong = Transformer.from_crs('EPSG:6657', field.latlong_crs)# 'epsg:32612')
    x_int = 0
    y_int = 1
    dps = df[data_to_use]
    print(len(dps))
    dps.dropna(axis=0, inplace=True)
    print(len(dps))
    new_df = pd.DataFrame()

    if transform_to_latlon:
        xy = np.array(
            [np.array(project_to_latlong.transform(x, y)) for x, y in zip(np.array(dps['x']), np.array(dps['y']))])
        dps.loc[:, 'x'] = xy[:, 0]
        dps.loc[:, 'y'] = xy[:, 1]
    for i, gridcell in enumerate(field.cell_list):
        if spatial_sampling:
            minx, miny, maxx, maxy = gridcell.true_bounds.bounds
            dx = (maxx - minx) / 2  # width of a small part
            dy = (maxy - miny) / 5  # height of a small part
            horizontal_splitters = [LineString([(minx, miny + i * dy), (maxx, miny + i * dy)]) for i in range(2)]
            vertical_splitters = [LineString([(minx + i * dx, miny), (minx + i * dx, maxy)]) for i in range(5)]
            splitters = horizontal_splitters + vertical_splitters
            result = gridcell.true_bounds
            for splitter in splitters:
                result = MultiPolygon(split(result, splitter))
            for part in result.geoms:
                bl_x, bl_y, ur_x, ur_y = part.bounds
                points_in_cell = dps[(dps['y'] >= bl_x) &
                                     (dps['y'] <= ur_x) &
                                     (dps['x'] <= ur_y) &
                                     (dps['x'] >= bl_y)].values.tolist()
                if len(points_in_cell) == 1:
                    cell_df = pd.DataFrame(random.sample(points_in_cell, 1))
                    new_df = pd.concat([cell_df, new_df])
                elif len(points_in_cell) > 1:
                    cell_df = pd.DataFrame(random.sample(points_in_cell, 2))
                    new_df = pd.concat([cell_df, new_df])
        else:
            bl_x, bl_y = gridcell.bottomleft_x, gridcell.bottomleft_y
            ur_x, ur_y = gridcell.upperright_x, gridcell.upperright_y
            points_in_cell = dps[(dps['y'] >= bl_x) &
                                 (dps['y'] <= ur_x) &
                                 (dps['x'] <= ur_y) &
                                 (dps['x'] >= bl_y)].values.tolist()
            if len(points_in_cell) > 0:
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
    current_working_dir = os.getcwd()
    path = re.search(r'^(.*?\\FEA)', current_working_dir)
    path = path.group()

    data_to_use = ['lon', 'lat', 'yld', 'prev_yld', 'aa_sr', 'prev_aa_sr', 'elev', 'slope', 'ndvi_py_s', 'ndvi_2py_s', 'ndvi_cy_s', 'ndwi_py_s', 'bm_wd', 'carboncontent10cm', 'phw10cm', 'watercontent10cm', 'sandcontent10cm', 'claycontent10cm']

    agg_file = "C:/Users/f24n127/OneDrive - Montana State University/Documents/raw-farm-data/millview/mv20.5.csv"
    field = pickle.load(open(path+'/utilities/saved_fields/millview.pickle', 'rb'))
    reduced = reduce_dataframe(field, data_to_use, agg_file, transform_to_latlon=False, spatial_sampling=True)
    reduced.to_csv("C:/Users/f24n127/OneDrive - Montana State University/Documents/raw-farm-data/millview/reduced_millview20_spatial.csv")