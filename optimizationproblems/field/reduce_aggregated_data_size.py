import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, MultiPolygon
from shapely.ops import split
import pickle, re, os, random
import numpy as np

"""
Function to reduce the data set for faster prediction calculations.
The predictive models are still TRAINED on the ENTIRE data set, 
this is only used for the optimization process to reduce the number of points for which we are getting yield predictions.
I.e. instead of having to predict yield for 20,000 points, we reduce it to 100-1,000 points (size depends on method used).

When using the CNN:
Cell ID for 2D patches and points -> enables direct mapping to reduce calculation cost
For extract 2D patches as well: uses double for loop with a kind of step size to determine how many points to skip, step could be increased
"""


def reduce_dataframe(
    field,
    agg_file,
    data_to_use=None,
    transform_to_latlon=False,
    transform_from_latlon=False,
    epsg_string="",
    sampling_type="random",
):
    """
    @param sampling_type: which type of sampling to apply: 'random', 'spatial', 'aggregate'
    """
    if isinstance(agg_file, pd.DataFrame):
        df = agg_file
    else:
        df = pd.read_csv(agg_file)
    project_from_latlong = Transformer.from_crs(field.latlong_crs, epsg_string)
    project_to_latlong = Transformer.from_crs(
        epsg_string, field.latlong_crs
    )  # 'epsg:32612') CANADA: 'EPSG:6657'
    x_int = 0
    y_int = 1
    if data_to_use is None:
        dps = df
    else:
        dps = df[data_to_use]
    try:
        dps = dps.dropna(axis=0, subset=["yld", "aa_n"])
    except KeyError:
        pass
    dps = dps.dropna(axis=1, how="all")
    dps = dps.select_dtypes(include=["number"])
    new_df = pd.DataFrame()
    if transform_to_latlon:
        xy = np.array(
            [
                np.array(project_to_latlong.transform(x, y))
                for x, y in zip(np.array(dps["x"]), np.array(dps["y"]))
            ]
        )
        dps.loc[:, "x"] = xy[:, 0]
        dps.loc[:, "y"] = xy[:, 1]
    for i, gridcell in enumerate(field.cell_list):
        if sampling_type == "spatial":
            minx, miny, maxx, maxy = gridcell.true_bounds.bounds
            dx = (maxx - minx) / 2  # width of a small part
            dy = (maxy - miny) / 5  # height of a small part
            horizontal_splitters = [
                LineString([(minx, miny + i * dy), (maxx, miny + i * dy)]) for i in range(2)
            ]
            vertical_splitters = [
                LineString([(minx + i * dx, miny), (minx + i * dx, maxy)]) for i in range(5)
            ]
            splitters = horizontal_splitters + vertical_splitters
            result = gridcell.true_bounds
            for splitter in splitters:
                result = MultiPolygon(split(result, splitter))
            for part in result.geoms:
                bl_x, bl_y, ur_x, ur_y = part.bounds
                points_in_cell = dps[
                    (dps["y"] >= bl_x)
                    & (dps["y"] <= ur_x)
                    & (dps["x"] <= ur_y)
                    & (dps["x"] >= bl_y)
                ].values.tolist()
                if len(points_in_cell) == 1:
                    cell_df = pd.DataFrame(random.sample(points_in_cell, 1))
                    cell_df.loc[:, "cell_index"] = i
                    new_df = pd.concat([cell_df, new_df])
                elif len(points_in_cell) > 1:
                    cell_df = pd.DataFrame(random.sample(points_in_cell, 2))
                    cell_df.loc[:, "cell_index"] = i
                    new_df = pd.concat([cell_df, new_df])
        elif sampling_type == "aggregate":
            bl_x, bl_y = gridcell.bottomleft_x, gridcell.bottomleft_y
            ur_x, ur_y = gridcell.upperright_x, gridcell.upperright_y
            points_in_cell = dps[
                (dps["y"] >= bl_x) & (dps["y"] <= ur_x) & (dps["x"] <= ur_y) & (dps["x"] >= bl_y)
            ].values.tolist()
            if len(points_in_cell) > 0:
                points_in_cell = pd.DataFrame(points_in_cell)
                aggregate_point = points_in_cell.mean(axis=0).to_numpy()
                cell_df = pd.DataFrame([aggregate_point])
                cell_df.loc[:, "cell_index"] = i
                new_df = pd.concat([cell_df, new_df])

        else:
            bl_x, bl_y = gridcell.bottomleft_x, gridcell.bottomleft_y
            ur_x, ur_y = gridcell.upperright_x, gridcell.upperright_y
            points_in_cell = dps[
                (dps["y"] >= bl_x) & (dps["y"] <= ur_x) & (dps["x"] <= ur_y) & (dps["x"] >= bl_y)
            ].values.tolist()
            if len(points_in_cell) > 0:
                if len(points_in_cell) > 10:
                    n_keep = 10
                else:
                    n_keep = len(points_in_cell)
                to_keep = random.sample(points_in_cell, n_keep)
                cell_df = pd.DataFrame(to_keep)
                cell_df.loc[:, "cell_index"] = i
                new_df = pd.concat([cell_df, new_df])
    if transform_from_latlon:
        xy = np.array(
            [
                np.array(project_from_latlong.transform(x, y))
                for x, y in zip(new_df[x_int], new_df[y_int])
            ]
        )
        new_df.loc[:, x_int] = xy[:, 0]
        new_df.loc[:, y_int] = xy[:, 1]
    if data_to_use is not None:
        new_df.columns = data_to_use
    else:
        columns = dps.columns.to_list()
        columns.append("cell_index")
        new_df.columns = columns
    return new_df


if __name__ == "__main__":
    current_working_dir = os.getcwd()
    path_ = re.search(r"^(.*?[\\/]FEA)", current_working_dir)
    path_ = path_.group()

    agg_file = "C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\aggregated_data\\wood_henrys_yl18_aggreg_20181203.csv"

    df = pd.read_csv(agg_file)
    df20 = df.loc[(df["field"] == "henrys")]  # (df['year']==2020) &
    # df21 = df.loc[(df['year']==2021) & (df['field']=='henrys')]

    # df_20_21 = pd.merge(df20, df21, on=('x', 'y', 'lon', 'lat', 'elev', 'slope'), sort=False, suffixes=('_2020', '_2021'))

    field = pickle.load(open(path_ + "/utilities/saved_fields/Henrys.pickle", "rb"))
    reduced = reduce_dataframe(
        field, df, transform_to_latlon=True, sampling_type="aggregate", epsg_string="epsg:32612"
    )  # Canada: 'EPSG:6657'
    reduced.to_csv(
        "C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\reduced_wood_henrys_aggregate.csv"
    )

    # reduced = reduce_dataframe(field, agg_file, transform_to_latlon=True, spatial_sampling=False)
    # reduced.to_csv("/home/amy/Documents/Work/OFPE/Data/Sec35Mid/reduced_broyles_sec35mid_cnn_random.csv")
