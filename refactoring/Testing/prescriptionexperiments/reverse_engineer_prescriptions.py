import pickle, glob
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, shape, Point
import pyproj
from shapely.ops import transform

field_files= ["../../utilities/saved_fields/sec35west.pickle","../../utilities/saved_fields/sec35mid.pickle", "../../utilities/saved_fields/Henrys.pickle"]
field_names = ['sec35west', 'sec35mid', 'henrys']
methods = ["CCEAMOO", "NSGA2", "FEAMOO"]
objectives = ["center", "fertilizer_rate", "jumps", "strat"]

#all_data = dict() #{'henrys': { 'cceamoo':{'jumps': 0, 'strat': 0, 'fertilizer_rate': 0, 'center': 0}, 'nsga2': {} } }
for fieldfile, field_name in zip(field_files, field_names):
    print(field_name)
    field = pickle.load(open(fieldfile, 'rb'))
    print(len(field.cell_list))
    #all_data[field_name] = dict()
    """
    transform_plots = False
    if field.field_crs.lower() != 'epsg:32612':
        project = pyproj.Transformer.from_crs(pyproj.CRS(field.field_crs), pyproj.CRS('epsg:32612'),
                                              always_xy=True).transform
        transform_plots = True
    for method in methods:
        print(method)
        #all_data[field_name][method] = dict()
        for obj in objectives:
            print(obj)
            to_find = 'I:/Prescriptions/' + method + '/' + method + "_" + field_name + '_prescription_' + obj + '*.csv'
            results = glob.glob(to_find)[0]
            if not results:
                to_find = 'I:/Prescriptions/' + method + '/' + field_name + "_" + method + '_prescription_' + obj + '*.csv'
                results = glob.glob(to_find)[0]

            results_df = pd.read_csv(results)
            applied_nitrogen = 0
            for i, plot in enumerate(field.cell_list):
                if transform_plots:
                    plot_bounds = transform(project, plot.true_bounds)
                else:
                    plot_bounds = plot.true_bounds
                #print('plot: ', i, plot_bounds)
                for idx, row in results_df.iterrows():
                    if plot_bounds.contains(Point(row['x'], row['y'])):
                        #print('added nitrogen: ', row['N'])
                        applied_nitrogen += row['N']
                        break
            print(applied_nitrogen)
            #all_data[field_name][method][obj] = applied_nitrogen

#print(all_data)
    """
