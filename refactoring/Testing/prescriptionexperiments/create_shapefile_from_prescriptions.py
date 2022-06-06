from copy import deepcopy
from operator import itemgetter
import pickle, fiona, os
from fiona.crs import from_epsg
from shapely.geometry import Polygon, shape, mapping, MultiPolygon
import numpy as np

from refactoring.utilities.util import PopulationMember

file_ccea = "/media/amy/WD Drive/Prescriptions/optimal/CCEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_50_0804190813.pickle" 
file_nsga = "/media/amy/WD Drive/Prescriptions/optimal/NSGA_Sec35Mid_strip_trial_3_objectives_ga_runs_100_population_25_1104102923.pickle"
fieldfile = "/home/amy/projects/FEA/refactoring/utilities/saved_fields/sec35mid.pickle"

field = pickle.load(open(fieldfile, 'rb'))
ccea = pickle.load(open(file_ccea, 'rb'))
nsga = pickle.load(open(file_nsga, 'rb'))

#self.jumps, self.fertilizer_rate, self.net_return
net_return_fitnesses = np.array([np.array(x.fitness[-1]) for x in ccea.nondom_archive])
net_return_sol = [x for y, x in sorted(zip(net_return_fitnesses, ccea.nondom_archive))][0]

print([x.fitness for x in ccea.nondom_archive])
print([x.fitness for y, x in sorted(zip(net_return_fitnesses, ccea.nondom_archive))])

fertilizer_fitnesses = np.array([np.array(x.fitness[0]) for x in ccea.nondom_archive])
fertilizer_sol = [x for y, x in sorted(zip(fertilizer_fitnesses, ccea.nondom_archive))][0]


fert_cells = []
nr_cells = []
for i, cell in enumerate(field.cell_list):
    fert_cell = deepcopy(cell)
    fert_cell.nitrogen = fertilizer_sol.variables[i]
    #print(fertilizer_sol.variables[i])
    fert_cells.append(fert_cell)
    nr_cell = deepcopy(cell)
    nr_cell.nitrogen = net_return_sol.variables[i]
    #print(net_return_sol.variables[i])
    nr_cells.append(nr_cell)

def create_prescription_shape_file(filename, cells, field_):
    # schema of the shapefile
    schema = {'geometry': 'Polygon', 'properties': {'ID' : 'int', 'AvgYield':'float:9.6', 'AvgProtein':'float:9.6', 'Nitrogen':'float:9.6'}}
    crs = from_epsg(4326)
    with fiona.open(filename + '.shp','w',driver='ESRI Shapefile', crs=crs,schema= schema) as output:
        prop = {'ID': 0, 'AvgYield': 0, 'AvgProtein': 0, 'Nitrogen': 100}
        #poly = field_.field_shape - MultiPolygon(field_.cell_polys) #field_.field_shape -
        output.write({'geometry': mapping(field_.field_shape), 'properties': prop})
        for cell in cells:
            prop = {'ID': cell.sorted_index, 'AvgYield': cell.yield_, 'AvgProtein': cell.pro_, 'Nitrogen': cell.nitrogen}
            output.write({'geometry': mapping(cell.true_bounds), 'properties': prop})

create_prescription_shape_file("/home/amy/Documents/Work/OFPE/optimal_maps/ccea_net_return", nr_cells, field)
create_prescription_shape_file("/home/amy/Documents/Work/OFPE/optimal_maps/ccea_fertilizer", fert_cells, field)