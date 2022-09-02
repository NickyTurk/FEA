"""
variable: Seeding rate
objectives: 
    1. net return/yield
    2. Minimizing weed -> one group of all weeds 
                        -> 60-85 points for 80-230 acre fields
                        -> each point is a quarter meter square
                        -> in each square, volume of the weeds is estimated (m3 per m2)
                        -> strong biomass correlation 
                        -> weeds compete with the crops for resources, so more weeds means more weed-seeds which could 
                        take over a field. Theoretically, planting a lot of crop seeds would reduce the weed influence, 
                        however, this has not necessarily been found to be correct.
        Thistle patch growth -> how does seeding rate influence this growth? High seeding rate seems beneficial for the First year,
        but the second year the thistle comes back stronger. But not enough data yet to include in a model.
    3. Rotation of crops to avoid creating optimal ecosystem for bugs and pests 

Organic vs Conventional: maneure spreading is the most expensive part, and more labor.
conventional farming: pesticide control -> ask bruce and paul about data for this: could influence net return since it costs money
plus how beneficial is it actually?

Hannah's bug research: area vs ecological refuge: based on natural features, if there is low yield in an area, 
should we change this to a biodiversity area, since having such an area positively influences yield right outside of it?
--> how can this be changed into an optimization problem.

!!!! Cell size: min 200m because that is the most accurate size for yield monitoring.
NDWI 
"""

from sklearn.ensemble import RandomForestRegressor
from predictionalgorithms.yieldprediction import YieldPredictor
from optimizationproblems.prescription import Prescription
from FEA.factorarchitecture import FactorArchitecture
from utilities.util import *
from MOO.MOEA import *
from MOO.MOFEA import MOFEA
import pandas as pd
from datetime import timedelta
import pickle, random, re, os, time

# suppressing chain warnings, should fix this
pd.options.mode.chained_assignment = None

fea_runs = 10
ga_runs = [20]
population_sizes = [25]
upper_bound = 225

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

field = pickle.load(open(path + '/utilities/saved_fields/millview.pickle', 'rb'))
agg_file = "C:/Users/amypeerlinck/Documents/work/OFPE/Data/millview/mv20.7.csv"
reduced_agg_file = "C:/Users/amypeerlinck/Documents/work/OFPE/Data/millview/reduced_millview20_spatial.csv"

df = pd.read_csv(agg_file)
y_label = df['yld']
yld_data_to_use = ['x', 'y', 'prev_yld', 'aa_sr', 'elev', 'slope', 'ndvi_py_s', 'ndvi_2py_s', 'ndvi_cy_s', 'ndwi_py_s', 'bm_wd', 'carboncontent10cm', 'phw10cm', 'watercontent10cm', 'sandcontent10cm', 'claycontent10cm']
x_data = df[yld_data_to_use]
rf_yld = RandomForestRegressor()
rf_yld.fit(x_data, y_label)

y_weedslabel = df['bm_wd']
wds_data_to_use = ['x', 'y', 'yld', 'prev_yld', 'aa_sr', 'elev', 'slope', 'ndvi_py_s', 'ndvi_2py_s', 'ndvi_cy_s', 'ndwi_py_s', 'carboncontent10cm', 'phw10cm', 'watercontent10cm', 'sandcontent10cm', 'claycontent10cm']
x_weedsdata = df[wds_data_to_use]
rf_weeds = RandomForestRegressor()
rf_weeds.fit(x_weedsdata, y_weedslabel)

print('Millview -- FEA')
field.fixed_costs = 1000
#random_global_variables = random.choices([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], k=len(field.cell_list))
random_global_variables = [random.randrange(0, upper_bound) for x in range(len(field.cell_list))]
pr = Prescription(variables=random_global_variables, field=field)
yp = YieldPredictor(prescription=pr, field=field, agg_data_file=reduced_agg_file, trained_model=rf_yld, weeds_model=rf_weeds, data_headers=yld_data_to_use, weeds_headers=wds_data_to_use)
FA = FactorArchitecture(len(field.cell_list))
FA.factors = field.create_strip_groups(overlap=True)
FA.get_factor_topology_elements()
nsga = NSGA2


@add_method(NSGA2)
def calc_fitness(variables, gs=None, factor=None):
    pres = Prescription(variables=variables, field=field, factor=factor, organic=True, yield_predictor=yp)
    if gs:
        pres.set_fitness(global_solution=gs.variables, cont_bool=True)
    else:
        pres.set_fitness(cont_bool=True)
    return pres.objective_values


for j in range(5):
    for population in population_sizes:
        for ga_run in ga_runs:
            start = time.time()
            filename = path + '/results/prescriptions/optimized/FEAMOO_millview_trial_organic_objectives_ga_runs_' + str(ga_run) + '_population_' + str(population) + time.strftime('_%d%m%H%M%S') + '.pickle'
            feamoo = MOFEA(fea_runs, ga_run, pop_size=population, factor_architecture=FA, base_alg=nsga, dimensions=len(field.cell_list),
                           value_range=[0,upper_bound], ref_point=[1, 1])  # , combinatorial_options=field.nitrogen_list)
            feamoo.run()
            # nsga = NSGA2(population_size=population, ea_runs=ga_run, dimensions=len(field.cell_list),
            #              upper_value_limit=150, ref_point=[1, 1, 1])
            # nsga.run()
            end = time.time()
            pickle.dump(nsga, open(filename, "wb"))
            elapsed = end-start
            print("NSGA with ga runs %d and population %d took %s"%(ga_run, population, str(timedelta(seconds=elapsed))))