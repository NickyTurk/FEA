from refactoring.optimizationproblems.prescription import Prescription
from refactoring.MOO.MOFEA import MOFEA
from refactoring.MOO.MOEA import NSGA2
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.utilities.field.field_creation import Field

#TODO: Change into unit test

field = Field()
field.field_shape_file = "C:\\Users\\f24n127\\Documents\\raw-farm-data\\Broyles-35west-boundary\\sec35west_bbox.shp"
field.yld_file = "C:\\Users\\f24n127\\Documents\\raw-farm-data\\broyles-35west-2020\\Broyles Farm_Broyles Fami_sec 35 west_Harvest_2020-08-07_00.shp"
field.create_field()

FA = FactorArchitecture(len(field.cell_list))
FA.linear_grouping(width=5, offset=3)

ga = NSGA2

p = Prescription(field=field)
p.jumps = 0.8
p.strat = 0.3
p.fertilizer_rate = 0.5
p.objective_values = [p.jumps, p.strat, p.fertilizer_rate]

p2 = Prescription(field=field)
p2.jumps = 0.5
p2.strat = 0.6
p2.fertilizer_rate = 0.5
p2.objective_values = [p2.jumps, p2.strat, p2.fertilizer_rate]

q = Prescription(field=field)
q.jumps = 0.3
q.strat = 0.5
q.fertilizer_rate = 0.4
q.objective_values = [q.jumps, q.strat, q.fertilizer_rate]

r = Prescription(field=field)
r.jumps = 0.2
r.strat = 0.3
r.fertilizer_rate = 0.2
r.objective_values = [r.jumps, r.strat, r.fertilizer_rate]

pop = [p, p2, q, r]

feamoo = MOFEA(Prescription, 3, 3, 100, FA, ga, 3, combinatorial_options=field.nitrogen_list, field=field)

sols = feamoo.pf.evaluate_pareto_dominance(pop)
[print(s.objective_values) for s in sols]
ga2 = NSGA2()
sorted = ga2.diversity_sort(pop)
pop.sort()

