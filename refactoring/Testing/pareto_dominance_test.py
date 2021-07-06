from refactoring.optimizationproblems.prescription import Prescription
from refactoring.MOO.FEAMOO import FEAMOO
from refactoring.utilities.field.field_creation import Field

field = Field()
field.field_shape_file = "C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\rosie\\rosie_bbox.shp"
field.yld_file = "C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\rosie\\19_rosies_yld.shp"
field.create_field()

p = Prescription(field=field)
p.jumps = 0.5
p.strat = 0.5
p.fertilizer_rate = 0.5
p.objective_values = [p.jumps, p.strat, p.fertilizer_rate]

p2 = Prescription(field=field)
p2.jumps = 0.5
p2.strat = 0.4
p2.fertilizer_rate = 0.5
p2.objective_values = [p2.jumps, p2.strat, p2.fertilizer_rate]

q = Prescription(field=field)
q.jumps = 0.3
q.strat = 0.5
q.fertilizer_rate = 0.4
q.objective_values = [q.jumps, q.strat, q.fertilizer_rate]

r = Prescription(field=field)
r.jumps = 0.6
r.strat = 0.3
r.fertilizer_rate = 0.4
r.objective_values = [r.jumps, r.strat, r.fertilizer_rate]

pop = [p, p2, q, r]

sols = FEAMOO.evaluate_pareto_dominance(pop)

[print(sol.objective_values) for sol in sols]