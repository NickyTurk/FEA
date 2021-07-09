from refactoring.optimizationproblems.prescription import Prescription
from refactoring.MOO.FEAMOO import FEAMOO
from refactoring.basealgorithms.MOO_GA import GA
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.utilities.field.field_creation import Field

field = Field()
field.field_shape_file = "C:\\Users\\f24n127\\Documents\\raw-farm-data\\Broyles-35west-boundary\\sec35west_bbox.shp"
field.yld_file = "C:\\Users\\f24n127\\Documents\\raw-farm-data\\broyles-35west-2020\\Broyles Farm_Broyles Fami_sec 35 west_Harvest_2020-08-07_00.shp"
field.create_field()

FA = FactorArchitecture(len(field.cell_list))
FA.linear_grouping(width=5, offset=3)
FA.get_factor_topology_elements()

ga = GA

feamoo = FEAMOO(Prescription, 3, 3, 20, FA, ga, dimensions=len(field.cell_list), combinatorial_options=field.nitrogen_list, field=field)
feamoo.run()
