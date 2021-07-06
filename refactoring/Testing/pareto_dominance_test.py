from refactoring.optimizationproblems.prescription import Prescription

p = Prescription()
p.jumps = 0.5
p.strat = 0.5
p.fertilizer_rate = 0.5
p.objective_values = [p.jumps, p.strat, p.fertilizer_rate]

p2 = Prescription()
p2.jumps = 0.5
p2.strat = 0.4
p2.fertilizer_rate = 0.5
p2.objective_values = [p2.jumps, p2.strat, p2.fertilizer_rate]