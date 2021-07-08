import numpy as np
from ..utilities.field.field_creation import Field


class Prescription:

    def __init__(self, variables=None, field=None, index=-1):
        if variables is not None:
            self.variables = variables
        elif field is not None:
            self.variables = field.assign_nitrogen_distribution()
        else:
            self.variables = []
        self.index = index
        self.overall_fitness = -1
        self.jumps = -1
        self.strat = -1
        self.fertilizer_rate = -1
        self.objective_values = [self.jumps, self.strat, self.fertilizer_rate]
        self.standard_nitrogen = 0
        self.size = len(self.variables)
        self.field = field
        self.objectives = [self.maximize_stratification, self.minimize_jumps,
                           self.minimize_overall_fertilizer_rate]
        self.n_obj = len(self.objectives)
        self.set_fitness()

    def __eq__(self, other):
        if self.objective_values == other.objective_values:
            return True
        else:
            return False

    def __gt__(self, other):
        if all(x >= y for x, y in zip(self.objective_values, other.objective_values)) \
                and any(x > y for x,y in zip(self.objective_values, other.objective_values)):
            return True
        else:
            return False

    # def __gt__(self, other):
    #     if all(x > y for x, y in zip(self.objective_values, other.objective_values)):
    #         return True
    #     else:
    #         return False

    def __lt__(self, other):
        if all(x <= y for x, y in zip(self.objective_values, other.objective_values)) \
                and any(x < y for x, y in zip(self.objective_values, other.objective_values)):
            return True
        else:
            return False

    # def __lt__(self, other):
    #     if all(x < y for x, y in zip(self.objective_values, other.objective_values)):
    #         return True
    #     else:
    #         return False

    def run(self, x, i):
        if i < self.n_obj:
            return self.objectives[i](x)
        elif i == self.n_obj:
            f = []
            for j in range(0, self.n_obj):
                f.append(self.objectives[j](x))
            return f

    def set_fitness(self, solution=None, global_solution=None, factor=None):
        if solution is not None:
            self.variables = solution
        if global_solution is not None:
            full_solution = [x for x in global_solution]
            for i, x in zip(factor, self.variables):
                full_solution[i] = x
        else:
            full_solution = self.variables
        self.overall_fitness, self.jumps, self.strat, self.fertilizer_rate \
            = self.calculate_overall_fitness(full_solution)
        self.objective_values = [self.jumps, self.strat, self.fertilizer_rate]

    def set_field(self, field):
        self.field = field
        self.field.nitrogen_list.sort()

    def calculate_overall_fitness(self, solution):
        jumps = self.minimize_jumps(solution)
        strat = self.maximize_stratification(solution)
        rate = self.minimize_overall_fertilizer_rate(solution)
        return (jumps + strat + rate) / 3, jumps, strat, rate

    def minimize_jumps(self, solution):
        jump_diff = 0

        for i, c in enumerate(solution):
            # calculate jump between nitrogen values for consecutive cells
            if i + 1 != len(solution):
                index_1 = self.field.nitrogen_list.index(c.nitrogen)
                index_2 = self.field.nitrogen_list.index(solution[i + 1].nitrogen)
                temp_jump = abs(index_1 - index_2)
                if temp_jump > 1:
                    jump_diff = jump_diff + temp_jump

        return jump_diff / self.field.max_jumps

    def maximize_stratification(self, solution):
        stratification_diff = 0
        nitrogen_counts = np.zeros((len(self.field.nitrogen_list), self.field.total_ylpro_bins), dtype=int)

        for i, c in enumerate(solution):
            # Count nitrogen values across yield and protein bin combinations
            ylpro_bin_string = int(str(c.yield_bin) + str(c.pro_bin - 1))
            ylpro_idx = self.field.ylpro_string_matrix.index(ylpro_bin_string)
            nitrogen_idx = self.field.nitrogen_list.index(c.nitrogen)
            curr_val = nitrogen_counts[nitrogen_idx, ylpro_idx]
            nitrogen_counts[nitrogen_idx, ylpro_idx] = curr_val + 1

        for ylpro_index, yl_pro_counts in enumerate(np.transpose(nitrogen_counts)):
            overall_strat_curr_bin = 0
            for nitrogen_index, count in enumerate(yl_pro_counts):
                curr_strat = abs(self.field.expected_nitrogen_strat[ylpro_index][0] - count) - \
                             self.field.expected_nitrogen_strat[ylpro_index][1]
                overall_strat_curr_bin = overall_strat_curr_bin + curr_strat
            stratification_diff = stratification_diff + overall_strat_curr_bin

        return stratification_diff / self.field.max_strat

    def minimize_overall_fertilizer_rate(self, solution):
        total_fertilizer = sum([c.nitrogen for c in solution])
        return total_fertilizer / self.field.max_fertilizer_rate
