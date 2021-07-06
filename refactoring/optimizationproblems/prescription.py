import numpy as np


class Prescription:

    def __init__(self):
        self.variables = []
        self.index = -1
        self.overall_fitness = -1
        self.jumps = -1
        self.strat = -1
        self.fertilizer_rate = -1
        self.objective_values = [self.jumps, self.strat, self.fertilizer_rate]
        self.standard_nitrogen = 0
        self.size = len(self.variables)
        self.field = None
        self.total_ylpro_bins = self.field.num_yield_bins * self.field.num_pro_bins
        self.ylpro_string_matrix = self.create_ylpro_string_matrix()
        self.expected_nitrogen_strat, self.max_strat, self.min_strat = self.calc_expected_bin_strat()
        self.max_rate = max(self.field.nitrogen_list) * self.size
        self.max_jumps = ((len(self.field.nitrogen_list) - 1) * (self.size - 1))

    def __eq__(self, other):
        if self.objective_values == other.objective_values:
            return True
        else:
            return False

    def __gt__(self, other):
        if all(x >= y for x,y in zip(self.objective_values, other.objective_values)) \
                and any(x > y for x,y in zip(self.objective_values, other.objective_values)):
            return True
        else:
            return False

    def __lt__(self, other):
        if all(x <= y for x,y in zip(self.objective_values, other.objective_values)) \
                and any(x < y for x,y in zip(self.objective_values, other.objective_values)):
            return True
        else:
            return False

    def set_fitness(self, function, solution):
        self.overall_fitness, self.jumps, self.strat, self.fertilizer_rate \
            = function.problem.calculate_overall_fitness(solution)
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

        return jump_diff / self.max_jumps

    def maximize_stratification(self, solution):
        stratification_diff = 0
        nitrogen_counts = np.zeros((len(self.field.nitrogen_list), self.total_ylpro_bins), dtype=int)

        for i, c in enumerate(solution):
            # Count nitrogen values across yield and protein bin combinations
            ylpro_bin_string = int(str(c.yield_bin) + str(c.pro_bin - 1))
            ylpro_idx = self.ylpro_string_matrix.index(ylpro_bin_string)
            nitrogen_idx = self.field.nitrogen_list.index(c.nitrogen)
            curr_val = nitrogen_counts[nitrogen_idx, ylpro_idx]
            nitrogen_counts[nitrogen_idx, ylpro_idx] = curr_val + 1

        for ylpro_index, yl_pro_counts in enumerate(np.transpose(nitrogen_counts)):
            overall_strat_curr_bin = 0
            for nitrogen_index, count in enumerate(yl_pro_counts):
                curr_strat = abs(self.expected_nitrogen_strat[ylpro_index][0] - count) - \
                             self.expected_nitrogen_strat[ylpro_index][1]
                overall_strat_curr_bin = overall_strat_curr_bin + curr_strat
            stratification_diff = stratification_diff + overall_strat_curr_bin

        return stratification_diff / self.max_strat

    def minimize_overall_fertilizer_rate(self, solution):
        total_fertilizer = sum([c.nitrogen for c in solution])
        return total_fertilizer / self.max_rate

    def create_ylpro_string_matrix(self):
        i = 1
        j = 0
        index = 0
        ylpro_string_matrix = np.zeros(self.total_ylpro_bins, dtype=int)
        while i <= self.field.num_yield_bins:
            while j < self.field.num_pro_bins:
                ylpro_string_matrix[index] = str(i) + str(j)
                index = index + 1
                j = j + 1
            i = i + 1
            j = 0
        return ylpro_string_matrix.tolist()

    def calc_expected_bin_strat(self):
        """
        Calculates expected number of cells for each nitrogen bin.
        Returns expected stratification, maximum and minimum potential stratification counts
        """
        num_nitrogen = len(self.field.nitrogen_list)
        num_cells = len(self.field.cell_list)
        ideal_nitrogen_cells = num_cells / num_nitrogen

        cell_strat = int(ideal_nitrogen_cells / self.total_ylpro_bins)
        min_strat = 0
        expected_bin_strat = []
        for i in range(1, self.field.num_yield_bins + 1):
            for j in range(1, self.field.num_pro_bins + 1):
                cells_in_bin = sum(cell.yield_bin == i and cell.pro_bin == j for cell in self.field.cell_list)
                cell_strat_min = cells_in_bin % num_nitrogen
                spread_of_min_strat_per_bin = cell_strat_min / num_nitrogen
                strat = cells_in_bin / num_nitrogen
                expected_bin_strat.append([strat, spread_of_min_strat_per_bin])
                min_strat += cell_strat_min
        max_strat = 2 * cell_strat * self.total_ylpro_bins * (num_nitrogen - 1)

        return expected_bin_strat, max_strat, min_strat
