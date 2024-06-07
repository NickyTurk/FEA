import random

# -o uid=$(id -u), gid=$(id -g)

import numpy as np
from copy import deepcopy


class Prescription:
    """
    Prescription object to create experimental or optimal fertilizer/seeding rate prescription maps.
    Experimental: optimizes stratification, jump minimization, and fertilizer minimization
    Optimal: optimizes net return,  jump minimization, and fertilizer minimization
    organic maps: optimizes net return and looks at weed minimization: needs work
    """

    def __init__(
        self,
        variables=None,
        field=None,
        factor=None,
        index=-1,
        normalize_objectives=False,
        optimized=False,
        organic=False,
        yield_predictor=None,
        applicator_cost=1,
        yield_price=5.40,
    ):
        """
        self.variables should be a list of GRIDCELL objects
            -> the size of the factor group if using FEA
            -> the size of the entire field otherwise
        :param variables: prescribed fertilizer rates for the cells
        :param field: farmfield object (Field)
        :param factor: factor group (based on FactorArchitectur, list of variable indeces)
        :param index: prescription index
        :param normalize_objectives: Are we normalizing yes or no
        :param optimized: Are we creating an optimal (=True) or experimental (=False) prescription
        :param organic: Are we creating an organic prescription (i.e. seeding rate)?
        :param yield_predictor: YieldPredictor object, uses CNN or RF
        :param applicator_cost: how expensive is the applicator ($/lb)
        :param yield_price: How much can the product be sold for?
        """

        # variables are defined, factor group and farmfield are not defined
        if variables and factor is None and field is None:
            self.variables = variables
        # variables, factor group and farmfield are all defined
        elif variables and factor and field:
            # assign correct cells based on factor group
            field_cells = [field.cell_list[i] for i in factor]
            # assign prescribed fertilizer to the cells
            for i, c in enumerate(field_cells):
                c.nitrogen = variables[i]
            self.variables = [cell for cell in field_cells]
        # variables and farmfield are defined, no factor group
        elif variables and factor is None and field:
            field_cells = [f for f in field.cell_list]
            for i, c in enumerate(field_cells):
                c.nitrogen = variables[i]
            self.variables = [cell for cell in field_cells]
        # farm field is defined, variables and factor group are not
        elif field and variables is None and factor is None:
            self.variables = field.assign_nitrogen_distribution()
        # factor group and farmfield are defined, variables are not
        elif factor and field and variables is None:
            field_cells = field.assign_nitrogen_distribution()
            self.variables = [field_cells[i] for i in factor]
        else:
            self.variables = []
        self.complete_variables = []
        self.index = index
        self.overall_fitness = -1
        self.jumps = -1
        self.strat = -1
        self.fertilizer_rate = -1
        self.net_return = -1
        self.yld = -1
        self.weeds_rate = -1
        self.standard_nitrogen = 0
        self.size = len(self.variables)
        self.field = field
        self.optimized = optimized
        self.organic = organic
        self.normalize = normalize_objectives
        self.yield_predictor = yield_predictor
        self.applicator_cost = (
            applicator_cost  # cost in dollars for fertilizer based on application measure
        )
        self.yield_price = (
            yield_price  # dollars made per unit, e.g. bushels per acre of winter wheat
        )
        # if not self.optimized:
        #     self.objective_values = (self.strat, self.jumps, self.fertilizer_rate)
        #     self.objectives = [self.maximize_stratification, self.minimize_jumps,
        #                        self.minimize_overall_fertilizer_rate]
        # else:
        self.objective_values = (self.jumps, self.fertilizer_rate, self.net_return)
        self.objectives = [
            self.minimize_jumps,
            self.minimize_overall_fertilizer_rate,
            self.optimize_yld,
        ]
        self.n_obj = len(self.objectives)
        self.factor = factor
        self.ref_point = [1, 1, 1]
        self.gridcell_size = self.field.cell_list[0].gridcell_size / 43560

    def __eq__(self, other):
        self_vars = [x.nitrogen for x in self.variables]
        other_vars = [x.nitrogen for x in other.variables]
        if all(x == y for x, y in zip(self_vars, other_vars)):
            return True
        else:
            return False

    def __hash__(self):
        string = str([str(x.nitrogen) for x in self.variables])
        return hash(string)

    def __gt__(self, other):
        if all(x >= y for x, y in zip(self.objective_values, other.objective_values)) and any(
            x > y for x, y in zip(self.objective_values, other.objective_values)
        ):
            return True
        else:
            return False

    def __le__(self, other):
        if all(x <= y for x, y in zip(self.objective_values, other.objective_values)) and any(
            x < y for x, y in zip(self.objective_values, other.objective_values)
        ):
            return True
        else:
            return False

    def __lt__(self, other):
        if any(x < y for x, y in zip(self.objective_values, other.objective_values)):
            return True
        else:
            return False

    def run(self, x, i):
        if i < self.n_obj:
            return self.objectives[i](x)
        elif i == self.n_obj:
            f = []
            for j in range(0, self.n_obj):
                f.append(self.objectives[j](x))
            return f

    def set_fitness(self, solution=None, global_solution=None, cont_bool=True):
        """
        depending on which map you are creating, select correct objective functions
        If part of FEA, create full solution using the global solution before sending to objective calculations.
        """
        if solution:
            self.variables = solution
        if global_solution:
            self.complete_variables = [x for x in self.field.cell_list]
            for i, x in enumerate(global_solution):
                self.complete_variables[i].nitrogen = x
            for i, x in zip(self.factor, self.variables):
                self.complete_variables[i].nitrogen = x.nitrogen
        else:
            self.complete_variables = self.variables
        if self.optimized:
            (
                self.overall_fitness,
                self.jumps,
                self.fertilizer_rate,
                self.net_return,
            ) = self.calculate_optimal_fitness(self.complete_variables, cont_bool)
            self.objective_values = (self.jumps, self.fertilizer_rate, self.net_return)
        elif self.organic:
            self.overall_fitness, self.weeds_rate, self.net_return = self.calculate_organic_fitness(
                self.complete_variables
            )
            self.objective_values = (self.weeds_rate, self.net_return)
        else:
            (
                self.overall_fitness,
                self.jumps,
                self.strat,
                self.fertilizer_rate,
            ) = self.calculate_experimental_fitness(self.complete_variables)
            self.objective_values = (self.jumps, self.fertilizer_rate, self.strat)

    def set_field(self, field):
        """
        Setter function for field object
        """
        self.field = field
        self.field.fertilizer_list_1.sort()

    def calculate_organic_fitness(self, solution):
        """
        organic maps objective functions
        """
        self.yield_predictor.adjust_nitrogen_data(solution, cnn=self.yield_predictor.cnn_bool)
        net_return = self.optimize_yld(solution)
        weeds = self.minimize_weeds()
        return (weeds + net_return) / 2, weeds, net_return

    def calculate_optimal_fitness(self, solution, cont_bool):
        """
        traditional maps objective function
        """
        jumps = self.minimize_jumps(solution, continuous=cont_bool)
        rate = self.minimize_overall_fertilizer_rate(solution)
        self.yield_predictor.adjust_nitrogen_data(solution, cnn=self.yield_predictor.cnn_bool)
        net_return = self.optimize_yld(solution)
        return (jumps + rate + net_return) / 3, jumps, rate, net_return

    def calculate_experimental_fitness(self, solution):
        """
        experimental maps objective functions
        """
        jumps = self.minimize_jumps(solution)
        strat = self.maximize_stratification(solution)
        rate = self.minimize_overall_fertilizer_rate(solution)
        return (jumps + strat + rate) / 3, jumps, strat, rate

    def minimize_jumps(self, solution, continuous=True):
        """
        minimize rate jumps between consecutive cells in grid
        Experimental maps: combinatorial problem, so we use indeces of fertilizer rates to calculate jump difference
        Optimal maps: continuous problem, so we simply sum the difference in jumps
        """
        jump_diff = 0
        for i, c in enumerate(solution):
            # calculate jump between nitrogen values for consecutive cells
            if i + 1 != len(solution):
                if not continuous:
                    index_1 = self.field.n_dict[c.nitrogen]
                    index_2 = self.field.n_dict[solution[i + 1].nitrogen]
                    temp_jump = abs(index_1 - index_2)
                    if temp_jump > 1:
                        jump_diff = jump_diff + temp_jump
                else:
                    jump_diff += abs(c.nitrogen - solution[i + 1].nitrogen)
        if continuous:
            final_jumps = jump_diff / (
                max(self.field.fertilizer_list_1) * (len(self.field.cell_list) - 1)
            )
        else:
            final_jumps = jump_diff / (
                (len(self.field.fertilizer_list_1) - 1) * (len(self.field.cell_list) - 1)
            )
        return final_jumps

    def maximize_stratification(self, solution):
        """
        for experimental maps only
        tries to optimize even spread of fertilizer rates across cells with yield and protein bins
        """
        stratification_diff = 0
        nitrogen_counts = np.zeros(
            (len(self.field.fertilizer_list_1), self.field.total_ylpro_bins), dtype=int
        )

        for i, c in enumerate(solution):
            # Count nitrogen values across yield and protein bin combinations
            ylpro_bin_string = int(str(c.yield_bin) + str(c.pro_bin - 1))
            ylpro_idx = self.field.ylpro_dict[ylpro_bin_string]
            nitrogen_idx = self.field.n_dict[c.nitrogen]
            curr_val = nitrogen_counts[nitrogen_idx, ylpro_idx]
            nitrogen_counts[nitrogen_idx, ylpro_idx] = curr_val + 1

        for ylpro_index, yl_pro_counts in enumerate(np.transpose(nitrogen_counts)):
            overall_strat_curr_bin = 0
            for nitrogen_index, count in enumerate(yl_pro_counts):
                curr_strat = (
                    abs(self.field.expected_nitrogen_strat[ylpro_index][0] - count)
                    - self.field.expected_nitrogen_strat[ylpro_index][1]
                )
                overall_strat_curr_bin = overall_strat_curr_bin + curr_strat
            stratification_diff = stratification_diff + overall_strat_curr_bin

        return stratification_diff / self.field.max_strat

    def minimize_overall_fertilizer_rate(self, solution):
        """
        sum fertilizer applied to get total fertilizer applied
         normalize by dividing over max potential fertilizer applied
        """
        total_fertilizer = sum([c.nitrogen for c in solution])
        return total_fertilizer / self.field.max_fertilizer_rate

    def optimize_yld(self, solution):
        """
        maximize net return
        predict yield using trained model (e.g. CNN or Random Forest), which is wrapped in the "YieldPredictor" class
        use predicted yield to calculate net return
        """
        predicted_yield = self.yield_predictor.calculate_yield(cnn=self.yield_predictor.cnn_bool)
        # P = base_price + ()
        applicator = sum([c.nitrogen * self.gridcell_size for c in solution])
        net_return = (
            predicted_yield * self.yield_price
            - (applicator * self.applicator_cost)
            - self.field.fixed_costs
        )
        return -net_return

    def overlay_experimental_plots(self, experiment_percentage=0.2, method="stdev"):
        """
        Overlay experimental rates
        :param experiment_percentage: how much of the field to select
        :param method: how to select the cells
        """
        cells_to_experiment = self.select_experimental_plots(
            experiment_percentage=experiment_percentage, method=method
        )
        self.assign_experimental_rates(cells_to_experiment)

    def select_experimental_plots(self, experiment_percentage, method="stdev"):
        """
        Select the cells to apply an experimental rate to
        :param experiment_percentage: how much of the field to select
        :param method: how to select the cells
        :return: list of cell indeces
        """
        to_select = round(len(self.field.cell_list) * experiment_percentage)
        if method == "stdev":
            stdev_sorted = sorted(
                self.yield_predictor.cell_prediction["stdev"].items(), key=lambda x: x[1]
            )
            cell_ids = stdev_sorted[-to_select:][0]
        elif method == "lowyield":
            yield_sorted = sorted(
                self.yield_predictor.cell_prediction["average"].items(), key=lambda x: x[1]
            )
            cell_ids = yield_sorted[-to_select:][0]
        else:
            cell_ids = random.sample(range(len(self.field.cell_list)), k=to_select)
        return cell_ids

    def assign_experimental_rates(self, cell_ids, defined_rates=None, max_overall_fertilizer=150):
        if defined_rates is not None:
            # if we have pre-defined rates, sample from these rates
            for id in cell_ids:
                self.complete_variables[id].nitrogen = random.choice(defined_rates)
        else:
            # sample uniform at random from the above intervals,
            # where weights are defined based on the size of the intervals
            # i.e., if the low rates interval is larger, the likelihood of sampling a low rate is higher
            fert_list = [cell.nitrogen for cell in self.complete_variables]
            max_prescribed_fertilizer = max(fert_list)
            min_prescribed_fertilizer = min(fert_list)
            intervals = [
                [0, min_prescribed_fertilizer],
                [max_prescribed_fertilizer, max_overall_fertilizer],
            ]
            for id in cell_ids:
                self.complete_variables[id].nitrogen = random.uniform(
                    *random.choices(intervals, weights=[r[1] - r[0] for r in intervals])[0]
                )

    def minimize_weeds(self):
        """
        for organic maps only.
        predict weeds volume using prediction model
        """
        from sklearn.preprocessing import MinMaxScaler

        weeds_predictions = self.yield_predictor.calculate_weeds(cnn=self.yield_predictor.cnn_bool)
        scaler = MinMaxScaler()
        scaler.fit(weeds_predictions.reshape(-1, 1))
        avg_weeds = np.mean(scaler.transform(weeds_predictions.reshape(-1, 1)))
        return avg_weeds
