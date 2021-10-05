#    This file is part of DEAP.
#    Edited by: https://github.com/mbelmadani/moead-py/blob/master/knapsack.py
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
from collections import namedtuple


KsItem = namedtuple('Item', ('weight', 'value', 'volume'))


class Knapsack:
    def __init__(self, number_of_items=100, max_bag_weight=50, max_nr_items=100, max_bag_volume=250, nr_objectives=3):
        self.init_size = 5
        self.max_nr_items = max_nr_items
        self.max_bag_weight = max_bag_weight
        self.max_bag_volume = max_bag_volume
        self.number_of_items = number_of_items
        self.objective_values = (1e4, 1e4, 1e4)
        self.ref_point = (1e5, 1e5, 1e5)
        self.nr_objectives = nr_objectives

        # Create random items
        self.total_items = []
        for i in range(self.number_of_items):
            item = KsItem(random.uniform(0, 5), random.uniform(0.1, 100), random.uniform(0.1, 10))
            self.total_items.append(item)

    def set_fitness(self, variables):
        items_in_knapsack = [x for i, x in enumerate(self.total_items) if variables[i] == 1]
        objective_values = self.eval_knapsack(items_in_knapsack)
        if self.nr_objectives == 4:
            objective_values.append(self.eval_knapsack_balanced_weight(items_in_knapsack))
        if self.nr_objectives == 5:
            objective_values.append(self.eval_knapsack_balanced_volume(items_in_knapsack))
        self.objective_values = objective_values
        #print('fitness ', objective_values)

    def eval_knapsack(self, individual):
        weight = 0.0
        value = 0.0
        volume = 0.0
        for item in individual:
            weight += item.weight
            value += item.value
            volume += item.volume
        #print(weight, volume)
        if len(individual) > self.max_nr_items:
            return (1e4, 1e4, 1e4)  # Ensure overweighted bags are dominated
        elif weight > self.max_bag_weight:
            return (1e4, 1e4, 1e4)
        elif volume > self.max_bag_volume:
            return (1e4, 1e4, 1e4)
        elif len(individual) == 0:
            return (1e4, 1e4, 1e4)
        return (weight, -value, volume)

    def eval_knapsack_balanced_weight(self, individual):
        """
        Variant of the original weight-value knapsack problem with added third object being minimizing weight difference between items.
        """
        balance = 0.0
        for a, b in zip(individual, list(individual)[1:]):
            balance += abs(a.weight-b.weight)
        return balance

    def eval_knapsack_balanced_volume(self, individual):
        """
        Variant of the original weight-value knapsack problem with added third object being minimizing weight difference between items.
        """
        balance = 0.0
        for a, b in zip(individual, list(individual)[1:]):
            balance += abs(a.volume-b.volume)
        return balance