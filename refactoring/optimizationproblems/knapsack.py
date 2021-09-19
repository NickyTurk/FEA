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

from deap import base
from deap import creator
from deap import tools


class Knapsack:
    def __init__(self, number_of_items=100, max_bag_weight=50, max_nr_items=50, max_bag_volume=100):
        self.init_size = 5
        self.max_nr_items = max_nr_items
        self.max_bag_weight = max_bag_weight
        self.max_bag_volume = max_bag_volume
        self.number_of_items = number_of_items


        # Create random items and store them in the items' dictionary.
        self.total_items = []
        for i in range(self.number_of_items):
            self.total_items[i] = (random.randint(1, 10), random.uniform(0, 100))

    def eval_knapsack(self, individual):
        weight = 0.0
        value = 0.0
        for item in individual:
            weight += self.total_items[item][0]
            value += self.total_items[item][1]
        if len(individual) > self.max_nr_items or weight > self.max_bag_weight:
            return 1e30, 0.0 # Ensure overweighted bags are dominated
        return weight, value

    def eval_knapsack_balanced(self, individual):
        """
        Variant of the original weight-value knapsack problem with added third object being minimizing weight difference between items.
        """
        weight, value = self.eval_knapsack(individual)
        balance = 0.0
        for a, b in zip(individual, list(individual)[1:]):
            balance += abs(self.total_items[a][0]-self.total_items[b][0])
        if len(individual) > self.max_nr_items or weight > self.max_bag_weight:
            return weight, value, 1e30 # Ensure overweighted bags are dominated
        return weight, value, balance