"""

"""

import random
from collections import namedtuple

SingleKsItem = namedtuple('Item', ('weight', 'value', 'volume'))


class Knapsack:
    def __init__(self, number_of_items=100, max_bag_weight=1250, max_nr_items=100, max_bag_volume=2500, nr_objectives=3,
                 nr_constraints=1, knapsack_type='multi'):
        """
        The Multi-Objective Knapsack base problem.
        @param number_of_items:
        @param max_bag_weight:
        @param max_nr_items:
        @param max_bag_volume:
        @param nr_objectives:
        """
        self.init_size = 5
        self.max_nr_items = max_nr_items
        self.max_bag_weight = max_bag_weight
        self.max_bag_volume = max_bag_volume
        self.number_of_items = number_of_items
        self.objective_values = tuple([1e4 for _ in range(nr_objectives)])
        self.ref_point = tuple([1e5 for _ in range(nr_objectives)])
        self.nr_objectives = nr_objectives
        self.total_items = None
        self.constraints = None
        if knapsack_type == 'single':
            self.initialize_single_knapsack()
        elif knapsack_type == 'multi':
            self.initialize_n_knapsacks()
            self.initialize_constraints(nr_constraints)

    def initialize_single_knapsack(self):
        # Create random items
        self.total_items = []
        for i in range(self.number_of_items):
            item = SingleKsItem(random.uniform(0.1, 5), random.uniform(0.1, 100), random.uniform(0.1, 10))
            self.total_items.append(item)

    def initialize_n_knapsacks(self):
        """
        Initialize the values/profit for each knapsack
        """
        self.total_items = dict()
        for j in range(self.nr_objectives):
            items_in_knapsack = []
            for i in range(self.number_of_items):
                items_in_knapsack.append(random.uniform(0.1, 100))
            self.total_items[j] = items_in_knapsack

    def initialize_constraints(self, nr_constraints):
        """
        Initialize weights for each knapsack
        """
        self.constraints = dict()
        for i in range(nr_constraints):
            items_in_knapsack = []
            for j in range(self.number_of_items):
                items_in_knapsack.append(random.uniform(0.1, 5))
            total_weight = sum(items_in_knapsack)
            print('max weight %d with constraint %f.' % (total_weight, total_weight/2))
            self.constraints[i] = [items_in_knapsack, total_weight/2]

    def set_fitness_multi_knapsack(self, variables):
        """
        fitness function that optimizes across different knapsacks with different capacities given item set.
        Subject to x number of constraints on the weight of the items
        """
        objective_values = []
        for ks in self.total_items.values():
            for c in self.constraints.values():
                total_weight = sum([x for i, x in enumerate(c[0]) if variables[i] == 1])
                if total_weight < c[1]:
                    # print('total weight for current knapsack below constraint: ', total_weight)
                    objective_values.append(-sum([x for i, x in enumerate(ks) if variables[i] == 1]))
                else:
                    # print(' TOO HEAVY: ', total_weight)
                    objective_values.append(1e4)
        #print('found objective values: ', objective_values)
        self.objective_values = tuple(objective_values)

    def set_fitness_single_knapsack(self, variables):
        """
        Fitness function that looks at single knapsack with added volume objective.
        4th objective: balance weight
        5th objective: balance volume
        @param variables:
        @return:
        """
        # print('ks vars len for fitness eval: ', len(variables))
        # print('total item len: ', len(self.total_items))
        items_in_knapsack = [x for i, x in enumerate(self.total_items) if variables[i] == 1]
        objective_values = list(self.eval_knapsack(items_in_knapsack))
        if self.nr_objectives >= 4:
            objective_values.append(self.eval_knapsack_balanced_weight(items_in_knapsack))
        if self.nr_objectives >= 5:
            objective_values.append(self.eval_knapsack_balanced_volume(items_in_knapsack))
        self.objective_values = tuple(objective_values)

    def eval_knapsack(self, individual):
        """

        @param individual:
        @return:
        """
        weight = 0.0
        value = 0.0
        volume = 0.0
        for item in individual:
            weight += item.weight
            value += item.value
            volume += item.volume
        # print(weight, volume)
        if len(individual) > self.max_nr_items:
            return 1e4, 1e4, 1e4  # Ensure overweighted bags are dominated
        elif weight > self.max_bag_weight:
            # print('w ', weight)
            return 1e4, 1e4, 1e4
        elif volume > self.max_bag_volume:
            # print('v ', volume)
            return 1e4, 1e4, 1e4
        elif len(individual) == 0:
            return 1e4, 1e4, 1e4
        return weight, -value, volume

    def eval_knapsack_balanced_weight(self, individual):
        """
        Variant of the original weight-value knapsack problem with added third object being minimizing weight difference between items.
        @param individual:
        @return:
        """
        balance = 0.0
        for a, b in zip(individual, list(individual)[1:]):
            balance += abs(a.weight - b.weight)
        return balance

    def eval_knapsack_balanced_volume(self, individual):
        """
        Variant of the original weight-value knapsack problem with added third object being minimizing weight difference between items.
        @param individual:
        @return:
        """
        balance = 0.0
        for a, b in zip(individual, list(individual)[1:]):
            balance += abs(a.volume - b.volume)
        return balance
