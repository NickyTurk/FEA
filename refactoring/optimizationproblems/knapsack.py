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

IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50
NBR_ITEMS = 20


NGEN = 50
MU = 50
LAMBDA = 2
CXPB = 0.7
MUTPB = 0.2

# Create random items and store them in the items' dictionary.
items = {}
for i in range(NBR_ITEMS):
    items[i] = (random.randint(1, 10), random.uniform(0, 100))


def evalKnapsack(individual):
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items[item][0]
        value += items[item][1]
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return 1e30, 0.0 # Ensure overweighted bags are dominated
    return weight, value

def evalKnapsackBalanced(individual):
    """
    Variant of the original weight-value knapsack problem with added third object being minimizing weight difference between items.
    """
    weight, value = evalKnapsack(individual)
    balance = 0.0
    for a,b in zip(individual, list(individual)[1:]):
        balance += abs(items[a][0]-items[b][0])
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return weight, value, 1e30 # Ensure overweighted bags are dominated
    return weight, value, balance