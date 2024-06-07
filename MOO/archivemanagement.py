import itertools
import random
import sys
from operator import itemgetter

import pandas as pd
from pymoo.core.result import Result
from scipy.spatial.distance import pdist, squareform
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
import numpy as np

from utilities.util import delete_multiple_elements, PopulationMember


class ObjectiveArchive:
    def __init__(
        self,
        nr_obj,
        dimensions,
        percent_best=0.25,
        percent_diversity=0.5,
        max_archive_size=100,
        per_objective_size=True,
    ):
        """

        :param nr_obj:
        :param dimensions:
        :param max_archive_size:
        :param per_objective_size:
        :param percent_best: top k percentage of solutions to keep per objective, default is 25%
        :param percent_diversity: l percent of diversity solutions to select from second k% of original solutions, default=50%

        For example:
            100 non-dominated solutions
            k = .25 -> keep "best" 25 solutions
            For diversity:
                Select next 25 "best solutions out of remaining 75
                l = .5 -> Keep at least 25*.5 solutions focusing on diversity
                Half of 25*.5 = 13 because of rounding up
                -> two different diversity approaches with equal solutions
                13*.5 = 7 because of rounding up:
                    1. 7 variable diversity solutions
                    2. 7 objective diversity solutions
                => 14 diversity solutions are being kept
            39 total solutions for one objective
        """
        self.nr_obj = nr_obj
        self.dimensions = dimensions
        if per_objective_size:
            self.max_archive_size = max_archive_size
        else:
            self.max_archive_size = max_archive_size / nr_obj
        self.per_objective_size = per_objective_size
        self.k = percent_best
        self.l = percent_diversity
        self.archive = []
        for i in range(self.nr_obj):
            self.archive.append([])
        # nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in nd_archive]))

    def update_archive(self, found_nondom_solutions):
        """
        for each objective keep:
            select k solutions that have the highest fitness for relevant objective
            (where k is at least double the archive size? or dynamically assign size?)
            Currently, k = top % based on number of objectives, e.g., if there are 10 objectives,
            find the difference between the best and worst solution's fitness for the objective.
            divide that difference by the number of objectives,
            and keep all solutions that have a fitness better than the best solution plus that difference.
            This means you are keeping the best solutions based on the range in the fitness value for each objective,
            keeping the top m% of values based on m objectives.
            This should result in less solutions being kept as the number of objectives go up.
            But since we are using the objective values and not the number of solutions found,
            it should not impact the quality of the solutions being kept per objective.
            Then fill out archive from selected solutions with at least:
                k best solutions
                2*k diversity
        """
        if isinstance(found_nondom_solutions, Result):
            found_nondom_solutions = [
                PopulationMember(variable, fitness, solid=index)
                for index, (variable, fitness) in enumerate(
                    zip(found_nondom_solutions.X, found_nondom_solutions.F)
                )
            ]
        for obj_idx in range(self.nr_obj):
            if self.archive[obj_idx]:
                objective_solutions = [x for x in self.archive[obj_idx]]
                objective_solutions.extend(found_nondom_solutions)
                fitnesses = np.array([np.array(x.fitness) for x in objective_solutions])
                fronts = fast_non_dominated_sort(fitnesses)
                nondom_indeces = fronts[0]
                objective_solutions = [objective_solutions[i] for i in nondom_indeces]
            else:
                objective_solutions = [x for x in found_nondom_solutions]
            found_best_solutions, remaining_solutions = self.keep_best_solutions(
                objective_solutions, obj_idx, k=self.k
            )
            nr_solutions = len(found_best_solutions)
            self.archive[obj_idx] = [x for x in found_best_solutions]
            self.archive[obj_idx].extend(
                self.diversify_archive(
                    remaining_solutions[:nr_solutions], int(nr_solutions * self.l)
                )
            )  # [:int(len(remaining_solutions)/2)], int(len(remaining_solutions)/4)))
            # print("final archive length for objective ", str(obj_idx), len(self.archive[obj_idx]))

    def keep_best_solutions(self, found_nondom_solutions, current_objective, k=0.25):
        """
        Find "best" solutions for a specific objective.
        1. Sort solutions according to objective
        2. Calc difference of best and worst solution for objective to determine the value range of solutions to keep
        -> R = |max(fitness)-min(fitness)| / M (m = nr of objectives)
        3. Keep all solutions with a fitness less than min(fitness)+R (when minimizing objectives)
        :param found_nondom_solutions:
        :param current_objective:
        :return: found "best" solutions and remaining solutions
        """
        solutions = [
            x for x in sorted(found_nondom_solutions, key=lambda x: x.fitness[current_objective])
        ]
        solutions_length = len(solutions)
        end_index = int(solutions_length * k)
        best_solutions = [x for x in solutions[:end_index]]
        del solutions[:end_index]
        return best_solutions, solutions

    def diversify_archive(self, solutions, nr_to_select):
        """
        approximately:
        k (nr_to_select) diverse non-dom solutions in variable space
        k (nr_to_select) of diverse non-dom solutions in objective space
        :param solutions:
        :param nr_to_select:
        :return:
        """
        if nr_to_select <= 3:
            nr_to_select = 4
        diverse_solutions = []

        # get solutions that are the most varied in terms of fitness/objective space
        fitness_solutions = [sol.fitness for sol in solutions]
        fitness_sol_indeces = self.get_diversity_indeces(fitness_solutions, int(nr_to_select / 2))
        diverse_solutions.extend([solutions[i] for i in fitness_sol_indeces])
        # delete found solutions to avoid duplicates in objective archive
        delete_multiple_elements(solutions, fitness_sol_indeces)

        # get solutions that are the most varied in terms of variable/solution space
        var_solutions = [sol.variables for sol in solutions]
        var_sol_indeces = self.get_diversity_indeces(var_solutions, int(nr_to_select / 2))
        diverse_solutions.extend([solutions[i] for i in var_sol_indeces])

        return diverse_solutions

    def find_archive_overlap(self, nr_archives_overlapping=3):
        """
        Finds the solutions that occur in more than one objective archive and adds them to their own solution set.
        :return:
        """

        overlapping_members = []
        for obj_arch in self.archive:
            unique_arch = set([x.solid for x in obj_arch])
            overlapping_members.extend([x for x in unique_arch])
        count_overlap = pd.Series(overlapping_members).value_counts()
        count_overlap = count_overlap.to_dict()
        more_than_four = [
            solid for solid, count in count_overlap.items() if count > nr_archives_overlapping
        ]
        nd_archive = self.flatten_archive()
        final_selected_members = []
        seen = []
        for sol in nd_archive:
            if sol.solid in more_than_four and sol.solid not in seen:
                final_selected_members.append(sol)
                seen.append(sol.solid)

        return final_selected_members

    def flatten_archive(self):
        flattened = []
        for arch in self.archive:
            flattened.extend([x for x in arch])
        return flattened

    def get_diversity_indeces(self, solutions, nr_to_select):
        """
        Find the indeces of the solutions that are the most dissimilar from each other.
        In other words, find the pairwise dissimilarity matrix for all solutions,
         select the k solutions with the highest dissimilarity value,
         and return the indeces of these solutions
        :param solutions: solution set to find most diverse solutions in (either fitness values or actual variables)
        :param nr_to_select: k "top" diversity solutions to select
        :return: indeces of most dissimilar solutions in the provided solution set
        """
        if len(solutions) > nr_to_select:
            # create mapping from condensed matrix to solution indeces
            condensed_matrix_indeces = list(
                itertools.combinations(list(range(0, len(solutions))), 2)
            )
            # use scipy pdist function to get dissimilarity matrix, scipy returns a "reduced matrix"
            dissim_matrix_fitness = pdist(solutions, "cosine")
            # get indeces of max values from reduced matrix
            max_indeces = np.argpartition(dissim_matrix_fitness, -nr_to_select)[-nr_to_select:]
            # map the max indeces to the solution indeces, this returns a list of pairs of indeces
            sol_pairs = list(itemgetter(*max_indeces)(condensed_matrix_indeces))
            # flatten the list of pairs
            sol_indeces = []
            for pair in sol_pairs:
                sol_indeces += pair
            # only return unique indeces
        else:
            print("CHECK THIS OUT:", len(solutions), nr_to_select)
            sol_indeces = [i for i, sol in enumerate(solutions)]
        return set(sol_indeces)

    def get_idx_from_pair_ids(self, i, j, sol_length) -> int:
        """
        For two solutions i and j, find the corresponding index in the reduced scipy matrix
        :param i: first element to compare
        :param j: second element to compare
        :param sol_length: total number of solutions
        :return: index in reduced matrix
        """
        first = (sol_length * (sol_length - 1)) / 2
        idx_length = sol_length - i
        second = (idx_length * (idx_length - 1)) / 2
        index = first - second + (j - i - 1)
        return index


def environmental_solution_selection_nsga2(archive, sol_size):
    fitnesses = np.array([np.array(x.fitness) for x in archive])
    fronts = fast_non_dominated_sort(fitnesses)
    solution_set = []
    i = 0
    for front in fronts:
        if len(front) + len(solution_set) < sol_size:
            solution_set.extend([x for x in front])
            i = i + 1
        else:
            break
    if len(solution_set) < sol_size:
        front_fitness = np.array([np.array(archive[idx].fitness) for idx in fronts[i]])
        size_to_add = sol_size - len(solution_set)
        crowd_dist = calc_crowding_distance(front_fitness)
        sorted_front = [x for y, x in sorted(zip(crowd_dist, fronts[i]))]
        solution_set.extend([x for x in sorted_front[:size_to_add]])
    return [archive[i] for i in solution_set]


"""
METHOD IF ARCHIVE IS OF FIXED SIZE:
     # if len(found_best_solutions) == self.max_archive_size:
    #     self.archive[obj_idx] = [x for x in found_best_solutions]
    # elif len(found_best_solutions) < self.max_archive_size:
    #     self.archive[obj_idx] = [x for x in found_best_solutions]
    #     self.archive[obj_idx].extend(
    #         self.diversify_archive(remaining_solutions, self.max_archive_size - len(found_best_solutions)))
    # else:
    #     nr_solutions = int(self.max_archive_size * self.percent_best)
    #     self.archive[obj_idx] = [x for x in found_best_solutions[:nr_solutions]]
    #     del found_best_solutions[:nr_solutions]
    #     self.archive[obj_idx].extend(
    #         self.diversify_archive(found_best_solutions, self.max_archive_size - nr_solutions))
    
    # delete found solutions to avoid duplicates in objective archive
    # delete_multiple_elements(solutions, var_sol_indeces)
    # add random nondom solutions from remaining solutions
    # leftover = nr_to_select-len(fitness_sol_indeces)-len(var_sol_indeces)
    # random_sols = random.choices(solutions, k=leftover)
    # diverse_solutions.extend(random_sols)
"""

"""
METHOD to keep best solutions within range of objective score
        # if solutions_length < 500:
        #     best_solutions = []
        #     # find the difference between the best and the worst fitness value for the current objective
        #     min_value = solutions[0].fitness[current_objective]
        #     diff = abs(solutions[0].fitness[current_objective] - solutions[-1].fitness[current_objective])
        #     # determine the value range of the solutions to keep
        #     range_to_keep = diff / (self.nr_obj*10)
        #     end_index = 0
        #     # find solutions that fall within the range
        #     for sol_idx in range(solutions_length):
        #         if solutions[sol_idx].fitness[current_objective] < min_value + range_to_keep:
        #             best_solutions.append(solutions[sol_idx])
        #         else:
        #             end_index = sol_idx
        #             break
        # else:
        #     end_index = 500
"""
