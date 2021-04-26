import numpy as np
import random
from copy import deepcopy
import itertools as it
import functools as ft


class Particle(object):
    def __init__(self, f, size):
        self.f = f
        self.position = np.random.uniform(f.lbound, f.ubound, size=size)
        self.lbest = self.position
        self.dim = size
        self.velocity = np.zeros(size)
        self.fitness = f.run(self.position)
        self.lbest_fitness = float('inf')

    def __le__(self, other):
        return self.lbest_fitness <= other.lbest_fitness

    def __lt__(self, other):
        return self.lbest_fitness < other.lbest_fitness

    def __gt__(self, other):
        return self.lbest_fitness > other.lbest_fitness

    def __str__(self):
        return ' '.join(
            ['Particle with current fitness:', str(self.fitness), 'and best fitness:', str(self.lbest_fitness)])

    def set_fitness(self, fit):
        self.fitness = fit

    def update_particle(self, omega, phi, global_best, v_max):
        self.update_velocity(omega, phi, global_best, v_max)
        self.update_position()

    def update_velocity(self, omega, phi, global_best, v_max):
        velocity = self.velocity
        n = self.dim

        inertia = np.multiply(omega, velocity)
        phi_1 = np.array([random.random() * phi for i in range(n)])  # exploration
        personal_exploitation = self.lbest - self.position  # exploitation
        personal = phi_1 * personal_exploitation
        phi_2 = np.array([random.random() * phi for i in range(n)])  # exploration
        social_exploitation = global_best.position - self.position  # exploitation
        social = phi_2 * social_exploitation
        new_velocity = inertia + personal + social
        self.velocity = np.array([self.clamp_value(v, -v_max, v_max) for v in new_velocity])

    def update_position(self):
        lo, hi = self.f.lbound, self.f.ubound
        position = self.velocity + self.position
        position = np.array([self.clamp_value(p, lo, hi) for p in position])
        fitness = self.f.run(position)

        if fitness < self.lbest_fitness:
            self.lbest, self.lbest_fitness = position, fitness
        self.position, self.fitness = position, fitness

    def clamp_value(self, to_clamp, lo, hi):
        if lo < to_clamp < hi:
            return to_clamp
        if to_clamp < lo:
            return to_clamp
        return hi


class PSO(object):
    def __init__(self, generations, pop_size, dim, f, omega=0.729, phi=1.49618):
        self.pop_size = pop_size
        self.pop = [Particle(f, dim) for x in range(pop_size)]
        self.omega = omega
        self.phi = phi
        self.dim = dim
        pbest_particle = Particle(f, dim)
        pbest_particle.set_fitness(float('inf'))
        self.pbest_history = [pbest_particle]
        self.gbest = pbest_particle
        self.v_max = abs((f.ubound - f.lbound))
        self.generations = generations
        self.current_loop = 0

    def find_current_best(self):
        return min(self.pop)

    def update_swarm(self):
        omega, phi, global_best, v_max = self.omega, self.phi, self.gbest, self.v_max
        for p in self.pop:
            p.update_particle(omega, phi, global_best, v_max)
        curr_best = self.find_current_best()
        self.pbest_history.append(curr_best)
        self.gbest = min(curr_best, self.gbest)

    def run(self):
        for i in range(self.generations):
            self.update_swarm()
            self.current_loop += 1
            print('-----------------------------------------')
            print(self.pbest_history[i])
        return self.gbest.position


if __name__ == '__main__':
    from refactoring.optimizationProblems.function import Function

    f = Function(function_number=1, shift_data_file="f01_o.txt")
    pso = PSO(10, 10, 10, f)
    pso.run()
    # [print(x) for x in pso.pop]
