import numpy as np
import random


class Particle(object):
    def __init__(self, f, size, factor=None, global_solution=None):
        self.f = f
        self.position = np.random.uniform(f.lbound, f.ubound, size=size)
        self.lbest_position = self.position
        self.dim = size
        self.factor = factor
        self.velocity = np.zeros(size)
        # TODO: change this to accept MOO fitness i.o. single fitness: this means changing the fitness comparison
        self.fitness = self.calculate_fitness(self.position, global_solution)
        self.lbest_fitness = float('inf')

    def __le__(self, other):
        if self.lbest_fitness is float:
            return self.lbest_fitness <= other.lbest_fitness

    def __lt__(self, other):
        if self.lbest_fitness is float:
            return self.lbest_fitness < other.lbest_fitness

    def __gt__(self, other):
        if self.lbest_fitness is float:
            return self.lbest_fitness > other.lbest_fitness

    def __eq__(self, other):
        return (self.position == other.position).all()

    def __str__(self):
        return ' '.join(
            ['Particle with current fitness:', str(self.fitness), 'and best fitness:', str(self.lbest_fitness)])

    def set_fitness(self, fit):
        self.fitness = fit
        if fit < self.lbest_fitness:
            self.lbest_fitness = fit
            self.lbest_position = self.position

    def set_position(self, position):
        self.position = np.array(position)

    def update_individual_after_compete(self, position, global_solution=None):
        self.position = position
        self.fitness = self.calculate_fitness(position, global_solution)
        return self

    def calculate_fitness(self, position, glob_solution):
        if glob_solution is None:
            return self.f.run(position)
        else:
            solution = [x for x in glob_solution]
            for i, x in zip(self.factor, position):
                solution[i] = x
            return self.f.run(np.array(solution))

    def update_particle(self, omega, phi, global_best, v_max, global_solution=None):
        self.update_velocity(omega, phi, global_best, v_max)
        self.update_position(global_solution)

    def update_velocity(self, omega, phi, global_best, v_max):
        velocity = self.velocity
        n = self.dim

        inertia = np.multiply(omega, velocity)
        phi_1 = np.array([random.random() * phi for i in range(n)])  # exploration
        personal_exploitation = self.lbest_position - self.position  # exploitation
        personal = phi_1 * personal_exploitation
        phi_2 = np.array([random.random() * phi for i in range(n)])  # exploration
        social_exploitation = global_best.position - self.position  # exploitation
        social = phi_2 * social_exploitation
        new_velocity = inertia + personal + social
        self.velocity = np.array([self.clamp_value(v, -v_max, v_max) for v in new_velocity])

    def update_position(self, global_solution=None):
        lo, hi = self.f.lbound, self.f.ubound
        position = self.velocity + self.position
        position = np.array([self.clamp_value(p, lo, hi) for p in position])
        fitness = self.calculate_fitness(position, global_solution)

        if fitness < self.lbest_fitness:
            self.lbest_position, self.lbest_fitness = position, fitness
        self.position, self.fitness = position, fitness

    def clamp_value(self, to_clamp, lo, hi):
        if lo < to_clamp < hi:
            return to_clamp
        if to_clamp < lo:
            return to_clamp
        return hi


class PSO(object):
    def __init__(self, generations, pop_size, f, dim, factor=None, global_solution=None, omega=0.729, phi=1.49618):
        self.pop_size = pop_size
        self.pop = [Particle(f, dim, factor, global_solution) for x in range(pop_size)]
        pos = [p.position for p in self.pop]
        with open('pso2.o', 'a') as file:
            file.write(str(pos))
            file.write('\n')

        self.omega = omega
        self.phi = phi
        self.f = f
        self.dim = dim
        pbest_particle = Particle(f, dim, factor, global_solution)
        pbest_particle.set_fitness(float('inf'))
        self.pbest_history = [pbest_particle]
        self.gbest = pbest_particle
        self.v_max = abs((f.ubound - f.lbound))
        self.generations = generations
        self.current_loop = 0
        self.factor = np.array(factor)
        self.global_solution = global_solution

    def find_current_best(self):
        return min(self.pop)

    def find_local_best(self):
        pass

    def update_swarm(self):
        global_solution = [x for x in self.global_solution]
        omega, phi, global_best, v_max = self.omega, self.phi, self.gbest, self.v_max
        for p in self.pop:
            p.update_particle(omega, phi, global_best, v_max, global_solution)
        curr_best = self.find_current_best()
        self.pbest_history.append(curr_best)
        self.gbest = min(curr_best, self.gbest)

    def replace_worst_solution(self, global_solution):
        # find worst particle
        self.global_solution = np.array([x for x in global_solution])
        worst_particle = max(self.pop)
        worst_particle_index = [i for i, x in enumerate(self.pop) if x == worst_particle]
        partial_solution = [x for i, x in enumerate(global_solution) if i in self.factor] # if i in self.factor
        self.pop[worst_particle_index[0]].set_position(partial_solution)
        self.pop[worst_particle_index[0]].set_fitness(self.f.run(self.global_solution))

    def run(self):
        for i in range(self.generations):
            self.update_swarm()
            self.current_loop += 1
            # print('-----------------------------------------')
            # print(self.pbest_history[i])
        return self.gbest.position


if __name__ == '__main__':
    from refactoring.optimizationproblems.continuous_functions import Function

    f = Function(function_number=1, shift_data_file="f01_o.txt")
    pso = PSO(10, 10, 10, f)
    pso.run()
    # [print(x) for x in pso.pop]
