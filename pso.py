# <markdowncell>

# # Particle Swarm Optimization (Basic Algorithm)

# <markcowncell>

# Here are the imports we need. The core library has some nice utility
# functions we can use. `pluck` can return multiple values out of a dictionary
# as a list. `add`, `mul` and `sub` are multiple argument functions that will
# take both scalars and lists and "do the right thing" (no dot product, though).
# We'll need random numbers, of course. And from collections, we use a few
# named tuples where the possible fields have settled down.

# <codecell>
import time

from core import pluck, add, mul, sub, dict_merge, Particle, Random
#import random
from copy import deepcopy
from functools import partial
# import threading
import pathos.pools as Pool

random = Random()
# <markdowncell>

# `initialize_particle` takes n (the dimension of the problem), domain (
# the domain of each (but every) dimension), and f (fitness function).
# One future elaboration might be for different dimensions to have their
# own domains.

# <codecell>

def initialize_particle(n, domain, f):
    """
    A Particle is a Named Tuple with fields: position, velocity and fitness.
    * position is randomly generated with length n. Each element is in the supplied domain.
    * velocity is all 0's of length n.
    * fitness is the value of the supplied function, f, on the position.
    """
    # print("dimemsnions ", n)
    position = [random.uniform( domain[ 0], domain[1]) for i in range( 0, n)]
    # print(position)
    velocity = [0.0 for i in range( 0, n)]
    fitness = f(position)
    return Particle(position, velocity, fitness)


# <markdowncell>

# The PSO basically has two operations that are repreated over and over again:
#
# 1. update the velocity (of each particle).
# 2. update the position (of each particle).
#
# There is also some bookkeeping that requires us to calculate or keep track of
# each particles personal best and the swarm's global best.

# <markdowncell>

# We start with a function to keep the velocity within certain bounds: (-v_max, v_max)

# <codecell>

def clamp_velocity( v, v_max):
    neg_v_max = -v_max
    if neg_v_max < v < v_max:
        return v
    if v < neg_v_max:
        return neg_v_max
    return v_max

# <markdowncell>

# Next we use values for $\omega$ and $\phi$ established in the literature. We should
# really be able to change these.

# <codecell>

omega = 0.729
phi = 1.49618

# <markcowncell>

# The first function we need is one that updates the particle's velocity. We have to think of the
# overall strategy. Do we update all the velocities, all the positions then all the fitnesses
# of the particles or do we update everything for all the particles?
#
# `update_velocity` will take the particle, personal best and global best as well as the
# v_max and update return an updated velocity.
#

# <codecell>

def update_velocity(v_max, particle, personal_best, global_best):
    n = len( particle.position)

    inertia = mul( omega, particle.velocity)

    phi_1 = [random.random() * phi for i in range( n)] # exploration
    personal_exploitation = sub( personal_best.position, particle.position) # exploitation
    personal = mul( phi_1, personal_exploitation)
    phi_2 = [random.random() * phi for i in range( n)] # exploration
    social_exploitation = sub( global_best.position, particle.position) # exploitation
    social = mul( phi_2, social_exploitation)
#    print("position", particle.position)
#    print("pbest", personal_best.position, "gbest", global_best.position)
#    print("phi1=", phi_1)
#    print("phi2=", phi_2)
#    print("personal=", personal)
#    print("social=", social)
    new_velocity = add( inertia, personal, social)
    new_velocity = [clamp_velocity( v, v_max) for v in new_velocity]
    return new_velocity
# end def

# <codecell>

def clamp_position( domain, p):
    lo, hi = domain
    if lo < p < hi:
        return p
    if p < lo:
        return lo
    return hi
# end def

# <codecell>

def update_position( domain, particle, new_velocity):
    new_position = add( particle.position, new_velocity)
    new_position = [clamp_position( domain, p) for p in new_position]
#    print("new position", new_position)
    return new_position
# end def

# <codecell>

def update_particle( domain, v_max, f, global_best, particle, personal_best):
    new_velocity = update_velocity(v_max, particle, personal_best, global_best)
    new_position = update_position(domain, particle, new_velocity)
    new_fitness = f(new_position)
    return Particle(new_position, new_velocity, new_fitness)
# end def

def print_particle_positions( swarm):
    for (i, particle) in enumerate( swarm["particles"]):
        print(i, map( lambda v: "%.4f" % v, particle.position))
    # for
# def

"""
Now all the fun begins...because we don't just want 1 particle and we need to do record keeping.

1. create an initial population.
2. calculate personal best for a population.
3. calculate the global best for a population.

We'll stick with Tuples for now because Python nicely destructures them: (global_best, swarm, personal_bests).
The really nice thing is that Tuples are Python's one and only immutable data structure.
"""

# <codecell>

def find_global_best( personal_bests):
    by_fitness = sorted( personal_bests, key=lambda x: x.fitness)
    return by_fitness[ 0] # tuples are immutable. deepcopy( by_fitness[ 0])
# end def

# <codecell>

# TODO: the main question here is if Swarm should stay a Dict or be a NamedTuple.
def initialize_swarm( p, n, domain, f):
    particles = [initialize_particle( n, domain, f) for i in range(p)]
    personal_bests = deepcopy(particles) # this *should* work down to primitives.
    global_best = find_global_best(personal_bests)
    return {"gbest": global_best, "particles": particles, "pbests": personal_bests, "domain": domain}
# end def

# <codecell>

def find_personal_bests( particles, personal_bests):
    def lesser(a, b):
        if a.fitness < b.fitness:
            return deepcopy(a)
        return deepcopy(b)
    return [lesser(a, b) for a,b in zip(particles, personal_bests)]
# end def

# <codecell>

"""
Updates each particle in the batch

updater: the partially filled in function of update_particle
new_particles: list where output is stored
index: index in global particles list where batch starts so order is preserved

TODO: maybe need lock
"""
def run_batch(params):
    b, updater, new_particles, index = params
    # print("Batch Length: " + str(len(b)))
    # print("New Particles Length: " + str(len(new_particles)))
    for i in range(len(b)):
        particle, personal_best = b[i]
        # Maybe need to lock this, but I doubt it since no thread considers same subset
        # print("Adding at: " + str(index + i))
        new_particle = updater(particle, personal_best)
        new_particles[index + i] = new_particle



def update_swarm(swarm, f):
    global_best, particles, personal_bests, domain = pluck(swarm, "gbest", "particles", "pbests", "domain")

    v_max = (domain[1] - domain[0]) / 2.0

    t_update_start = time.time()

    updater = partial(update_particle, domain, v_max, f, global_best)

    batch_size = 1
    new_particles = [None for _ in particles]  # make blank array so no out of bounds
    batch_args = []

    for i in range(0, len(particles), batch_size): # switch to numeric iteration for baching of threads
        batch = []
        indx = i
        for j in range(batch_size):
            if i + j >= len(particles):
                break
            batch.append((particles[i + j], personal_bests[i + j]))
        # threading.Thread(target=optimize_swarm, args=(swarm, pso_stop, indx, new_swarms, lock))
        if len(batch) > 0:
            batch_args.append([batch, updater, new_particles, indx])
    # update each batch

    # print("Number of updater threads: " + str(len(threads)))
    pool = Pool.ProcessPool(len(batch_args))
    pool.close()
    pool.join()

    t_update_end = time.time()
    print("\t\tTime to update particles: " + str(t_update_end - t_update_start))

    t_find_start = time.time()

    new_personal_bests = find_personal_bests(new_particles, personal_bests)

    t_find_end = time.time()
    print("\t\tTime to find personal bests: " + str(t_find_end - t_find_start))

    #paired_particles = zip( new_particles, new_personal_bests)
    #paired_particles.sort( key=lambda x: x[ 1].fitness)
    #new_particles, new_personal_bests = zip( *paired_particles)

    t_sort_start = time.time()
    sorted_bests = sorted(new_personal_bests, key=lambda x: x.fitness)

    t_sort_end = time.time()
    print("\t\tTime to sort: " + str(t_sort_end - t_sort_start))

    new_global_best = sorted_bests[0]
    new_swarm = {"gbest": new_global_best, "particles": list(new_particles), "pbests": list(new_personal_bests)}

    # if swarm has any metadata/context/etc., this makes sure we preserve it. `dict_merge` makes a
    # copy so this is effectively an immutable operation to one level (it is a shallow copy).
#    print(random.calls(), random.random())
    return dict_merge(swarm, new_swarm)
# end def

# <codecell>

def pso( f, p, n, domain, stop):
    swarm = initialize_swarm(p, n, domain, f)
    gbest = swarm[ "gbest"]
    result = [gbest]
    t = 0
    while not stop( t, gbest.fitness):
        t = t + 1
        swarm = update_swarm( swarm, f)
        gbest = swarm["gbest"]
        result.append(gbest)
    return result
# end def

if __name__ == "__main__":
    from benchmarks import benchmarks
    from evaluation import create_bootstrap_function, analyze_bootstrap_sample
    from topology import generate_linear_topology
    import random
    import json

    names = list(benchmarks.keys())
    names.sort()

    for name in names[:1]:
        with open("results/" + name + "-pso.json", "w") as outfile:
            print("working on", name)
            summary = {"name": name}
            fitnesses = []
            for trial in range(0, 50):
                benchmark = benchmarks[name]
                f = benchmark["function"]
                domain = benchmark["interval"]

#                random.seed(seeds[trial])
                result = pso(f, 80, 8, domain, lambda t, f: t == 100)
                fitnesses.append(result[-1].fitness)
            bootstrap = create_bootstrap_function(250)
            replications = bootstrap(fitnesses)
            statistics = analyze_bootstrap_sample(replications)
            summary["statistics"] = statistics
            summary["fitnesses"] = fitnesses
            json.dump(summary, outfile)
#