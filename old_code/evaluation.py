import random
from random import randint

def mean( xs):
    return sum( xs) / float( len( xs))
# end def

def var(xs):
    n = float( len( xs))
    mean = sum( xs) / n
    sse = sum([(x - mean)**2 for x in xs])
    return sse / n
# end def

def create_bootstrap_function( replications, metric=mean):
    def bootstrap( fitnesses):
        n = len( fitnesses)
        replicates = []
        for i in range( replications):
            sample = [fitnesses[ randint(0, n - 1)] for i in range( n)]
            replicates.append( metric( sample))
        replicates.sort()
        return replicates
    return bootstrap
# end def

def analyze_bootstrap_sample( replicates):
    bootstrap_samples = len( replicates)

    two_and_a_half = int((bootstrap_samples - (0.95 * bootstrap_samples)) / 2)
    lower_bound = two_and_a_half - 1
    upper_bound = bootstrap_samples - (two_and_a_half + 1)

    return replicates[ 0], replicates[ lower_bound], mean( replicates), replicates[ upper_bound], replicates[ -1]
# end def
