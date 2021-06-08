# coding: utf-8
"""
# Benchmark Functions

These Benchmark functions were selected from the article [A Literature Survey of Benchmark Functions For Global
Optimization Problems](https://arxiv.org/pdf/1308.4008.pdf).
In general, we have picked as many benchmark functions as possible with the condition that they be scalable to arbitrary
numbers of dimensions and that they represent differing degrees of decomposability.

Based on [Virtual Library of Simulation Experiments: Test Functions and Data Sets](
https://www.sfu.ca/~ssurjano/optimization.html) taxonomy, we can categorize the Benchmark functions by various
classes such as **Many Local Optima** (for example, Eggholder), **Bowl Shaped** (for example, Sphere),
**Plate Shaped** (for example, Zakharov), **Valley Shaped** (for example, Rosenbrock), **Steep Ridges** (for example,
Shaffer F6) and **Other**. Of course, some Benchmark functions might fall into more than one category. For example,
Rastrigin might be included in both the Bowl Shaped and Many Local Optima categories.

Of course, one person's plate is another person's bowl so these categories are approximate. Since not all the
functions in *Benchmarks* are covered in *Virtual Library*, we had to use our judgment in those cases.

Although we picked functions that were scalable to arbitrary dimensions, the functions are not all defined over the
same domains. We are unaware experiments that show how domain size affects performance of PSO vis-a-vis the number of
particles independent of functional complexity or dimensionality. In any case, it seems *a priori* reasonable that a
larger number of particles is required for a domain of (-1000, 1000) than (-5, 5), other things being equal.

Because many of the functions are defined on the domain (-10, 10) and others retain their interesting characteristics
on that domain, unless otherwise noted, we have used that domain for all experiments so that results across functions
would have one less difference. The only exceptions are for the Eggholder function and any experiments specifically
involving domains. This means the results are not necessarily directly comparable with others in "The Literature" but
we are not convinced that this is always the case anyway.

## Exploration Functions

Because of the computational expense involved in evaluating so many functions, we start experiments with easier
functions from each class:

* Bowl - Sphere
* Many Local Optima - Griewank
* Valley - Rosebrock
* Plate - Zakharov
* Ridge - Exponential

In[1]:


get_ipython().magic('matplotlib inline')


In[2]:
"""

from __future__ import division
import math
import random
from functools import reduce


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import seaborn as sns

# import numpy as np

# import IPython.display as display
# import seaborn

# colormap = cm.viridis
# sns.set(style="whitegrid")


# ## Helpers

# In[3]:


def greater_than(x, y):
    return x > y


def lesser_than(x, y):
    return x < y


def product(xs):
    return reduce(lambda x, y: x * y, xs)


# In[4]:


def draw_benchmark(params, interval=(-10.0, 10.0)):
    pass


"""
    if not interval:
        interval = params["interval"]
    step = (interval[1] - interval[0]) / 40.0
    xs = math.arange(interval[0], interval[1], step)
    ys = math.arange(interval[0], interval[1], step)
    xs, ys = math.meshgrid(xs, ys)
    f = params[ "function"]
    zs = math.array(map( lambda ks: map( f, ks), map( zip, xs, ys)))

    print "min:", math.min( zs.flatten())

    # Plot the 3d Surface Plot.
    figure = plt.figure(figsize=(10,6))

    axes = figure.add_subplot(1, 2, 1, projection="3d")
    surf = axes.plot_surface(xs, ys, zs, cmap=colormap, linewidth=0, antialiased=True)

    # Plot the 2d Color Projection.
    axes = figure.add_subplot(1, 2, 2)
    axes.pcolor(xs, ys, zs, cmap=colormap, antialiased=True)

    plt.show()
    plt.close()
# def
"""

# # Benchmarks

# In[5]:


_benchmarks = {}


# ## Sphere
#
# $\sum^d_i x_i^2$

# In[6]:


def sphere(xs):
    return sum([x ** 2 for x in xs])

def sphere_string():
    return "sum([x ** 2 for x in d])"

# end def

_benchmarks["sphere"] = {
    "name": "sphere",
    "function": sphere,
    "category": "bowl",
    "interval": (0.0, 10.0),
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum^d_i x_i^2$",
    "variable-domain": True,
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[7]:


draw_benchmark(_benchmarks["sphere"])


# ## Ackley-1
#
# $-20e^{-0.02\sqrt{d^{-1}\sum^d_i x^2_i}}-e^{d^{-1}\sum^d_i cos(2\pi x_i)}+20+e$
#
# This is the correct definition. The minimum must be off because of, you know, floating point representations.

# In[8]:


def ackley_1(xs):
    d = len(xs)
    a = (1.0 / d) * sum([x ** 2 for x in xs])
    a = -0.02 * math.sqrt(a)
    a = -20.0 * math.exp(a)
    b = (1.0 / d) * sum([math.cos(2.0 * math.pi * x) for x in xs])
    b = math.exp(b)
    return a - b + 20.0 + math.e


# end def

_benchmarks["ackley-1"] = {
    "name": "ackley-1",
    "function": ackley_1,
    "category": "local optima",
    "interval": (-35.0, 35.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$-20e^{-0.02\sqrt{d^{-1}\sum^d_i x^2_i}}-e^{d^{-1}\sum^d_i cos(2\pi x_i)}+20+e$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[9]:


draw_benchmark(_benchmarks["ackley-1"])


# ## Brown
#
# $\sum^{d-1}_i (x^2_i)^{(x^2_{i+1}+1)}+(x^2_{i+1})^{(x^2_i+1)}$
#
#

# In[10]:


def brown(xs):
    result = 0.0
    for i in range(0, len(xs) - 1):
        a = (xs[i] ** 2) ** (xs[i + 1] ** 2 + 1)
        b = (xs[i + 1] ** 2) ** (xs[i] ** 2 + 1)
        result += a + b
    return result


# end def

_benchmarks["brown"] = {
    "name": "brown",
    "function": brown,
    "category": "plate",
    "interval": (-1.0, 4.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum^{d-1}_i (x^2_i)^{(x^2_{i+1}+1)}+(x^2_{i+1})^{(x^2_i+1)}$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[11]:


draw_benchmark(_benchmarks["brown"])


# ## Dixon-Price
#
# Mathematical version:
#
# $(x_1 - 1)^2 + \sum^d_{i=2} i (2x^2_i - x_{i-1})^2$
#
# Programming version:
#
# $(x_0 - 1)^2 + \sum^d_{i=1} (i + 1)(2x^2_i - x_{i-1})^2$
#
# Note: the empirical minimum shown is not the actual minimum because the actual minimum is not on the grid.

# In[12]:


def dixon_price(xs):
    term1 = (xs[0] - 1) ** 2
    term2 = 0.0
    for i in range(1, len(xs)):
        home = i + 1  # because math and programming indices are different.
        term = ((2.0 * xs[i] ** 2) - xs[i - 1]) ** 2
        term2 += home * term
    return term1 + term2


# end def

def dixon_price_x(i):
    return 2.0 ** (-1.0 * ((2 ** i - 2) / 2 ** i))  # there's a typo in the Survey paper.


# def

_benchmarks["dixon-price"] = {
    "name": "dixon-price",
    "function": dixon_price,
    "category": "valley",
    "interval": (-10.0, 10.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$(x_1 - 1)^2 + \sum^d_{i=2} i (2x^2_i - x_{i-1})^2$",
    "optimum": {
        "xs": [dixon_price_x(1), dixon_price_x(2), dixon_price_x(3)],  # math indices!
        "y*": 0
    }
}

# In[13]:


draw_benchmark(_benchmarks["dixon-price"])


# ## Eggholder
#
# $\sum^{d-1}_i[-(x_{i+1}+47)sin \sqrt{|x_{i+1} + x_i/2 + 47|} - x_i sin \sqrt{|x_i - (x_{i+1} + 47)|}$

# In[14]:


def eggholder(xs):
    result = 0.0
    for i in range(0, len(xs) - 1):
        a1 = -1.0 * (xs[i + 1] + 47.0)
        a2 = math.sqrt(math.fabs(xs[i + 1] + (xs[i] / 2.0) + 47.0))
        a = a1 * math.sin(a2)

        b1 = -1.0 * xs[i]
        b2 = math.sqrt(math.fabs(xs[i] - (xs[i + 1] + 47)))
        b = b1 * math.sin(b2)
        result += (a + b)
    return result


# end def

_benchmarks["eggholder"] = {
    "name": "eggholder",
    "function": eggholder,
    "category": "local optima",
    "interval": (-512.0, 512.0),
    "variable-domain": False,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum^{d-1}_i[-(x_{i+1}+47)sin \sqrt{|x_{i+1} + x_i/2 + 47|} - x_i sin \sqrt{|x_i - (x_{i+1} + 47)|}$",
    "optimum": {
        "xs": [512.0, 404.2309],
        "y*": -959.6407
    }
}
# -61.62 in (-10.0, 10)


# In[15]:


draw_benchmark(_benchmarks["eggholder"], None)


# ## Exponential
#
# $-exp(-0.5\sum^d_i x_i^2)$

# In[16]:


def exponential(xs):
    summation = sum([x ** 2 for x in xs])
    return -1.0 * math.exp(-0.5 * summation)


# end def

_benchmarks["exponential"] = {
    "name": "exponential",
    "function": exponential,
    "category": "ridge",
    "interval": (-1.0, 1.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$-exp(-0.5\sum^d_i x_i^2)$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": -1.0  # there's a typo in the Survey paper.
    }
}

# In[17]:


draw_benchmark(_benchmarks["exponential"])


# ## Griewank
#
# $\sum^d_i \frac{x^2}{4000} - \prod^d_i cos(\frac{x_i}{\sqrt i}) + 1$

# In[18]:


def griewank(xs):
    a = (1 / 4000) * sum([x ** 2 for x in xs])
    b = product([math.cos(x / math.sqrt(i + 1)) for i, x in enumerate(xs)])
    return a - b + 1


_benchmarks["griewank"] = {
    "name": "griewank",
    "function": griewank,
    "category": "local optima",
    "interval": (-100.0, 100.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum^d_i \frac{x^2}{4000} - \prod^d_i cos(\frac{x_i}{\sqrt i}) + 1$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[19]:


draw_benchmark(_benchmarks["griewank"])


# ## Michalewicz
#
# Note: This function is not included in the Benchmark paper.
#
# $f(x) = \sum_i^d sin(x_i) sin^{2m}(\frac{ix_i^2}{\pi})$
#
# $m$ controls the steepness of the ridges. The suggested value for $m$ is 10.

# In[20]:


def michalewicz(xs):
    result = 0.0
    for i, x in enumerate(xs):
        a = math.sin(x)
        b = (i + 1) * x ** 2
        b = b / math.pi
        b = math.sin(b)
        b = b ** (2 * 10)
        result += a * b
    return result


# def

_benchmarks["michalewicz"] = {
    "name": "michalewicz",
    "function": michalewicz,
    "category": "ridge",
    "interval": (0, 10),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[21]:


draw_benchmark(_benchmarks["michalewicz"])


# ## Rastrigin
#
# Note: This function is not in the Benchmark article.
#
# $10d + \sum_i^d (x_i^2 + 10cos(2\pi x_i))$

# In[22]:


def rastrigin(xs):
    d = float(len(xs))
    return 10.0 * d + sum(map(lambda x: x ** 2 - 10 * math.cos(2.0 * math.pi * x), xs))


# def

_benchmarks["rastrigin"] = {
    "name": "rastrigin",
    "function": rastrigin,
    "category": "local optima",
    "interval": (-5.12, 5.12),
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$10d + \sum_i^d (x_i^2 + 10cos(2\pi x_i))$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[23]:


draw_benchmark(_benchmarks["rastrigin"])


# ## Rosenbrock
#
# $f(x) = \sum^{d-1}_i [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]$

# In[24]:


def rosenbrock(xs):
    result = 0.0
    for i in range(0, len(xs) - 1):
        a = 100 * (xs[i + 1] - xs[i] ** 2) ** 2
        b = (xs[i] - 1) ** 2
        result += a + b
    return result


# end def
_benchmarks["rosenbrock"] = {
    "name": "rosenbrock",
    "function": rosenbrock,
    "category": "valley",
    "interval": (-10.0, 10.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum^{d-1}[100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]$",
    "optimum": {
        "xs": [1.0, 1.0, 1.0, 1.0, 1.0],
        "y*": 0.0
    }
}

# In[25]:


draw_benchmark(_benchmarks["rosenbrock"])


# ## Salomon
#
# $f(x) = 1 - cos(2 \pi \sqrt{\sum_i^d x_i^2}) + 0.1\sqrt{\sum_i^d x_i^2}$

# In[26]:


def salomon(xs):
    total = math.sqrt(sum([x ** 2 for x in xs]))
    return 1 - math.cos(2 * math.pi * total) + 0.1 * total


# end def

_benchmarks["salomon"] = {
    "name": "salomon",
    "function": salomon,
    "category": "local optima",
    "interval": (-100.0, 100.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$1 - cos(2 \pi \sqrt{\sum_i^d x_i^2}) + 0.1\sqrt{\sum_i^d x_i^2}$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[27]:


draw_benchmark(_benchmarks["salomon"])


# ## Sargan
#
# $f(x) = d \sum_i^d (x_i^2 + 0.4 \sum_{j=2}^d x_i x_j)$

# In[28]:


def sargan(xs):
    n = len(xs)
    result = 0.0
    for i in range(0, n):
        subresult = 0.0
        for j in range(1, n):
            subresult += xs[i] * xs[j]
        result += (xs[i] ** 2 + 0.4 * subresult)
    return n * result


# end def

_benchmarks["sargan"] = {
    "name": "sargan",
    "function": sargan,
    "category": "bowl",
    "interval": (-10.0, 10.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[29]:


draw_benchmark(_benchmarks["sargan"])


# ## Schwefel 1.2
#
# $f(x) = \sum_i^d (\sum_j^i x_j)^2$

# In[30]:


def schwefel_1_2(xs):
    n = len(xs)
    result = 0.0
    for i in range(0, n):
        subresult = 0.0
        for j in range(0, i + 1):
            subresult += xs[j]
        result += subresult ** 2
    return result


# end def

_benchmarks["schwefel-1.2"] = {
    "name": "schwefel-1.2",
    "function": schwefel_1_2,
    "category": "valley",
    "interval": (-100.0, 100.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum_i^d (\sum_j^i x_j)^2$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[31]:


draw_benchmark(_benchmarks["schwefel-1.2"])


# ## Schwefel 2.22
#
# $f(x) = \sum_i^d |x_i| + \prod_i^d |x_i|$

# In[32]:


def schwefel_2_22(xs):
    a = sum([math.fabs(x) for x in xs])
    b = product([math.fabs(x) for x in xs])
    return a + b


# end def

_benchmarks["schwefel-2.22"] = {
    "name": "schwefel-2.22",
    "function": schwefel_2_22,
    "category": "ridge",
    "interval": (-100.0, 100.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum_i^d |x_i| + \prod_i^d |x_i|$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[33]:


draw_benchmark(_benchmarks["schwefel-2.22"])


# ## Schwefel 2.23
#
# $f(x) = \sum_i^d x_i^{10}$

# In[34]:


def schwefel_2_23(xs):
    return sum([x ** 10 for x in xs])


# end def

_benchmarks["schwefel-2.23"] = {
    "name": "schwefel-2.23",
    "function": schwefel_2_23,
    "category": "plate",
    "interval": (-10.0, 10.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum_i^d x_i^{10}$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[35]:


draw_benchmark(_benchmarks["schwefel-2.23"])


# ## Schaffer F6
#
# $\sum_i^{d-1} 0.5 + \frac{sin^2(\sqrt{x_i^2 + x_{i+1}^2}) - 0.5}{1 + 0.001 (x_i^2 + x_{i+1}^2)^2}$
#
# Interestingly enough, if you take out the square term, you get a more regular shape with ridges:
#
# $\sum_i^{d-1} 0.5 + \frac{sin(\sqrt{x_i^2 + x_{i+1}^2}) - 0.5}{1 + 0.001 (x_i^2 + x_{i+1}^2)^2}$

# In[36]:


def schaffer_f6(xs):
    result = 0.0
    for i in range(0, len(xs) - 1):
        x_factor = xs[i] ** 2 + xs[i + 1] ** 2
        numerator = math.sin(math.sqrt(x_factor)) ** 2 - 0.5
        denominator = (1.0 + 0.001 * (x_factor)) ** 2
        result += 0.5 + numerator / denominator
    return result


# end def

_benchmarks["schaffer-f6"] = {
    "name": "schaffer-f6",
    "function": schaffer_f6,
    "category": "ridge",
    "interval": (-10.0, 10.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum_i^d 0.5 + \frac{sin^2(\sqrt{x_i^2 + x_{i+1}^2}) - 0.5}{1 + 0.001 (x_i^2 + x_{i+1}^2)^2}$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[37]:


draw_benchmark(_benchmarks["schaffer-f6"])


# ## Stretched V
#
# $f(x) = \sum_i^{d-1} (x_{i+1}^2 + x_i^2)^{0.25}[sin^2[50(x_{i+1}^2 + x_i^2)^{0.1}] + 0.1]$

# In[38]:


def stretched_v(xs):
    result = 0.0
    for i in range(0, len(xs) - 1):
        x_factor = xs[i + 1] ** 2 + xs[i] ** 2
        result += (x_factor ** 0.25) * (math.sin(50.0 * x_factor ** 0.1) ** 2 + 0.1)
    return result


# end def

_benchmarks["stretched-v"] = {
    "name": "stretched-v",
    "function": stretched_v,
    "interval": (-10.0, 10.0),
    "category": "local optima",
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum_i^{d-1} (x_{i+1}^2 + x_i^2)^{0.25}[sin^2[50(x_{i+1}^2 + x_i^2)^{0.1}] + 0.1]$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[39]:


draw_benchmark(_benchmarks["stretched-v"])


# ## Whitley
#
# $\sum_i^d \sum_j^d [\frac{(100(x_i^2 - x_j)^2 + (1 - x_j)^2)^2}{4000} - cos(100(x_i^2 - x_j)^2 + (1 - x_j)^2) + 1]$

# In[40]:


def whitley(xs):
    result = 0.0
    for i in range(0, len(xs)):
        sub_result = 0.0
        for j in range(0, len(xs)):
            a = (100 * (xs[i] ** 2 - xs[j]) ** 2 + (1 - xs[j]) ** 2) ** 2
            a = a / 4000.0
            b = 100 * (xs[i] ** 2 - xs[j]) ** 2 + (1 - xs[j]) ** 2
            b = math.cos(b)
            sub_result += (a - b + 1.0)
        result += sub_result
    return result


# end def

_benchmarks["whitley"] = {
    "name": "whitley",
    "function": whitley,
    "category": "plate",
    "interval": (-10.0, 10.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum_i^d \sum_j^d [\frac{(100(x_i^2 - x_j)^2 + (1 - x_j)^2)^2}{4000} - cos(100(x_i^2 - x_j)^2 + (1 - x_j)^2 + 1)]$",
    "optimum": {
        "xs": [1.0, 1.0, 1.0, 1.0, 1.0],
        "y*": 0.0
    }
}

# In[41]:


draw_benchmark(_benchmarks["whitley"])


# ## Zakharov
#
# Mathematical version:
#
# $f(x) = \sum_i^d x_i^2 + (0.5 \sum_i^d ix_i)^2 + (0.5\sum_i^d ix_i)^4$
#
# Programming version:
#
# $f(x) = \sum_i^d x_i^2 + (0.5 \sum_i^d (i+1)x_i)^2 + (0.5\sum_i^d (i+1)x_i)^4$

# In[42]:


def zakharov(xs):
    a = sum([x ** 2 for x in xs])
    b = (0.5 * sum([(i + 1) * x for i, x in enumerate(xs)])) ** 2
    c = (0.5 * sum([(i + 1) * x for i, x in enumerate(xs)])) ** 4
    return a + b + c


# end def

_benchmarks["zakharov"] = {
    "name": "zakharov",
    "function": zakharov,
    "category": "plate",
    "interval": (-5.0, 10.0),
    "variable-domain": True,
    "dimensions": None,
    "comparator": lesser_than,
    "latex": "$\sum_i^d x_i^2 + (0.5 \sum_i^d ix_i)^2 + (0.5\sum_i^d ix_i)^4$",
    "optimum": {
        "xs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "y*": 0.0
    }
}

# In[43]:


draw_benchmark(_benchmarks["zakharov"])

# In[ ]:
