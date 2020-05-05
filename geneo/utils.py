import numpy as np
import math
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

def init_operators(op_type, number=8, size=5, centers=5, sigma=.1):
    """Returns a certain number of operators of type op_type. Now the kwargs are
    the one corresponding to the only kind of operator implemented (IENEO).
    """
    return [op_type(size, sigma = sigma, centers = centers)
            for i in range(number)]

def bin_coeff(n,k):
    if k == n:
        cb = 1
    elif k == 1:
        cb = n
    elif k > n:
        cb = 0
    else:
        a = math.factorial(n)
        b = math.factorial(k)
        c = math.factorial(n-k)
        cb = a / (b*c)
    return int(cb)

def plot_diagram(dgm, ax = None, show = False, labels = False):
    """Modified from dionysus to accept an axis instance as input.
    Plot the persistence diagram. The original code is stored at
    https://github.com/mrzv/dionysus/blob/master/bindings/python/dionysus/plot.py
    """

    inf = float('inf')
    min_birth = min(p.birth for p in dgm if p.birth != inf)
    max_birth = max(p.birth for p in dgm if p.birth != inf)
    min_death = min(p.death for p in dgm if p.death != inf)
    max_death = max(p.death for p in dgm if p.death != inf)
    if ax is None:
        ax = plt.axes()
    ax.set_aspect('equal', 'datalim')
    min_diag = min(min_birth, min_death)
    max_diag = max(max_birth, max_death)
    ax.scatter([p.birth for p in dgm], [p.death for p in dgm])
    ax.plot([min_diag, max_diag], [min_diag, max_diag])
    if labels:
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
    if show:
        plt.show()

def is_included(i1, i2):
    return i1[0] > i2[0] and i1[1] < i2[1]

def get_order_of_magnitude(array):
    epsilon = 10e-6
    q
    return np.log10(np.abs(array)+epsilon).astype(np.int)
