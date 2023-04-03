
import numpy as np
import factibilidad as fc
from cmath import inf

def fitness(SCP, sol):
    cost = SCP["C_s"]
    feasible = fc.check_sol(SCP, sol)[0]
    if not feasible:
        return inf
    else:
        fitness = np.dot(sol, cost)
        return fitness