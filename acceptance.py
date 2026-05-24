"""
acceptance.py
---
Contains the acceptance criterion for Iterated Greedy, combining greedy acceptance with a Metropolis criterion for worse solutions.
The acceptance function mutates the current solution and best solution in place based on the new solution and the temperature parameter.
The function accepts a new solution if it is better than the current solution, and also updates the best solution if improved.
If the new solution is worse, it may still be accepted with a probability that decreases as the solution gets worse and as the temperature decreases.
"""
import math
import random
from solution import Solution

def acceptance(current_solution: Solution,
                        new_solution: Solution,
                        best_solution: Solution,
                        temperature: float):
    """
    Accept or reject new_solution using greedy + Metropolis criterion.
    Mutates current_solution and best_solution in place.
    """
    if new_solution.cmax <= current_solution.cmax:
        # Accept new solution
        current_solution.perm = new_solution.perm.copy()
        current_solution.cmax = new_solution.cmax

        # Update best solution if improved
        if current_solution.cmax < best_solution.cmax:
            best_solution.perm = current_solution.perm.copy()
            best_solution.cmax = current_solution.cmax
    else:
        # Metropolis acceptance
        delta = new_solution.cmax - current_solution.cmax
        probability = math.exp(-delta / temperature)
        if random.random() <= probability:
            current_solution.perm = new_solution.perm.copy()
            current_solution.cmax = new_solution.cmax