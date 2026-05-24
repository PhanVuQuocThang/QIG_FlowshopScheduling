"""
ig.py
---
Iterated Greedy algorithm implementation for flow shop scheduling.
This module defines the IteratedGreedyAlgorithm class, which encapsulates the main loop and state of the algorithm.
It uses the operators and strategies defined in separate modules for modularity and clarity.
The main components are:
- IteratedGreedyAlgorithm: main class with execute() method to run the algorithm
- calc_temp(): calculates the temperature parameter based on processing times
- execute(): main loop that initializes the solution and applies the chosen strategy
The algorithm supports different strategies (individual, random, qlearning) that can be selected at initialization.
"""

import random
import numpy as np
from datetime import datetime, timedelta

from solution import Solution
from initialization import neh
from operators import local_search
import strategy

class IteratedGreedyAlgorithm:

    def __init__(self, instance_processing_times, strategy='individual', d=1):
        # Store processing times internally as a numpy array of shape (m, n)
        self.p_np = np.asarray(instance_processing_times, dtype=np.int64)
        
        # Initialize empty Solution dataclasses (no processing times embedded)
        self.current_solution = Solution(perm=[], cmax=0)
        self.new_solution = Solution(perm=[], cmax=0)
        self.best_solution = Solution(perm=[], cmax=0)
        
        self.tau = 0.7  # temperature parameter
        self.strategy = strategy  # default strategy
        
        ### Settings for QIG, RIG, IIGs, IG_RS, IG_PTL, IG_FF, IG_DPS
        self.tie_breaking_within_NEH = True
        self.tie_breaking_complete_initial_solution = True
        self.tie_breaking_partial_initial = True
        self.tie_breaking_destruction_partial_solution = False
        self.tie_breaking_construction = True
        self.tie_breaking_main_LS = True
        
        self.until_no_improvement = False

        self.main_local_search = 'insertion_neighborhood'
        self.local_search_destruction_partial_solution = None
        self.local_search_on_complete_initial_solution = None
        self.local_search_within_NEH = None
        
        self.ref_best = False  # insertion based on order in best solution
        
        self.operator_list_perturbation = [d] if strategy == 'individual' else [1, 2, 3]  # Values of d (jobs to remove)
        self.exe_time = []
        self.best_fitness_list = []
        self.current_fitness_list = []
        
        # Q-learning parameters
        self.episode_size = 6
        self.epsilon_greedy = 0.8
        self.epsilon_greedy_decay = 0.996
        self.gamma_learning = 0.8
        self.alpha_learning = 0.6

    def execute(self, stopping_criterion, runtime_in_miliseconds, max_iteration):
        """
        Run the algorithm using either Max Iteration or Max CPU Time.
        """
        # 0) Define constant temperature and run time
        if stopping_criterion == 'max_iteration':
            self.max_iteration = max_iteration
            self.runtime = float('inf')
            self.time_limit = datetime.now() + timedelta(milliseconds=runtime_in_miliseconds)
        elif stopping_criterion == 'CPU_time':
            self.max_iteration = float('inf')
            self.runtime = runtime_in_miliseconds
            self.time_limit = datetime.now() + timedelta(milliseconds=runtime_in_miliseconds)
            
        self.iterations = 0
        self.temperature = self.calc_temp()

        ### Generate initial solution ###
        # New neh() takes p_np and returns a Solution instance
        self.current_solution = neh(self.p_np)
        
        # Sync best_solution with the generated initial solution
        self.best_solution.perm = self.current_solution.perm.copy()
        self.best_solution.cmax = self.current_solution.cmax
        
        self.best_fitness_list.append(self.best_solution.cmax)
        
        # Optional local search on complete initial solution
        if self.local_search_on_complete_initial_solution == 'insertion_neighborhood':
            local_search(
                self.p_np, 
                self.current_solution, 
                method='insertion_neighborhood', 
                ref_best=self.best_solution if self.ref_best else None, 
                until_no_improvement=self.until_no_improvement, 
                tie_breaking=self.tie_breaking_complete_initial_solution
            )
        
        # Re-sync best_solution post local search
        if self.current_solution.cmax < self.best_solution.cmax:
            self.best_solution.perm = self.current_solution.perm.copy()
            self.best_solution.cmax = self.current_solution.cmax
        
        self.best_fitness_list.append(self.best_solution.cmax)
        
        """ Main loop of the algorithm """
        if self.strategy == 'individual':
            strategy.ig_individual(self)
        elif self.strategy == 'qlearning':
            strategy.ig_qlearning(self)
        elif self.strategy == 'random':
            strategy.ig_random(self)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def calc_temp(self):
        """ Calculate temperature using simplified p_np mean calculation """
        return self.tau * self.p_np.mean() / 10