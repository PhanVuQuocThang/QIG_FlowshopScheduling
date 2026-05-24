"""
strategy.py
---
Contains the main loop implementations for different Iterated Greedy strategies:
- ig_individual: fixed perturbation size
- ig_random: random perturbation size 
- ig_qlearning: adaptive perturbation size based on Q-learning
"""

import numpy as np
from datetime import datetime
import time

# Import decoupled operations from our new codebase layout
from operators import perturbation, local_search
from acceptance import acceptance

def ig_qlearning(algo_self):
    """
    Q-learning strategy tracking algorithm performance across episodic destruction choices.
    """
    algo_self.Q_matrix = np.zeros((2, len(algo_self.operator_list_perturbation)))                
    algo_self.state = 0
    episode_size = algo_self.episode_size
    
    algo_self.actions_sequence = []
    algo_self.states_sequence = []
        
    while datetime.now() < algo_self.time_limit and algo_self.iterations < algo_self.max_iteration:        
        current_fit_during_episodes = [algo_self.current_solution.cmax]
        best_fit_during_episodes = [algo_self.best_solution.cmax]
        
        #### Selecting action ####
        if np.random.rand() < algo_self.epsilon_greedy:
            algo_self.action = np.random.randint(len(algo_self.operator_list_perturbation))
        else:
            algo_self.action = np.argmax(algo_self.Q_matrix[algo_self.state])
            
        algo_self.actions_sequence.append(algo_self.action)
        algo_self.states_sequence.append(algo_self.state)

        for ep in range(episode_size):
            # Hard stop: respect time limit even mid-episode
            if datetime.now() >= algo_self.time_limit:
                break

            # 1) Exploration via perturbation: requires deep/shallow copy of current layout to modify
            # Since operators mutate in place, we prep algo_self.new_solution fields first
            algo_self.new_solution.perm = algo_self.current_solution.perm.copy()
            algo_self.new_solution.cmax = algo_self.current_solution.cmax
            
            d_jobs = algo_self.operator_list_perturbation[algo_self.action]
            perturbation(
                algo_self.p_np,
                algo_self.new_solution,
                num_jobs_remove=d_jobs,
                local_search_partial=algo_self.local_search_destruction_partial_solution,  # ← thêm dòng này
                until_no_improvement=algo_self.until_no_improvement,
                tie_breaking=algo_self.tie_breaking_destruction_partial_solution
            )
            
            # 2) Exploitation via Local Search 
            local_search(
                algo_self.p_np, 
                algo_self.new_solution, 
                method=algo_self.main_local_search,
                ref_best=algo_self.best_solution if algo_self.ref_best else None,
                until_no_improvement=algo_self.until_no_improvement,
                tie_breaking=algo_self.tie_breaking_main_LS
            )
            
            # 3) Process evaluation metrics and accept/reject shifts
            acceptance(
                algo_self.current_solution,
                algo_self.new_solution,
                algo_self.best_solution,
                algo_self.temperature
            )
            
            algo_self.exe_time.append(time.time())
            algo_self.iterations += 1
            
            algo_self.best_fitness_list.append(algo_self.best_solution.cmax)
            algo_self.current_fitness_list.append(algo_self.current_solution.cmax)
            
            current_fit_during_episodes.append(algo_self.current_solution.cmax)
            best_fit_during_episodes.append(algo_self.best_solution.cmax)
            
        actual_eps = len(current_fit_during_episodes) - 1  # actual steps completed (may be < episode_size if time ran out)
        if actual_eps == 0:
            algo_self.epsilon_greedy *= algo_self.epsilon_greedy_decay
            continue
        mins_cur = [current_fit_during_episodes[0]]
        improvement_num = 0
        for imp in range(0, actual_eps):
            if current_fit_during_episodes[imp+1] < mins_cur[-1]:
                mins_cur.append(current_fit_during_episodes[imp+1])
        
        mins_best = [best_fit_during_episodes[0]]
        for imp in range(0, actual_eps):
            if best_fit_during_episodes[imp+1] < mins_best[-1]:
                improvement_num += 1
                mins_best.append(best_fit_during_episodes[imp+1])
        
        Diff_L = current_fit_during_episodes[0] - mins_cur[-1]
        Diff_G = best_fit_during_episodes[0] - mins_best[-1]
        
        DL = Diff_L / current_fit_during_episodes[0]
        DG = Diff_G / best_fit_during_episodes[0] 
        
        # Calculate rewards based on relative changes
        reward = 0.3 * max(DL, 0) + 0.7 * max(DG, 0)
        
        if improvement_num > 0:
            next_state = 1
            algo_self.Q_matrix[algo_self.state][algo_self.action] = (
                algo_self.Q_matrix[algo_self.state][algo_self.action] + 
                algo_self.alpha_learning * (
                    reward + -algo_self.gamma_learning * np.max(algo_self.Q_matrix[next_state]) - 
                    algo_self.Q_matrix[algo_self.state][algo_self.action]
                )
            )
            algo_self.state = next_state
        else:
            next_state = 0
            algo_self.Q_matrix[algo_self.state][algo_self.action] = (   # dùng old state
                algo_self.Q_matrix[algo_self.state][algo_self.action] +
                algo_self.alpha_learning * (
                    reward + algo_self.gamma_learning * np.max(algo_self.Q_matrix[next_state])
                    - algo_self.Q_matrix[algo_self.state][algo_self.action]
                )
            )
            algo_self.state = next_state  # update state SAU
        
        algo_self.epsilon_greedy *= algo_self.epsilon_greedy_decay
        # -algo_self.gamma_learning *= algo_self.epsilon_greedy_decay


def ig_individual(algo_self):
    """
    Runs Iterated Greedy strategy using a fixed perturbation sizing (index 0).
    """
    while datetime.now() < algo_self.time_limit and algo_self.iterations < algo_self.max_iteration:
        # Prepare targets for the upcoming mutation sequence
        algo_self.new_solution.perm = algo_self.current_solution.perm.copy()
        algo_self.new_solution.cmax = algo_self.current_solution.cmax
        
        # 1) Exploration via perturbation
        perturbation(
            algo_self.p_np,
            algo_self.new_solution,
            num_jobs_remove=algo_self.operator_list_perturbation[0],
            local_search_partial=algo_self.local_search_destruction_partial_solution,
            until_no_improvement=algo_self.until_no_improvement,
            tie_breaking=algo_self.tie_breaking_destruction_partial_solution
        )
        
        # 2) Exploitation via Local Search
        local_search(
            algo_self.p_np, 
            algo_self.new_solution, 
            method=algo_self.main_local_search,
            ref_best=algo_self.best_solution if algo_self.ref_best else None,
            until_no_improvement=algo_self.until_no_improvement,
            tie_breaking=algo_self.tie_breaking_main_LS
        )
        
        # 3) Acceptance Criteria
        acceptance(
            algo_self.current_solution,
            algo_self.new_solution,
            algo_self.best_solution,
            algo_self.temperature
        )
        
        algo_self.exe_time.append(time.time())
        algo_self.iterations += 1
        
        algo_self.best_fitness_list.append(algo_self.best_solution.cmax)
        algo_self.current_fitness_list.append(algo_self.current_solution.cmax)

            
def ig_random(algo_self):
    """
    Runs Iterated Greedy strategy using randomly selected destruction operators.
    """
    while datetime.now() < algo_self.time_limit and algo_self.iterations < algo_self.max_iteration:
        algo_self.new_solution.perm = algo_self.current_solution.perm.copy()
        algo_self.new_solution.cmax = algo_self.current_solution.cmax
        
        # 1) Exploration via perturbation
        indx = int(np.random.randint(0, len(algo_self.operator_list_perturbation)))
        perturbation(
            algo_self.p_np, 
            algo_self.new_solution, 
            num_jobs_remove=algo_self.operator_list_perturbation[indx],
            local_search_partial=algo_self.local_search_destruction_partial_solution,
            until_no_improvement=algo_self.until_no_improvement,
            tie_breaking=algo_self.tie_breaking_destruction_partial_solution
        )
        
        # 2) Exploitation via Local Search
        local_search(
            algo_self.p_np, 
            algo_self.new_solution, 
            method=algo_self.main_local_search,
            ref_best=algo_self.best_solution if algo_self.ref_best else None,
            until_no_improvement=algo_self.until_no_improvement,
            tie_breaking=algo_self.tie_breaking_main_LS
        )
        
        # 3) Acceptance Criteria
        acceptance(
            algo_self.current_solution,
            algo_self.new_solution,
            algo_self.best_solution,
            algo_self.temperature
        )
        
        algo_self.exe_time.append(time.time())
        algo_self.iterations += 1
        
        algo_self.best_fitness_list.append(algo_self.best_solution.cmax)
        algo_self.current_fitness_list.append(algo_self.current_solution.cmax)