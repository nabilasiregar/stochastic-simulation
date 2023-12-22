"""This file contains methods to run different type of simulation for data collection"""
from map import *
from simulated_annealing import *
from map_config import *
import numpy as np
import pandas as pd
import sys

nodes = read_csv(MEDIUM_MAP) # adjust map size according to need
paths = add_paths(MEDIUM_OPT) # adjust map reference according to need
known_best_length = calculate_path_length(paths, nodes)
T = 1000
alpha = 0.999
stopping_T = 0.1
chain_length = 1000
temp_list_length = 1000
starting_path = np.random.permutation(range(1, len(nodes) + 1))
p0 = 0.5

# Number of runs
num_runs = 20

# Lists to store results
sim_annealing_results = []
fast_annealing_results = []
list_sim_annealing_results = []


def all_annealing_types():
    """To run regular simulated annealing, fast simulated annealing, and list-based simulated annealing all at once.
        Simulated annealing parameters are kept constant throughout simulation.
    """
    for run in range(num_runs):
        # Running regular simulated annealing
        best_path, best_length, iter, t_list, length_list = sim_annealing(nodes, T, alpha, stopping_T, chain_length, starting_path)
        sim_annealing_results.append({
                'method': "sim_annealing",
                'best_path': best_path,
                'best_length': best_length,
                'iterations': iter, 
                't_list': t_list, 
                'length_list': length_list
            })

        print(f"Run {run} for normal simulated annealing complete.")
        
        # Running fast simulated annealing
        best_path, best_length, iter, t_list, length_list = fast_annealing(nodes, T, alpha, stopping_T, chain_length, starting_path)
        fast_annealing_results.append({
                'method': "fast_annealing",  
                'best_path': best_path,
                'best_length': best_length,
                'iterations': iter, 
                't_list': t_list, 
                'length_list': length_list
            })

        print(f"Run {run} for fast simulated annealing complete.")

        #Running list-based simulated annealing
        temperature_list = get_temperature_list(nodes, temp_list_length, p0, starting_path)
        best_path, best_length, iter, t_list, length_list = sim_annealing_list(nodes, temp_list_length, chain_length, starting_path, temperature_list)
        list_sim_annealing_results.append({
                'method': "list_sim_annealing",  
                'best_path': best_path,
                'best_length': best_length,
                'iterations': iter, 
                't_list': t_list, 
                'length_list': length_list
            })
        
        print(f"Run {run} for list-based simulated annealing complete.")
        print()
        
    columns = ["method", "best_path", "best_length", "iterations", "t_list", "length_list"]
    results = pd.DataFrame(sim_annealing_results + fast_annealing_results + list_sim_annealing_results, columns=columns)
    results.to_csv("method_results_medium.csv", index=False)


def vary_cooling_factor():
    """To run regular simulated annealing with different cooling factors"""
    cooling_factor_list = [0.55, 0.75, 0.99]
    for run in range(num_runs):
        for cooling_factor in cooling_factor_list:
            print(f'Run {run} for cooling factor {cooling_factor}')
            best_path, best_length, iter, t_list, length_list = sim_annealing(nodes, T, cooling_factor, stopping_T, chain_length, starting_path)
            sim_annealing_results.append({
                    'cooling_factor': cooling_factor,
                    'best_length': best_length,
                    'best_path': best_path,
                    'iterations': iter, 
                    't_list': t_list, 
                    'length_list': length_list
                })
    
    columns = ["cooling_factor", "best_length", "best_path", "iterations", "t_list", "length_list"]
    results = pd.DataFrame(sim_annealing_results, columns=columns)
    results.to_csv("cooling_factor_results.csv", index=False)


def vary_chain_length():
    """To run regular simulated annealing with different markov chain's length"""
    chain_length_list = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    for run in range(num_runs):
        for markov_chain in chain_length_list:
            print(f'Run {run} with chain length: {markov_chain}')
            best_path, best_length, iter, t_list, length_list = sim_annealing(nodes, T, alpha, stopping_T, markov_chain, starting_path)
            sim_annealing_results.append({
                    'chain_length': markov_chain,
                    'best_length': best_length,
                    'best_path': best_path,
                    'iterations': iter, 
                    't_list': t_list, 
                    'length_list': length_list
                })
    
    columns = ["chain_length", "best_length", "best_path", "iterations", "t_list", "length_list"]
    results = pd.DataFrame(sim_annealing_results, columns=columns)
    results.to_csv("chain_length_results.csv", index=False)