from map import *
from simulated_annealing import *
from map_config import *
import numpy as np
import pandas as pd

nodes = read_csv(SMALL_MAP)
paths = add_paths(SMALL_OPT)
known_best_length = calculate_path_length(paths, nodes)
T = 1000
alpha = 0.999
stopping_T = 0.1
chain_length = 1000
starting_path = np.random.permutation(range(1, len(nodes) + 1))
p0 = 0.5
list_length = 120

# Number of runs
num_runs = 1

# Lists to store results
sim_annealing_results = []
fast_annealing_results = []
list_sim_annealing_results = []

for run in range(num_runs):
    # Running regular simulated annealing
    best_path, best_length, iter, t_list, length_list = sim_annealing(nodes, T, alpha, stopping_T, chain_length, starting_path)
    sim_annealing_results.append({
            'method': "sim_annealing",
            'best_path': best_path.tolist(),
            'best_length': best_length,
            'iterations': iter
        })
    print(sim_annealing_results)
    # Running fast simulated annealing
    best_path, best_length, iter, t_list, length_list = fast_annealing(nodes, T, alpha, stopping_T, chain_length, starting_path)
    fast_annealing_results.append({
            'method': "fast_annealing",  
            'best_path': best_path.tolist(),
            'best_length': best_length,
            'iterations': iter
        })
    print(fast_annealing_results)
    #Running list-based simulated annealing
    temperature_list = get_temperature_list(nodes, list_length, p0, starting_path)
    best_path, best_length, iter, length_list = sim_annealing_list(nodes, len(temperature_list), chain_length, starting_path, temperature_list)
    list_sim_annealing_results.append({
            'method': "list_sim_annealing",  
            'best_path': best_path.tolist(),
            'best_length': best_length,
            'iterations': iter
        })
    print(list_sim_annealing_results)
    
    
columns = ["Method", "Best_Path", "Best_Length", "Iterations"]
df1 = pd.DataFrame(sim_annealing_results, columns=columns)
print(df1)
df1.to_csv("results.csv", index=False)
