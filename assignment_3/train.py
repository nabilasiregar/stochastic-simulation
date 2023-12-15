from omegaconf import DictConfig, OmegaConf
from map_config import *
from map import *
from simulated_annealing import sim_annealing
import hydra
import numpy as np
import os
import csv

def save_to_csv(initial_temperature, cooling_factor, best_path, best_length, iterations, csv_filepath):
    # Check if the file exists. If not, create it and write the header
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, 'a', newline='') as csvfile:
        fieldnames = ['initial_temperature', 'cooling_factor', 'best_path', 'best_length', 'iterations']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'initial_temperature': initial_temperature,
            'cooling_factor': cooling_factor,
            'best_path': best_path,
            'best_length': best_length,
            'iterations': iterations
        })

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.to_yaml(cfg)
    nodes = read_csv(SMALL_MAP)
    paths = add_paths(SMALL_OPT)

    # Define parameters for simulated annealing
    initial_temperature = 1000.0
    cooling_factor = 0.995
    stopping_temperature = 0.1
    stopping_iterations = 1000

    np.random.shuffle(paths.copy())

    # Run Simulated Annealing with parameters from Hydra/Optuna
    best_path, best_length, iterations, t_list, length_list = sim_annealing(
        nodes, initial_temperature, cooling_factor, stopping_temperature, stopping_iterations, np.random.permutation(range(1,len(nodes)+1)))
    
    # Run Simulated Annealing with parameters from Hydra/Optuna
    best_path, best_length, iterations, temp_list, length_list = sim_annealing(
        nodes, cfg.initial_temperature, cfg.cooling_factor, cfg.stopping_temperature, cfg.stopping_iterations, np.random.permutation(range(1,len(nodes)+1)))
      
    original_cwd = hydra.utils.get_original_cwd()
    csv_filepath = os.path.join(original_cwd, './multirun/simulation_results.csv')
    save_to_csv(cfg.initial_temperature, cfg.cooling_factor, best_path, best_length, iterations, csv_filepath)

    return best_length

if __name__ == "__main__":
    main()