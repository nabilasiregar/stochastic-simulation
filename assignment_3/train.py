from map_config import SMALL_MAP, MEDIUM_MAP, LARGE_MAP
from map import *
from simulated_annealing import sim_annealing
import csv
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import optuna
import os

def save_to_csv(trial_number, initial_temperature, cooling_factor, chain_length, best_length, iterations, csv_filepath):
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, 'a', newline='') as csvfile:
        fieldnames = ['trial', 'initial_temperature', 'cooling_factor', 'chain_length', 'best_length', 'iterations']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'trial': trial_number,
            'initial_temperature': initial_temperature,
            'cooling_factor': cooling_factor,
            'chain_length': chain_length,
            'best_length': best_length,
            'iterations': iterations
        })

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.to_yaml(cfg)

    def objective(trial):
        map_file = {
                    'small': SMALL_MAP,
                    'medium': MEDIUM_MAP,
                    'large': LARGE_MAP
                }.get(cfg.map_size, MEDIUM_MAP) # Default to MEDIUM_MAP if an unknown map size is specified
        
        initial_temperature = trial.suggest_int("initial_temperature", 10, 1000)
        cooling_factor = trial.suggest_float("cooling_factor", 0.80, 0.99)
        chain_length = trial.suggest_int("chain_length", 1000, 20000)

        total_best_length = []
        original_cwd = hydra.utils.get_original_cwd()
        map_name = os.path.join(original_cwd, map_file)
        nodes = read_csv(map_name)
        csv_filepath = os.path.join(original_cwd, './multirun/train_results.csv')

        for _ in range(10):  # Run the algorithm 10 times for each trial
            starting_path = np.random.permutation(range(1, len(nodes) + 1))
            _, best_length, iters, _, _ = sim_annealing(
                nodes, initial_temperature, cooling_factor, cfg.stopping_temperature, chain_length, starting_path)       
            save_to_csv(trial.number, initial_temperature, cooling_factor, chain_length, best_length, iters, csv_filepath)
            total_best_length.append(best_length)

        average_best_length = np.average(total_best_length)

        return average_best_length

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

if __name__ == "__main__":
    main()