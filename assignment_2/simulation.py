from main import Simulation
import os
import csv
import random
from configs import configs
import matplotlib.pyplot as plt

results_dir = "simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def save_average_results_to_csv(average_results, file_path, run_number, n_server, kwargs):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['n_server', 'dist_wait', 'dist_serve', 'priority', 'preempt', 'avg_waiting_time', 'avg_system_time', 'avg_utilization']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'n_server': n_server,
            'dist_wait': kwargs["dist_wait"].__name__,
            'dist_serve': kwargs["dist_serve"].__name__,
            'priority': kwargs["priority"],
            'preempt': kwargs["preempt"],
            'avg_waiting_time': average_results['avg_waiting_time'],
            'avg_system_time': average_results['avg_system_time'],
            'avg_utilization': average_results['avg_utilization']
        })

file_path = os.path.join(results_dir, 'results.csv')

with open(file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['n_server', 'dist_wait', 'dist_serve', 'priority', 'preempt', 'avg_waiting_time', 'avg_system_time', 'avg_utilization'])
    writer.writeheader()

num_runs = 10
for experiment in configs():
    print(f"Running simulations for {experiment}...")
    for n_servers in [1, 2, 4]:
        for run_number in range(1, num_runs + 1):
            experiment_config = configs()[experiment]['kwargs']
            experiment_config['lam'] *= n_servers
            random.seed(run_number)
            simulation = Simulation(**experiment_config, n_servers=n_servers)
            results = simulation.run()

            # Calculate averages for each run
            avg_waiting_time = sum(results['waiting_times']) / len(results['waiting_times'])
            avg_system_time = sum(results['system_times']) / len(results['system_times'])
            avg_utilization = sum(results['utilization']) / len(results['utilization'])

            print("\033[A                                                                                             \033[A")
            print(f"Run {run_number}/{num_runs} for {n_servers} servers, {experiment}, avg wait {avg_waiting_time:.2f}, observations {len(results['waiting_times'])}")
            average_results = {
                'avg_waiting_time': avg_waiting_time,
                'avg_system_time': avg_system_time,
                'avg_utilization': avg_utilization
            }

            save_average_results_to_csv(average_results, file_path, run_number, n_servers, experiment_config)

print(f'Results for simulation saved to {file_path}')