from main import Simulation
import os
import csv
import random
from configs import configs


results_dir = "simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def save_results_to_csv(results, file_path, run_number, n_server, kwargs):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['n_server', 'dist_wait', 'dist_serve', 'priority', 'preempt', 'waiting_time', 'system_time', 'utilization']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for wt, st, ut in zip(results['waiting_times'], results['system_times'], results['utilization']):
            writer.writerow({
                'n_server': n_server,
                'dist_wait': kwargs["dist_wait"].__name__,
                'dist_serve': kwargs["dist_serve"].__name__,
                'priority': kwargs["priority"],
                'preempt': kwargs["preempt"],
                'waiting_time': wt,
                'system_time': st,
                'utilization': ut
            })

file_path = os.path.join(results_dir, f'results.csv')

with open(file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['n_server', 'dist_wait', 'dist_serve', 'priority', 'preempt', 'waiting_time', 'system_time', 'utilization'])
    writer.writeheader()

num_runs = 100
for experiment in configs():
    print(f"Running simulations for {experiment}...")
    for n_servers in [1, 2, 4]:
        for run_number in range(1, num_runs + 1):
            print ("\033[A                                                        \033[A")
            print(f"Run {run_number}/{num_runs} for {n_servers} servers")
            experiment_config = configs()[experiment]['kwargs']
            experiment_config['lam'] *= n_servers
            random.seed(run_number)
            simulation = Simulation(**experiment_config, n_servers=n_servers)
            results = simulation.run()
            save_results_to_csv(results, file_path, run_number, n_servers, experiment_config)


print(f'Results for simulation saved to {file_path}')
