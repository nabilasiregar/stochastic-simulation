from main import Simulation
import os
import csv
import random

results_dir = "simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def save_results_to_csv(results, file_path, run_number, n_server):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['n_server', 'dist_wait', 'dist_serve', 'priority', 'waiting_time', 'system_time', 'utilization']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for wt, st, ut in zip(results['waiting_times'], results['system_times'], results['utilization']):
            writer.writerow({
                'n_server': n_server,
                'dist_wait': dist_wait,
                'dist_serve': dist_serve,
                'priority': priority,
                'waiting_time': wt,
                'system_time': st,
                'utilization': ut
            })

file_path = os.path.join(results_dir, f'results.csv')

with open(file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['n_server', 'dist_wait', 'dist_serve', 'priority', 'waiting_time', 'system_time', 'utilization'])
    writer.writeheader()

# Set parameters for the simulation
mu = 0.8
lam = 0.28
priority = True
debug = False
runtime = 1000
num_runs = 100
dist_wait = random.expovariate
dist_serve = random.expovariate

for n_servers in [1, 2, 4]:
    print(f"Running simulations for {n_servers} servers...")
    for run_number in range(1, num_runs + 1):
        simulation = Simulation(lam, mu, dist_wait, dist_serve, n_servers, priority, debug, runtime)
        results = simulation.run()
        save_results_to_csv(results, file_path, run_number, n_servers)

print(f'Results for simulation saved to {file_path}')
