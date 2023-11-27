from main import Simulation
import os
import csv

results_dir = "simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def aggregate_results(all_results):
    aggregated = {'waiting_times': [], 'system_times': [], 'utilization': []}
    for result in all_results:
        aggregated['waiting_times'].extend(result['waiting_times'])
        aggregated['system_times'].extend(result['system_times'])
        aggregated['utilization'].extend(result['utilization'])
    return aggregated

def run_simulation(n_servers, num_runs):
    all_results = []
    for _ in range(num_runs):
        simulation = Simulation(lam, mu, n_servers, priority, debug, runtime)
        results = simulation.run()
        all_results.append(results)
    return aggregate_results(all_results)

def save_results_to_csv(results, file_path, run_number):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['run_number', 'waiting_time', 'system_time', 'utilization']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for wt, st, ut in zip(results['waiting_times'], results['system_times'], results['utilization']):
            writer.writerow({
                'run_number': run_number,
                'waiting_time': wt,
                'system_time': st,
                'utilization': ut
            })

# Set parameters for the simulation
mu = 0.8
lam = 0.28
priority = True
debug = False
runtime = 1000
num_runs = 100

for n_servers in [1, 2, 4]:
    file_path = os.path.join(results_dir, f'results_n_{n_servers}.csv')

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['run_number', 'waiting_time', 'system_time', 'utilization'])
        writer.writeheader()

    for run_number in range(1, num_runs + 1):
        simulation = Simulation(lam, mu, n_servers, priority, debug, runtime)
        results = simulation.run()
        save_results_to_csv(results, file_path, run_number)

    print(f'Results for {n_servers} servers saved to {file_path}')
