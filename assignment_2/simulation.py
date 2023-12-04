import csv
import os
import random
import sys
from configs import configs
from main import Simulation

RESULTS_DIR = "simulation_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

NUM_RUNS = 20

def save_average_results_to_csv(average_results, file_path, n_server, kwargs):
    """
     A function to save experiment results from the simulation to a csv file

     Input parameters
     average_results: averaged simulation results from each run
     file_path: to specify the csv filename and location
     n_server: the number of servers
     kwargs: arguments from configs file
    """
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
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

def save_results_to_csv_with_rho(results, file_path, n_server, kwargs, rho):
    """
     A function to save experiment results from the simulation to a csv file with extra rho column appended

     Input parameters
     average_results: averaged simulation results from each run
     file_path: to specify the csv filename and location
     n_server: the number of servers
     kwargs: arguments from configs file
     rho: analytical rho based on mu, lambda, and number of servers
    """
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['n_server', 'dist_wait', 'dist_serve', 'priority', 'preempt', 'waiting_time', 'system_time', 'utilization', 'rho']
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
                'utilization': ut,
                'rho': rho
            })

    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['n_server', 'dist_wait', 'dist_serve', 'priority', 'preempt', 'waiting_time', 'system_time', 'utilization', 'rho'])
        writer.writeheader()

def main_simulation():
    """The main simulation function to run the experiment"""
    file_path = os.path.join(RESULTS_DIR, 'results.csv')
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['n_server', 'dist_wait', 'dist_serve', 'priority', 'preempt', 'avg_waiting_time', 'avg_system_time', 'avg_utilization'])
        writer.writeheader()

    for experiment in configs():
        print(f"Running simulations for {experiment}...")
        for n_servers in [1, 2, 4]:
            for run_number in range(1, NUM_RUNS + 1):
                experiment_config = configs()[experiment]['kwargs']
                experiment_config['lam'] *= n_servers
                random.seed(run_number) # Set the seed for reproducibility
                simulation = Simulation(**experiment_config, n_servers=n_servers)
                results = simulation.run()
                exclude = int(0.3 * len(results['waiting_times']))
                # Calculate averages for each run
                avg_waiting_time = sum(results['waiting_times'][exclude:]) / (len(results['waiting_times']) - exclude)
                avg_system_time = sum(results['system_times'][exclude:]) / (len(results['system_times']) - exclude)
                avg_utilization = sum(results['utilization'][exclude:]) / (len(results['utilization']) - exclude)

                print("\033[A                                                                                             \033[A")
                print(f"Run {run_number}/{NUM_RUNS} for {n_servers} servers, {experiment}, avg wait {avg_waiting_time:.2f}, observations {len(results['waiting_times'])}")
                average_results = {
                    'avg_waiting_time': avg_waiting_time,
                    'avg_system_time': avg_system_time,
                    'avg_utilization': avg_utilization
                }

                save_average_results_to_csv(average_results, file_path, n_servers, experiment_config)
    print(f'Results for simulation saved to {file_path}')

def simulate_with_different_arrival_rates(lambda_values, server_counts):
    """Simulation function to see how the number of measurements (e.g. varying the arrival rate) depend on rho"""
    for lam in lambda_values:
        for n_servers in server_counts:
            # Only running M/M/n queue type
            experiment_name = 'experiment_1'
            experiment_config = configs()[experiment_name]['kwargs']
            print(f"Running simulations for {experiment_name} with lambda={lam} and {n_servers} servers...")

            for run_number in range(1, NUM_RUNS + 1):
                print(f"Run {run_number}/{NUM_RUNS} for {n_servers} servers")

                # Adjust lambda based on the number of servers
                experiment_config['lam'] = lam * n_servers

                # Calculate rho for saving in CSV
                rho = experiment_config['lam'] / n_servers

                # Set the seed for reproducibility
                random.seed(run_number)

                # Initializing and running the simulation
                simulation = Simulation(**experiment_config, n_servers=n_servers)
                results = simulation.run()

                file_path = os.path.join(RESULTS_DIR, 'results_with_rho.csv')
                save_results_to_csv_with_rho(results, file_path, n_servers, experiment_config, rho)
    print(f'Results for simulation saved to {file_path}')

def run_simulation(simulation_type):
    """Function to switch between simulation types from terminal"""
    if simulation_type == "general":
        main_simulation()
    elif simulation_type == "test system load":
        # To test the system load (rho) we need to vary the arrival rate
        lambda_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        server_counts = [1, 2, 4]
        simulate_with_different_arrival_rates(lambda_values, server_counts)
    else:
        print("Invalid simulation type")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sim_type = sys.argv[1]
    else:
        sim_type = input("Enter the simulation type (general or test system load): ")
    run_simulation(sim_type)
