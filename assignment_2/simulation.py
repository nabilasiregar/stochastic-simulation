from main import Simulation
import os

results_dir = "simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Set parameters for the simulation
mu = 0.8
lam = 0.28
priority = True
debug = False
runtime = 1000

for n_servers in [1, 2, 4]:
    simulation = Simulation(lam, mu, n_servers, priority, debug, runtime)
    results = simulation.run()
    file_path = os.path.join(results_dir, f'results_n_{n_servers}.csv')
    simulation.save_simulation_to_csv(file_path)

    print(f'Results for {n_servers} servers saved to {file_path}')
