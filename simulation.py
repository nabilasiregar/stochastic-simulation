from Engels_14947994_Sahrani_12661651_Siregar_1486305 import uniform_circle, uniform_square, orthogonal_circle, latin_hypercube, orthogonal, monte_carlo_integration
import pandas as pd
import sys

num_runs = 100
num_samples = 1000000
iteration_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
sample_sizes = [100, 10000, 1000000]

def run_simulation_all(method, shape, num_runs, num_samples):
    results = []
    for run in range(num_runs):
        samples = method(-2, 2, num_samples)
        area = monte_carlo_integration(-2, 2, 1000, shape, samples)
        results.append({'method': method.__name__, 'run': run + 1, 'samples': num_samples, 'area': area})
    return results

def run_simulation_for_sample_size(method, shape, num_runs, sample_size):
    results = []
    for run in range(num_runs):
        samples = method(-2, 2, sample_size) 
        area = monte_carlo_integration(-2, 2, 1000, shape, samples)
        results.append({
            'method': method.__name__,
            'sample_size': sample_size,
            'run': run + 1,
            'area': area
        })
    return results

def run_simulation_for_iterations(method, shape, N_iteration, num_samples):
    results = []
    for run in range(num_runs):
        samples = method(-2, 2, num_samples)
        area = monte_carlo_integration(-2, 2, N_iteration, shape, samples)
        results.append({
            'method': method.__name__,
            'num_samples': num_samples,
            'run': run + 1,
            'iterations': N_iteration,
            'area': area
        })
    return results

def run_simulation(simulation_type):
    print(f"Running simulation for {simulation_type}")
    if simulation_type == "all":
        results = []
        results.extend(run_simulation_for_sample_size(orthogonal_circle, "circle", num_runs, sample))
        results.extend(run_simulation_all(uniform_square, "square", num_runs, num_samples))
        results.extend(run_simulation_all(uniform_circle, "circle", num_runs, num_samples))
        results.extend(run_simulation_all(latin_hypercube, "square", num_runs, num_samples))
        results.extend(run_simulation_all(orthogonal, "square", num_runs, num_samples))

        df = pd.DataFrame(results)
        means = df.groupby('method')['area'].mean().reset_index()

        # Adding mean values to the results
        for index, row in means.iterrows():
            method = row['method']
            mean_area = row['area']
            for result in results:
                if result['method'] == method:
                    result['mean_area'] = mean_area

        df = pd.DataFrame(results)
        df.to_csv(f"./data/mandelbrot_estimations.csv", index=False)
    elif simulation_type == "sample_size":
        results = []
        for sample in sample_sizes:
            results.extend(run_simulation_for_sample_size(orthogonal_circle, "circle", num_runs, sample))
            results.extend(run_simulation_for_sample_size(uniform_square, "square", num_runs, sample))
            results.extend(run_simulation_for_sample_size(uniform_circle, "circle", num_runs, sample))
            results.extend(run_simulation_for_sample_size(latin_hypercube, "square", num_runs, sample))
            results.extend(run_simulation_for_sample_size(orthogonal, "square", num_runs, sample))

        df = pd.DataFrame(results)
        df.to_csv(f"./assets/mandelbrot_sample_size_comparison.csv", index=False)
    elif simulation_type == "iterations":
        results = []
        for iteration in iteration_values:
            results.extend(run_simulation_for_iterations(uniform_square, "square", iteration, num_samples))

        df = pd.DataFrame(results)
        df.to_csv(f"./assets/mandelbrot_iterations_comparison.csv", index=False)
    else:
        print("Invalid simulation type")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sim_type = sys.argv[1]
    else:
        sim_type = input("Enter the simulation type (all, sample_size, iterations): ")
    run_simulation(sim_type)