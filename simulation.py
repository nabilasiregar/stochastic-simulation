from Engels_14947994_Sahrani_12661651_Siregar_1486305 import uniform_circle, uniform_square, latin_hypercube, orthogonal, monte_carlo_integration
import pandas as pd
import csv

num_runs = 10
num_samples = 1000000

def run_simulation(method, shape, num_runs, num_samples):
    results = []
    for run in range(num_runs):
        samples = method(-2, 2, num_samples)
        area = monte_carlo_integration(-2, 2, 1000, shape, samples)
        results.append({'method': method.__name__, 'run': run + 1, 'samples': num_samples, 'area': area})
    return results

results = []
results.extend(run_simulation(uniform_square, "square", num_runs, num_samples))
results.extend(run_simulation(uniform_circle, "circle", num_runs, num_samples))
results.extend(run_simulation(latin_hypercube, "square", num_runs, num_samples))
results.extend(run_simulation(orthogonal, "square", num_runs, num_samples))

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
df.to_csv(f"./assets/mandelbrot_estimations_samples:{num_samples}_runs:{num_runs}.csv", index=False)