from Engels_14947994_Sahrani_12661651_Siregar_1486305 import normalized_palette
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_sample_size_comparison():
    data = pd.read_csv('./assets/mandelbrot_sample_size_comparison.csv')
    mean_areas = data.groupby(['method', 'sample_size'])['area'].mean().reset_index()

    method_colors = {
        "uniform_square": normalized_palette["green"],
        "uniform_circle": normalized_palette["blue"],
        "latin_hypercube": normalized_palette["midnight"],
        "orthogonal": normalized_palette["crayola"]
    }

    plt.figure(figsize=(10, 6))
    for method in mean_areas['method'].unique():
        pdb.set_trace()
        method_data = mean_areas[mean_areas['method'] == method]
        plt.plot(method_data['sample_size'], method_data['area'],
                marker='o', label=method, color=method_colors[method])

    plt.xscale('log')
    plt.xlabel('Sample Size', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Estimated Area', fontsize=18)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.savefig('./assets/comparison_by_sample_size.png')

def plot_iterations_comparison():
    data = pd.read_csv('./assets/mandelbrot_iterations_comparison.csv')
    iteration_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    filtered_data = data[data['iterations'].isin(iteration_counts)]
    mean_areas = filtered_data.groupby('iterations')['area'].mean()
    std_error = filtered_data.groupby('iterations')['area'].std() / np.sqrt(filtered_data.groupby('iterations')['area'].count())

    plt.figure(figsize=(10, 6))
    plt.plot(mean_areas.index, mean_areas, marker='o', linestyle='-', color=normalized_palette["crayola"],)
    plt.fill_between(mean_areas.index, mean_areas - std_error, mean_areas + std_error, color=normalized_palette["blue"], alpha=0.5)

    plt.xlabel('Iterations', fontsize=18)
    plt.xticks(iteration_counts, fontsize=16)
    plt.ylabel('Estimated Area', fontsize=18)
    plt.yticks(fontsize=16)
    plt.savefig('./assets/comparison_by_iterations')
