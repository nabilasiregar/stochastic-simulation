from Engels_14947994_Sahrani_12661651_Siregar_1486305 import normalized_palette
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# def plot_sample_size_comparison():
#     data = pd.read_csv('./data/mandelbrot_sample_size_comparison.csv')
#     mean_areas = data.groupby(['method', 'sample_size'])['area'].mean().reset_index()

#     method_colors = {
#         "uniform_square": normalized_palette["green"],
#         "uniform_circle": normalized_palette["blue"],
#         "latin_hypercube": normalized_palette["midnight"],
#         "orthogonal": normalized_palette["crayola"]
#     }

#     plt.figure(figsize=(10, 6))
#     for method in mean_areas['method'].unique():
#         method_data = mean_areas[mean_areas['method'] == method]
#         plt.plot(method_data['sample_size'], method_data['area'],
#                 marker='o', label=method, color=method_colors[method])

#     plt.xscale('log')
#     plt.xlabel('Sample Size', fontsize=18)
#     plt.xticks(fontsize=16)
#     plt.ylabel('Estimated Area', fontsize=18)
#     plt.yticks(fontsize=16)
#     plt.legend(fontsize=18)
#     plt.savefig('./assets/comparison_by_sample_size.png')

# def plot_iterations_comparison():
#     data = pd.read_csv('./data/mandelbrot_iterations_comparison.csv')
#     iteration_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#     filtered_data = data[data['iterations'].isin(iteration_counts)]
#     mean_areas = filtered_data.groupby('iterations')['area'].mean()
#     std_error = filtered_data.groupby('iterations')['area'].std() / np.sqrt(filtered_data.groupby('iterations')['area'].count())

#     plt.figure(figsize=(10, 6))
#     plt.plot(mean_areas.index, mean_areas, marker='o', linestyle='-', color=normalized_palette["crayola"],)
#     plt.fill_between(mean_areas.index, mean_areas - std_error, mean_areas + std_error, color=normalized_palette["blue"], alpha=0.5)

#     plt.xlabel('Iterations', fontsize=18)
#     plt.xticks(iteration_counts, fontsize=16)
#     plt.ylabel('Estimated Area', fontsize=18)
#     plt.yticks(fontsize=16)
#     plt.savefig('./assets/comparison_by_iterations')
    
# def plot_iterations_comparisons_all_methods(): 
#     methods = ["Uniform Square", "Uniform Circle", "Latin Hypercube", "Orthogonal Square", "Orthogonal Circle"]
#     usqu_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_usqu.csv")
#     ucir_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_ucir.csv")
#     lhs_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_lhcs.csv")
#     osqu_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_osqu.csv")
#     ocir_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_ocir.csv")

#     iteration_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

#     mean_uniform_square_data = usqu_csv.groupby("iterations")["area"].mean()
#     mean_uniform_circle_data = ucir_csv.groupby("iterations")["area"].mean()
#     mean_latin_hypercube_data = lhs_csv.groupby("iterations")["area"].mean()
#     mean_orth_square_data = osqu_csv.groupby("iterations")["area"].mean()
#     mean_orth_circle_data = ocir_csv.groupby("iterations")["area"].mean()

#     std_uniform_square_data = usqu_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())
#     std_uniform_circle_data = ucir_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())
#     std_latin_hypercube_data = lhs_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())
#     std_orth_square_data = osqu_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())
#     std_orth_circle_data = ocir_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())

#     plt.figure(figsize=(10, 6))
#     plt.plot(mean_uniform_square_data.index, np.abs(mean_uniform_square_data[1000] - mean_uniform_square_data), marker='o', linestyle='-', color=normalized_palette["green"])
#     plt.plot(mean_uniform_circle_data.index, np.abs(mean_uniform_circle_data[1000] - mean_uniform_circle_data), marker='o', linestyle='-', color=normalized_palette["blue"])
#     plt.plot(mean_latin_hypercube_data.index, np.abs(mean_latin_hypercube_data[1000] - mean_latin_hypercube_data), marker='o', linestyle='-', color=normalized_palette["midnight"])
#     plt.plot(mean_orth_square_data.index, np.abs(mean_orth_square_data[1000] - mean_orth_square_data), marker='o', linestyle='-', color=normalized_palette["violet"])
#     plt.plot(mean_orth_circle_data.index, np.abs(mean_orth_circle_data[1000] - mean_orth_circle_data), marker='o', linestyle='-', color=normalized_palette["crayola"])
#     plt.fill_between(mean_uniform_square_data.index,np.abs(mean_uniform_square_data[1000] - mean_uniform_square_data) - std_uniform_square_data, np.abs(mean_uniform_square_data[1000] - mean_uniform_square_data) + std_uniform_square_data, color=normalized_palette["green"],  alpha=0.5)
#     plt.fill_between(mean_uniform_circle_data.index, np.abs(mean_uniform_circle_data[1000] - mean_uniform_circle_data) - std_uniform_circle_data, np.abs(mean_uniform_circle_data[1000] - mean_uniform_circle_data) + std_uniform_circle_data, color=normalized_palette["blue"],  alpha=0.5)
#     plt.fill_between(mean_latin_hypercube_data.index, np.abs(mean_latin_hypercube_data[1000] - mean_latin_hypercube_data) - std_latin_hypercube_data, np.abs(mean_latin_hypercube_data[1000] - mean_latin_hypercube_data) + std_latin_hypercube_data, color=normalized_palette["midnight"], alpha=0.5)
#     plt.fill_between(mean_orth_square_data.index, np.abs(mean_orth_square_data[1000] - mean_orth_square_data) - std_orth_square_data, np.abs(mean_orth_square_data[1000] - mean_orth_square_data) + std_orth_square_data, color=normalized_palette["violet"],  alpha=0.5)
#     plt.fill_between(mean_orth_circle_data.index, np.abs(mean_orth_circle_data[1000] - mean_orth_circle_data) - std_orth_circle_data, np.abs(mean_orth_circle_data[1000] - mean_orth_circle_data) + std_orth_circle_data, color=normalized_palette["crayola"],  alpha=0.5)

#     plt.xlabel('Iterations', fontsize=16)
#     plt.legend(methods, fontsize = 16)
#     plt.xticks(iteration_counts, fontsize=16)
#     plt.ylabel('|$A_{iteration,s} - A_{1000,s}$|', fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.title("Area Estimation Differences for Varying Sampling Methods - s = 10201", fontsize = 16)
#     plt.show()

def plot_variances():
    methods = ["Uniform Square", "Uniform Circle", "Latin Hypercube", "Orthogonal Square", "Orthogonal Circle"]
    usqu_csv = pd.read_csv("./data/mandelbrot_iterations_comparison_usqu.csv")
    ucir_csv = pd.read_csv("./data/mandelbrot_iterations_comparison_ucir.csv")
    lhs_csv = pd.read_csv("./data/mandelbrot_iterations_comparison_lhcs.csv")
    osqu_csv = pd.read_csv("./data/mandelbrot_iterations_comparison_osqu.csv")
    ocir_csv = pd.read_csv("./data/mandelbrot_iterations_comparison_ocir.csv")
    
    var_uniform_square_data = usqu_csv.groupby("iterations")["area"].var()
    var_uniform_circle_data = ucir_csv.groupby("iterations")["area"].var()
    var_latin_hypercube_data = lhs_csv.groupby("iterations")["area"].var()
    var_orth_square_data = osqu_csv.groupby("iterations")["area"].var()
    var_orth_circle_data = ocir_csv.groupby("iterations")["area"].var()
    
    plt.figure(figsize=(10, 6))
    plt.plot(var_uniform_square_data.index, var_uniform_square_data, marker='o', linestyle='-', color=normalized_palette["green"])
    plt.plot(var_uniform_circle_data.index, var_uniform_circle_data, marker='o', linestyle='-', color=normalized_palette["blue"])
    plt.plot(var_latin_hypercube_data.index, var_latin_hypercube_data, marker='o', linestyle='-', color=normalized_palette["midnight"])
    plt.plot(var_orth_square_data.index, var_orth_square_data, marker='o', linestyle='-', color=normalized_palette["violet"])
    plt.plot(var_orth_circle_data.index, var_orth_circle_data, marker='o', linestyle='-', color=normalized_palette["crayola"])
    
    plt.legend(methods, fontsize = 14, bbox_to_anchor=(1.01, 1), loc= "upper left", borderaxespad=0)
    plt.xlabel("Iterations", fontsize = 16)
    plt.ylabel("Variance", fontsize = 16)
    plt.title("Variance between 100 Runs for Each Sampling Method", fontsize = 16)
    plt.savefig('./assets/variances', bbox_inches='tight')

# def choose_plot(plot_type):
#     if plot_type == 'sample_size':
#         plot_sample_size_comparison()
#     elif plot_type == 'iterations':
#         plot_iterations_comparison()
#     elif plot_type == "all_iterations":
#         plot_iterations_comparisons_all_methods()
#     elif plot_type == "variances":
#         plot_variances()
#     else:
#         print('Invalid plot type input')

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         plot_type = sys.argv[1]
#     else:
#         plot_type = input("Enter the simulation result you want to plot (sample_size or iterations): ")
#     choose_plot(plot_type)

plot_variances()