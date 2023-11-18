import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

palette = {
    "green": (139, 191, 159),
    "blue": (131, 188, 255),
    "midnight": (18, 69, 89),
    "violet": (89, 52, 79),
    "crayola": (238, 32, 77)
}

normalized_palette = {key: tuple(val / 255.0 for val in value)
                      for key, value in palette.items()}

colors = list(normalized_palette.values())
color_stops = [i / (len(colors) - 1) for i in range(len(colors))]
cmap = LinearSegmentedColormap.from_list(
    "custom_gradient", list(zip(color_stops, colors)))


methods = ["Uniform Square", "Uniform Circle", "Latin Hypercube", "Orthogonal Square", "Orthogonal Circle"]
usqu_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_usqu.csv")
ucir_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_ucir.csv")
lhs_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_lhcs.csv")
osqu_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_osqu.csv")
ocir_csv = pd.read_csv("./assets/mandelbrot_iterations_comparison_ocir.csv")

iteration_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

mean_uniform_square_data = usqu_csv.groupby("iterations")["area"].mean()
mean_uniform_circle_data = ucir_csv.groupby("iterations")["area"].mean()
mean_latin_hypercube_data = lhs_csv.groupby("iterations")["area"].mean()
mean_orth_square_data = osqu_csv.groupby("iterations")["area"].mean()
mean_orth_circle_data = ocir_csv.groupby("iterations")["area"].mean()

std_uniform_square_data = usqu_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())
std_uniform_circle_data = ucir_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())
std_latin_hypercube_data = lhs_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())
std_orth_square_data = osqu_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())
std_orth_circle_data = ocir_csv.groupby("iterations")["area"].std() / np.sqrt(usqu_csv.groupby("iterations")["area"].count())

plt.figure(figsize=(10, 6))
plt.plot(mean_uniform_square_data.index, np.abs(mean_uniform_square_data[1000] - mean_uniform_square_data), marker='o', linestyle='-', color=normalized_palette["green"])
plt.plot(mean_uniform_circle_data.index, np.abs(mean_uniform_circle_data[1000] - mean_uniform_circle_data), marker='o', linestyle='-', color=normalized_palette["blue"])
plt.plot(mean_latin_hypercube_data.index, np.abs(mean_latin_hypercube_data[1000] - mean_latin_hypercube_data), marker='o', linestyle='-', color=normalized_palette["midnight"])
plt.plot(mean_orth_square_data.index, np.abs(mean_orth_square_data[1000] - mean_orth_square_data), marker='o', linestyle='-', color=normalized_palette["violet"])
plt.plot(mean_orth_circle_data.index, np.abs(mean_orth_circle_data[1000] - mean_orth_circle_data), marker='o', linestyle='-', color=normalized_palette["crayola"])
plt.fill_between(mean_uniform_square_data.index,np.abs(mean_uniform_square_data[1000] - mean_uniform_square_data) - std_uniform_square_data, np.abs(mean_uniform_square_data[1000] - mean_uniform_square_data) + std_uniform_square_data, color=normalized_palette["green"],  alpha=0.5)
plt.fill_between(mean_uniform_circle_data.index, np.abs(mean_uniform_circle_data[1000] - mean_uniform_circle_data) - std_uniform_circle_data, np.abs(mean_uniform_circle_data[1000] - mean_uniform_circle_data) + std_uniform_circle_data, color=normalized_palette["blue"],  alpha=0.5)
plt.fill_between(mean_latin_hypercube_data.index, np.abs(mean_latin_hypercube_data[1000] - mean_latin_hypercube_data) - std_latin_hypercube_data, np.abs(mean_latin_hypercube_data[1000] - mean_latin_hypercube_data) + std_latin_hypercube_data, color=normalized_palette["midnight"], alpha=0.5)
plt.fill_between(mean_orth_square_data.index, np.abs(mean_orth_square_data[1000] - mean_orth_square_data) - std_orth_square_data, np.abs(mean_orth_square_data[1000] - mean_orth_square_data) + std_orth_square_data, color=normalized_palette["violet"],  alpha=0.5)
plt.fill_between(mean_orth_circle_data.index, np.abs(mean_orth_circle_data[1000] - mean_orth_circle_data) - std_orth_circle_data, np.abs(mean_orth_circle_data[1000] - mean_orth_circle_data) + std_orth_circle_data, color=normalized_palette["crayola"],  alpha=0.5)

plt.xlabel('Iterations', fontsize=16)
plt.legend(methods, fontsize = 16)
plt.xticks(iteration_counts, fontsize=16)
plt.ylabel('|$A_{iteration,s} - A_{1000,s}$|', fontsize=16)
plt.yticks(fontsize=16)
plt.title("Area Estimation Error for Varying Sampling Methods - s = 10201", fontsize = 16)
plt.show()