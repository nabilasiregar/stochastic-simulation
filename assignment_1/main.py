import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc
import random, os, cmath
import matplotlib.pyplot as plt
import pingouin as pg
from matplotlib.colors import LinearSegmentedColormap
from numba import njit, prange

filepath = './'

if not os.path.exists(os.path.join(filepath, 'assets')):
    os.makedirs(os.path.join(filepath, 'assets'))
    print("'assets' folder has been created to store plots and figures")

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

@njit
def get_mandelbrot_set(x, y, max_iteration):
    c = complex(x, y)
    z = 0
    iteration = 0
    while abs(z) < 2 and iteration < max_iteration:
        z = z**2 + c
        iteration += 1
    return iteration

@njit(parallel=True)
def mandelbrot(x, y, matrix, max_iteration):
    for i in prange(matrix.shape[0]):
        for j in prange(matrix.shape[1]):
            matrix[i, j] = get_mandelbrot_set(x[i], y[j], max_iteration)
    return matrix

# Sampling Techniques
def uniform_square(lower_bound, upper_bound, N_samples):
    samples = np.empty((N_samples, 2))
    for i in prange(N_samples):
        sample_x = random.random()
        sample_y = random.random()
        sample_x = sample_x * (upper_bound - lower_bound) + lower_bound
        sample_y = sample_y * (upper_bound - lower_bound) + lower_bound
        samples[i, 0] = sample_x
        samples[i, 1] = sample_y
    return samples

def uniform_circle(lower_bound, upper_bound, N_samples):
    center_x = (lower_bound + upper_bound) / 2
    center_y = (lower_bound + upper_bound) / 2
    diameter = upper_bound - lower_bound
    samples = np.empty((N_samples, 2))

    for i in prange(N_samples):
        radius = random.uniform(0, diameter/2)
        theta = random.uniform(0, 2*np.pi)
        sample_x = (center_x + np.sqrt(radius) * np.cos(theta))
        sample_y = (center_y + np.sqrt(radius) * np.sin(theta))
        samples[i, 0] = sample_x
        samples[i, 1] = sample_y

    return samples

def orthogonal_circle(lower_bound, upper_bound, N_samples):
    center_x = (lower_bound + upper_bound) / 2
    center_y = (lower_bound + upper_bound) / 2
    diameter = upper_bound - lower_bound
    radius = diameter / 2

    samples = orthogonal(0,1,N_samples)

    theta = 2 * np.pi * samples[:, 0]
    r = np.sqrt(radius * samples[:, 1])
    samples[:, 0] = center_x + (r * np.cos(theta))
    samples[:, 1] = center_y + (r * np.sin(theta))

    return samples

def latin_hypercube(lower_bound, upper_bound, N_samples):
    dimensions = 2
    samples = np.random.uniform(size=(N_samples, dimensions))
    tiles = np.tile(np.arange(1, N_samples+1), (dimensions, 1))
    for i in range(tiles.shape[0]):
        np.random.shuffle(tiles[i, :])
    tiles = tiles.T
    samples = (tiles - samples) / N_samples
    samples = samples * (upper_bound - lower_bound) + lower_bound
    return samples

def orthogonal(lower_bound, upper_bound, N_samples):
    size = int(np.sqrt(N_samples))

    if size ** 2 != N_samples:
        raise ValueError("N must be a perfect square!")

    samples = np.empty((N_samples, 2))
    step = (upper_bound - lower_bound) / N_samples

    x_bins = np.array(np.split(np.linspace(lower_bound, upper_bound, N_samples), size))
    y_bins = np.array(np.split(np.linspace(lower_bound, upper_bound, N_samples), size))
    available_rows = [list(range(size)) for _ in range(size)]
    available_cols = [list(range(size)) for _ in range(size)]

    r = 0
    for col in range(size):
        for row in range(size):
            i = np.random.choice(available_cols[col])
            j = np.random.choice(available_rows[row])
            norm_x = np.random.uniform(x_bins[col][i], x_bins[col][i] + step)
            norm_y = np.random.uniform(y_bins[row][j], x_bins[row][j] + step)

            samples[r, 0] = norm_x
            samples[r, 1] = norm_y

            available_cols[col].remove(i)
            available_rows[row].remove(j)
            r += 1

    return samples

def monte_carlo_integration(lower_bound, upper_bound, N_iterations, shape, samples):
    # if you use orthogonal sampling, remember to input N_samples = sqrt(N_samples)
    accept = 0
    N_samples = len(samples)
    for i in samples:
        result = get_mandelbrot_set(i[0], i[1], N_iterations)
        if result == N_iterations:
            accept += 1
    if shape == "square":
        return accept * (upper_bound - lower_bound) ** 2 / N_samples
    if shape == "circle":
        diameter = upper_bound - lower_bound
        circle_area = np.pi * (np.sqrt(diameter/2)) ** 2
        return accept * circle_area / N_samples

if __name__ == "__main__":

    def visualize_mandelbrot(output):
        plt.matshow(output, extent=(np.min(x), np.max(x), np.min(y), np.max(y)), cmap=cmap, origin = "lower")
        plt.ylabel("Real Numbers")
        plt.xlabel("Imaginary Numbers")
        plt.savefig('./assets/mandelbrot.png')
        plt.close()

    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    values = np.ndarray((x.shape[0], y.shape[0]))

    print("Visualizing mandelbrot set...")
    visualize_mandelbrot(mandelbrot(x, y, values, 1000))
    print("Saved! Please check assets folder.\n")

    def confidence_intervals(filename, alpha):
        methods = ["Uniform Square", "Uniform Circle", "Latin Hypercube", "Orthogonal Square", "Orthogonal Circle"]
        data = pd.read_csv(filename)
        
        uniform_square_data = data.groupby("method").get_group("uniform_square")
        uniform_circle_data = data.groupby("method").get_group("uniform_circle")
        latin_hypercube_data = data.groupby("method").get_group("latin_hypercube")
        orthogonal_square_data = data.groupby("method").get_group("orthogonal")
        orthogonal_circle_data = data.groupby("method").get_group("orthogonal_circle")
        
        mean_uniform_square = uniform_square_data["mean_area"].iloc[0]
        mean_uniform_circle = uniform_circle_data["mean_area"].iloc[0]
        mean_latin_hypercube = latin_hypercube_data["mean_area"].iloc[0]
        mean_orthogonal_square = orthogonal_square_data["mean_area"].iloc[0]
        mean_orthogonal_circle = orthogonal_circle_data["mean_area"].iloc[0]
        
        means = [mean_uniform_square, mean_uniform_circle, mean_latin_hypercube, mean_orthogonal_square, mean_orthogonal_circle]
        
        std_uniform_square = np.std(uniform_square_data.loc[:,"area"])
        std_uniform_circle = np.std(uniform_circle_data.loc[:,"area"])
        std_latin_hypercube = np.std(latin_hypercube_data.loc[:,"area"])
        std_orthogonal_square = np.std(orthogonal_square_data.loc[:,"area"])
        std_orthogonal_circle = np.std(orthogonal_circle_data.loc[:,"area"])
        
        stds = [std_uniform_square, std_uniform_circle, std_latin_hypercube, std_orthogonal_square, std_orthogonal_circle]
        
        plt.figure(figsize=(10, 6))
        
        for i in range(len(stds)):
            standard_error = stds[i]/ 10
            df = 99
            conf_interval = stats.t.interval(1-alpha, df, means[i], scale=standard_error)
            print(f"{methods[i]} Confidence Interval: {conf_interval}")
            plt.errorbar(x=i, y=means[i], yerr=(conf_interval[1] - means[i]), fmt='o', label=methods[i], color = colors[i])

        plt.xticks(range(len(means)), [methods[i] for i in range(len(means))], fontsize = 12)
        plt.ylabel('Estimated Area', fontsize = 16)
        plt.title('Confidence Intervals for Mandelbrot Set Area', fontsize = 16)
        plt.savefig('./assets/confidence_intervals.png')
        plt.close()
        
        welch_result = pg.welch_anova(data=data, dv="area", between="method")
        print(f"Welch's ANOVA statistic: {welch_result['F'][0]}    p-value: {welch_result['p-unc'][0]}")
        
        posthoc_result = pg.pairwise_gameshowell(data=data, dv="area", between="method")
        print(posthoc_result)

    print("Calculating confidence intervals...")
    confidence_intervals("./data/mandelbrot_estimations.csv", 0.01)