import numpy as np
import pandas as pd
from scipy import stats
import cmath
import os
import matplotlib.pyplot as plt
import random
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


x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
values = np.ndarray((x.shape[0], y.shape[0]))


@njit(parallel=True)
def mandelbrot(x, y, matrix, max_iteration):
    for i in prange(matrix.shape[0]):
        for j in prange(matrix.shape[1]):
            matrix[i, j] = get_mandelbrot_set(x[i], y[j], max_iteration)
    return matrix


def visualize_mandelbrot(output):
    plt.matshow(output, extent=(np.min(x), np.max(x), np.min(y), np.max(y)), cmap=cmap, origin = "lower")
    plt.ylabel("Real Numbers")
    plt.xlabel("Imaginary Numbers")
    #plt.xticks(np.linspace(np.min(x), np.max(x), num=5))
    plt.savefig('./assets/mandelbrot.png')
    plt.close()

visualize_mandelbrot(mandelbrot(x, y, values, 1000))

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


samples_unif_square = uniform_square(-2, 2, 1000000)
samples_unif_circle = uniform_circle(-2, 2, 1000000)
samples_lhc = latin_hypercube(-2, 2, 1000000)
samples_ortho = orthogonal(-2, 2, 1000000)

uniform_square_results = monte_carlo_integration(
    -2, 2,  1000, "square", samples_unif_square)
uniform_circle_results = monte_carlo_integration(
    -2, 2,  1000, "circle", samples_unif_circle)
lhc_results = monte_carlo_integration(-2,
    2, 1000, "square", samples_lhc)
orthogonal_results = monte_carlo_integration(
    -2, 2, 1000, "square", samples_ortho)

print("Area with Uniform Sampling over a Square: " + str(uniform_square_results))
print(f"Area with Uniform Sampling over a Circle: " + str(uniform_circle_results))
print(f"Area with Latin Hypercube Sampling over a Square: " + str(lhc_results))
print(f"Area with Orthogonal Sampling over a Square: " + str(orthogonal_results))

#Experimenting with Sobol Sampling
from scipy.stats import qmc
sampler = qmc.Sobol(d=2)
upper_bound = 2
lower_bound = -2
samples_sobol = sampler.random_base2(m=10) * ( upper_bound - lower_bound) + lower_bound
print("Area with Sobol Sampling over a Square: " + str(monte_carlo_integration(-2, 2, 1024, 1000, "square", samples_sobol)))

if __name__ == "__main__":
    def plot_convergence(lower_bound, upper_bound, N_samples, N_iterations, sampling_methods_info, start_j=1, x_max=None, y_max=None):
        plt.figure(figsize=(10, 8))

        # Ai,s is the reference area with i = 300
        reference_areas = {}
        for sampling_function, label, shape in sampling_methods_info:
            samples = sampling_function(lower_bound, upper_bound, N_samples)
            reference_areas[label] = monte_carlo_integration(lower_bound, upper_bound, N_iterations, shape, samples)

        for (sampling_function, label, shape) in sampling_methods_info:
            match label:
                case 'Uniform Square':
                    color = colors[0]
                case 'Uniform Circle':
                    color = colors[1]
                case 'Latin Hypercube':
                    color = colors[2]
                case 'Orthogonal':
                    color = colors[-1]

            iters = np.arange(start_j, 100 + 1)
            errors = []
            # Aj,s is the estimated area with j = 100
            for j in iters:
                samples = sampling_function(lower_bound, upper_bound, N_samples)
                estimated_area = monte_carlo_integration(lower_bound, upper_bound, j, shape, samples)
                error = estimated_area - reference_areas[label]
                errors.append(error)

            plt.scatter(iters, errors, color=color, label=label, s=35)

        plt.xlabel("j", fontsize = 18)
        plt.xticks(fontsize = 16)
        plt.ylabel("A_j,s - A_i,s", fontsize = 18)
        plt.yticks(fontsize = 16)
        plt.title("i = 300, s = 10000", fontsize = 18)
        plt.legend(fontsize = 16)

        if x_max is not None:
            plt.xlim(0, x_max)
        if y_max is not None:
            plt.ylim(0, y_max)

        plt.savefig('./assets/convergence.png')


    sampling_methods_info = [
        (uniform_square, 'Uniform Square', 'square'),
        (uniform_circle, 'Uniform Circle', 'circle'),
        (latin_hypercube, 'Latin Hypercube', 'square'),
        (orthogonal, 'Orthogonal', 'square')
    ]

    print("Plotting convergence...")
    plot_convergence(-2, 2, 10000, 300, sampling_methods_info, start_j=5)
    print("Finished plotting, please check assets folder.")

    def confidence_intervals(filename, alpha):
        methods = ["Uniform Square", "Uniform Circle", "Latin Hypercube", "Orthogonal"]
        data = pd.read_csv(filename)
        
        uniform_square_data = data.groupby("method").get_group("uniform_square")
        uniform_circle_data = data.groupby("method").get_group("uniform_circle")
        latin_hypercube_data = data.groupby("method").get_group("latin_hypercube")
        orthogonal_data = data.groupby("method").get_group("orthogonal")
        
        mean_uniform_square = uniform_square_data["mean_area"].iloc[0]
        mean_uniform_circle = uniform_circle_data["mean_area"].iloc[0]
        mean_latin_hypercube = latin_hypercube_data["mean_area"].iloc[0]
        mean_orthogonal = orthogonal_data["mean_area"].iloc[0]
        
        means = [mean_uniform_square, mean_uniform_circle, mean_latin_hypercube, mean_orthogonal]
        
        std_uniform_square = np.std(uniform_square_data.loc[:,"area"])
        std_uniform_circle = np.std(uniform_circle_data.loc[:,"area"])
        std_latin_hypercube = np.std(latin_hypercube_data.loc[:,"area"])
        std_orthogonal = np.std(orthogonal_data.loc[:,"area"])
        
        stds = [std_uniform_square, std_uniform_circle, std_latin_hypercube, std_orthogonal]
        
        for i in range(len(stds)):
            standard_error = stds[i]/ 10
            df = 99
            conf_interval = stats.t.interval(1-alpha, df, means[i], scale=standard_error)
            print(f"{methods[i]} Confidence Interval: {conf_interval}")
            plt.errorbar(x=i, y=means[i], yerr=(conf_interval[1] - means[i]), fmt='o', label=methods[i])

        plt.xticks(range(len(means)), [methods[i] for i in range(len(means))])
        plt.ylabel('Area')
        plt.title('Confidence Intervals for Mandelbrot Set Area')
        plt.show()
        
        welch_result = pg.welch_anova(data=data, dv="area", between="method")
        print(f"Welch's ANOVA statistic: {welch_result['F'][0]}    p-value: {welch_result['p-unc'][0]}")
        
        posthoc_result = pg.pairwise_gameshowell(data=data, dv="area", between="method")
        print(posthoc_result)


    confidence_intervals("mandelbrot_estimations.csv", 0.01)