import numpy as np
import cmath
import os
import matplotlib.pyplot as plt
import random
from matplotlib.colors import LinearSegmentedColormap
from numba import njit, prange

filepath = './'

if not os.path.exists(os.path.join(filepath, 'assets')):
    os.makedirs(os.path.join(filepath, 'assets'))
    print("'assets' folder has been created in the specified filepath.")
else:
    print("'assets' folder already exists in the specified filepath.")

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

x = np.linspace(-2, 0.47, 1000)
y = np.linspace(-1.12, 1.12, 1000)
values = np.ndarray((x.shape[0], y.shape[0]))

@njit(parallel=True)
def mandelbrot(x, y, matrix, max_iteration):
    for i in prange(matrix.shape[0]):
        for j in prange(matrix.shape[1]):
            matrix[i, j] = get_mandelbrot_set(x[i], y[j], max_iteration)
    return matrix

def visualize_mandelbrot(output):
    plt.matshow(output, cmap=cmap)
    plt.ylabel("Real Numbers")
    plt.xlabel("Imaginary Numbers")
    plt.colorbar()
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
        theta = random.uniform(0, 2 * np.pi)
        sample_x = (center_x + np.sqrt(radius) * np.cos(theta))
        sample_y = (center_y + np.sqrt(radius) * np.sin(theta))
        samples[i, 0] = sample_x
        samples[i, 1] = sample_y
    
    return samples


def hyper_cube_integration(a, b, N, s):
    dimensions = 2
    samples = np.random.uniform(size=(N_samples, dimensions))
    tiles = np.tile(np.arange(1, N_samples+1), (dimensions, 1))
    for i in range(tiles.shape[0]):
        np.random.shuffle(tiles[i, :])
    print(tiles)
    tiles = tiles.T
    samples = (tiles - samples) / N_samples
    samples = samples * (upper_bound - lower_bound) + lower_bound
    return samples

def orthogonal(lower_bound, upper_bound, N_samples):
    size = int(np.sqrt(N_samples))

    if size ** 2 != N_samples:
        raise ValueError("N must be a perfect square!")
    
    samples = np.empty((size, 2))
    
    x_bins = np.array(np.split(np.linspace(0, size, N_samples), size))
    y_bins = np.array(np.split(np.linspace(0, size, N_samples), size))
    available_rows = list(range(size))
    available_cols = list(range(size))

    for _ in range(size):
        i = np.random.choice(available_rows)
        j = np.random.choice(available_cols)
        norm_x = (x_bins[i][j] + np.random.uniform())/(size + 1)
        norm_y = (y_bins[j][i] + np.random.uniform())/(size + 1)
        samples[_, 0] = norm_x * ( upper_bound - lower_bound) + lower_bound
        samples[_, 1] = norm_y * ( upper_bound - lower_bound) + lower_bound

        available_rows.remove(i)
        available_cols.remove(j)
    
    return samples

def monte_carlo_integration(lower_bound, upper_bound, N_samples, N_iterations, shape, samples):
    # if you use orthogonal sampling, remember to input N_samples = sqrt(N_samples)
    accept = 0
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

def orthogonal_integration(a, b, N, s):
    x_values = []
    y_values = []
    x_bins = np.array(np.split(np.linspace(a, b, N), np.sqrt(N)))
    y_bins = np.array(np.split(np.linspace(a, b, N), np.sqrt(N)))

    for i in range(x_bins.shape[0]):
        np.random.shuffle(x_bins[i,:])
        np.random.shuffle(y_bins[i,:])

    for i in range(x_bins.shape[0]):
        for j in range(x_bins.shape[1]):
            x_values.append(x_bins[i].pop()  + np.random.uniform())
            y_values.append(x_bins[i].pop()  + np.random.uniform())
    return x_values, y_values 


# samples_sizes = [4, 5, 6, 7]
# for i in samples_sizes:
#     hyper = hyper_cube_integration(-1.5, 1, 10**i, 1000)
#     uniform = mc_integrate(-1.5, 1, 10**i, 1000)
#     print(f"Estimate standard uniform sampling: {uniform} \t latin hypercube sampling: {hyper} \t sample size: {10**i}")
x,y = orthogonal_integration(0,3,16,1)
plt.plot(x,y)
plt.show()

print("Area with Uniform Sampling over a Square: " + str(uniform_square_results))
print(f"Area with Uniform Sampling over a Circle: " + str(uniform_circle_results))
print(f"Area with Latin Hypercube Sampling over a Square: " + str(lhc_results))
print(f"Area with Orthogonal Sampling over a Square: " + str(orthogonal_results))

def plot_convergence(lower_bound, upper_bound, N_samples, N_iterations, sampling_methods_info, start_iter=1, x_max=None, y_max=None):
    plt.figure(figsize=(10, 6))

    reference_areas = {}
    for sampling_function, label, shape in sampling_methods_info:
        samples = sampling_function(lower_bound, upper_bound, N_samples)
        reference_areas[label] = monte_carlo_integration(lower_bound, upper_bound, N_samples, N_iterations, shape, samples)

    for (sampling_function, label, shape), color in zip(sampling_methods_info, normalized_palette.values()):
        iters = np.arange(start_iter, N_iterations + 1)
        errors = []

        for i in iters:
            samples = sampling_function(lower_bound, upper_bound, N_samples)
            estimated_area = monte_carlo_integration(lower_bound, upper_bound, N_samples, i, shape, samples)
            error = estimated_area - reference_areas[label]
            errors.append(error)
              # plt.plot(iters, errors, label=label)
        plt.scatter(iters, errors, color=color, label=label, s=5)

    plt.xlabel("j")
    plt.ylabel("A_j,s - A_i,s")
    plt.title("Convergence of Monte Carlo Estimates by Sampling Method")
    plt.legend()

    if x_max is not None:
        plt.xlim(0, x_max)
    if y_max is not None:
        plt.ylim(0, y_max)

    plt.savefig('./assets/convergence.png')

sampling_methods_info = [
    (uniform_square, 'Uniform Square', 'square'),
    (uniform_circle, 'Uniform Circle', 'circle'),
    (latin_hypercube, 'Latin Hypercube', 'square')
]
plot_convergence(-2, 2, 10000, 1000, sampling_methods_info, start_iter=3, x_max=200)
