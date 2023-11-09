import numpy as np
import cmath
import matplotlib.pyplot as plt
import random
from matplotlib.colors import LinearSegmentedColormap
from numba import njit, prange
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
def mandelbrot(x, y, max_iteration):
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
def get_mandelbrot(x, y, matrix, max_iteration):
    for i in prange(matrix.shape[0]):
        for j in prange(matrix.shape[1]):
            matrix[i, j] = mandelbrot(x[i], y[j], max_iteration)
    return matrix


# max_iterations = 1000
# output = get_mandelbrot(x, y, values, max_iterations)
# plt.matshow(output, cmap=cmap)
# plt.ylabel("Real Numbers")
# plt.xlabel("Imaginary Numbers")
# plt.colorbar()
# plt.show()

@njit(parallel=True)
def mc_integrate(lower_bound, upper_bound, N_samples, num_of_iterations):
    accept = 0
    for i in prange(N_samples):
        sample_x = random.random()
        sample_y = random.random()
        sample_x = sample_x*(upper_bound-lower_bound) + lower_bound
        sample_y = sample_y*(upper_bound-lower_bound) + lower_bound
        result = mandelbrot(sample_x, sample_y, num_of_iterations)
        if result == num_of_iterations:
            accept += 1
    return accept*(upper_bound-lower_bound)**2/N_samples


def hypercube_integration(lower_bound, upper_bound, N_samples, num_of_iterations):
    dimensions = 2
    samples = np.random.uniform(size=(N_samples, dimensions))
    tiles = np.tile(np.arange(1, N_samples+1), (dimensions, 1))
    for i in range(tiles.shape[0]):
        np.random.shuffle(tiles[i, :])
    tiles = tiles.T
    samples = (tiles-samples)/N_samples
    samples = samples * (upper_bound-lower_bound) + lower_bound
    accept = 0
    for i in samples:
        result = mandelbrot(i[0], i[1], num_of_iterations)
        if result == num_of_iterations:
            accept += 1
    return accept*(upper_bound-lower_bound)**2/N_samples


# samples_sizes = [4, 5, 6, 7]
# for i in samples_sizes:
#     hyper = hypercube_integration(-1.5, 1, 10**i, 1000)
#     uniform = mc_integrate(-1.5, 1, 10**i, 1000)
#     print(f"Estimate standard uniform sampling: {uniform} \t latin hypercube sampling: {hyper} \t sample size: {10**i}")

#Plotting Convergence

def plot_convergence(a, b, N_iterations, N_samples):
    iters = np.arange(1, N_iterations + 1)
    areas = np.zeros(N_iterations)
    errors = np.zeros(N_iterations)
    area_i = mc_integrate(a, b, N_iterations, N_samples)

    for i in range(1,N_iterations):
        areas[i-1] = mc_integrate(a, b, i, N_iterations)
        errors[i - 1] = areas[i - 1] - area_i

    plt.scatter(iters, errors)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("j")
    plt.ylabel("A_j,s - A_i,s")
    plt.title("Absolute Error in Mandelbrot Integration")
    plt.show()
    
#plot_convergence(-2, 2, 1000, 1000)

def mc_integrate_ellipse(lower_bound, upper_bound, N_samples, num_of_iterations):
    
    center_x = (lower_bound + upper_bound)/2
    center_y = (lower_bound + upper_bound)/2
    major_axis = upper_bound - lower_bound
    minor_axis = upper_bound - lower_bound
    
    accept = 0
    x = []
    y = []
    c = []
    for i in prange(N_samples):
        radius = random.uniform(0, major_axis/2)
        theta = random.uniform(0, 2*np.pi)
        sample_x = np.sqrt(radius) * np.cos(theta)
        sample_y = np.sqrt(radius) * np.sin(theta)
        #if ((sample_x - center_x)/major_axis/2)**2 + ((sample_y - center_y)/minor_axis/2)**2 <= 1:
        x.append(sample_x)
        y.append(sample_y)
        result = mandelbrot(sample_x, sample_y, num_of_iterations)
        if result == num_of_iterations:
            accept += 1
            c.append("g")
        else:
            c.append("r")
    plt.scatter(x,y,color=c, s=2)
    plt.show()
            
    ellipse_area = np.pi * (major_axis/2)**2 
        
    return accept*ellipse_area/N_samples

