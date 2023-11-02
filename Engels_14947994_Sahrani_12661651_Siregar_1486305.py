import numpy as np
import cmath
import matplotlib.pyplot as plt
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
def mandelbrott(x, y, max_iteration):
    c = complex(x, y)
    z = 0
    iteration = 0
    while abs(z) < 1 and iteration < max_iteration:
        z = z**2 + c
        iteration += 1
    return iteration


x = np.linspace(-2, 0.47, 2000)
y = np.linspace(-1.12, 1.12, 2000)
values = np.ndarray((x.shape[0], y.shape[0]))


@njit(parallel=True)
def get_mandelbrot(x, y, matrix, max_iteration):
    for i in prange(matrix.shape[0]):
        for j in prange(matrix.shape[1]):
            matrix[i, j] = mandelbrott(x[i], y[j], max_iteration)
    return matrix


max_iterations = 1000
output = get_mandelbrot(x, y, values, max_iterations)
plt.matshow(output, cmap=cmap)
plt.ylabel("Real Numbers")
plt.xlabel("Imaginary Numbers")
plt.colorbar()
plt.show()
