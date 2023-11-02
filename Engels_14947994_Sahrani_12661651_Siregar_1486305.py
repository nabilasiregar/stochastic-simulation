import math
import cmath
import numpy as np
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


palette = {
    "green": "8bbf9f",
    "blue": "83bcff",
    "midnight": " 124559",
    "violet": "59344f",
    "crayola": "EE204D"
}
    
def mandelbrot(c, num_set):
    z = 0
    for i in range(num_set):
        z = z**2 + c
        if abs(z) > 2:
            return i
    return num_set

def plot_mandelbrot():
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    mandelbrot_set = np.zeros((len(y), len(x)))
    
    for i, x_coord in enumerate(x):
        for j, y_coord in enumerate(y):
            c = complex(x_coord, y_coord)
            val = mandelbrot(c, 100)
            mandelbrot_set[i, j] = val
    
    plt.imshow(mandelbrot_set, cmap=cmap)
    plt.ylabel("Real Numbers")
    plt.xlabel("Imaginary Numbers")
    plt.show()
            
plot_mandelbrot()