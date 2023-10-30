import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

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
        if abs(z) > 100:
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
    
    plt.imshow(mandelbrot_set, cmap='viridis')
    plt.show()
            
plot_mandelbrot()