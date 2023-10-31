import numpy as np
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


def mandelbrott(x, y, max):
    iteration = 0
    x_start = x
    y_start = y
    while (x**2 + y**2) < 4 and iteration < max:
        x_temp = x**2 - y**2 + x_start
        y = 2*x*y + y_start
        x = x_temp
        iteration += 1
    return iteration


x = np.linspace(-2, 0.47, 500)
y = np.linspace(-1.12, 1.12, 500)
values = np.ndarray((x.shape[0], y.shape[0]))
max_iterations = 1000

for i in range(len(x)):
    for j in range(len(y)):
        values[i, j] = mandelbrott(x[i], y[j], max_iterations)

plt.matshow(values, cmap=cmap)
plt.colorbar()
plt.show()
