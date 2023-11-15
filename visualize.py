from Engels_14947994_Sahrani_12661651_Siregar_1486305 import normalized_palette
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./assets/mandelbrot_sample_size_comparison.csv')
mean_areas = data.groupby(['method', 'sample_size'])['area'].mean().reset_index()

method_colors = {
    "uniform_square": normalized_palette["green"],
    "uniform_circle": normalized_palette["blue"],
    "latin_hypercube": normalized_palette["midnight"],
    "orthogonal": normalized_palette["crayola"]
}

plt.figure(figsize=(10, 6))
for method in mean_areas['method'].unique():
    method_data = mean_areas[mean_areas['method'] == method]
    plt.plot(method_data['sample_size'], method_data['area'],
             marker='o', label=method, color=method_colors[method])

plt.xscale('log')
plt.xlabel('Sample Size', fontsize=18)
plt.xticks(fontsize=16)
plt.ylabel('Estimated Area', fontsize=18)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.savefig('./assets/comparison_by_sample_size.png')
