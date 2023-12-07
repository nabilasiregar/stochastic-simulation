"""
Comprehensive file to plot rho values against variances in waiting times from results_with_rho.csv 
Input: Path to file results_with_rho.csv
Output: Plot for rho against variances in waiting times and saved under simulation_results folder
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./simulation_results/results_with_rho.csv')

queue_filters = {
    "M/M/1": (df["n_server"] == 1) & (df["dist_serve"] == "expovariate") & (df["priority"] == False),
    "M/M/2": (df["n_server"] == 2) & (df["dist_serve"] == "expovariate") & (df["priority"] == False),
    "M/M/4": (df["n_server"] == 4) & (df["dist_serve"] == "expovariate") & (df["priority"] == False)
}

plt.figure(figsize=(14, 8))

method_colors = {
    "M/M/1": "#8BBF9F",
    "M/M/2": "#83BCFF",
    "M/M/4": "#EE204D"
    }

# Plotting average waiting time variances
for method, filter_condition in queue_filters.items():
    filtered_df = df[filter_condition]
    grouped_variance = filtered_df.groupby('rho')['waiting_time'].var().reset_index()
    std_deviation = filtered_df.groupby('rho')['waiting_time'].std().reset_index()
    plt.plot(grouped_variance['rho'], grouped_variance['waiting_time'], marker='', linestyle='-', label=method, color=method_colors[method])
    plt.fill_between(grouped_variance['rho'], 
                     grouped_variance['waiting_time'] - std_deviation['waiting_time'],
                     grouped_variance['waiting_time'] + std_deviation['waiting_time'],
                     color=method_colors[method], alpha=0.5)

plt.xlabel(r"$\rho$", fontsize = 16)
plt.xticks(fontsize = 18)
plt.ylabel('Variance of Waiting Times', fontsize = 16)
plt.yticks(fontsize = 18)
plt.title('Waiting Time Variances for M/M/1, M/M/2, M/M/4 Based on Rho', fontsize = 18)
plt.legend(fontsize = 16)
plt.savefig('./simulation_results/variances.png')
plt.close()
