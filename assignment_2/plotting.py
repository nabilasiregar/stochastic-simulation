"""
Comprehensive file to plot rho values against waiting times and variances from results_with_rho.csv
Input: Path to file results_with_rho.csv
Output: Plots for rho against average waiting time and variances 

Plots are saved under simulation_results folder
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

df = pd.read_csv('./simulation_results/results_with_rho.csv')

queue_filters = {
    "M/M/1": (df["n_server"] == 1) & (df["dist_serve"] == "expovariate") & (df["priority"] is False),
    "M/M/2": (df["n_server"] == 2) & (df["dist_serve"] == "expovariate") & (df["priority"] is False),
    "M/M/4": (df["n_server"] == 4) & (df["dist_serve"] == "expovariate") & (df["priority"] is False)
}

plt.figure(figsize=(14, 8))

method_colors = {
    "M/M/1": "#8BBF9F",
    "M/M/2": "#83BCFF",
    "M/M/4": "#EE204D"
    }

# Plotting average waiting time under varying values of rho
for method, filter_condition in queue_filters.items():
    filtered_df = df[filter_condition]
    grouped_data = filtered_df.groupby('rho')['waiting_time'].mean().reset_index()
    std_deviation = filtered_df.groupby('rho')['waiting_time'].std().reset_index()
    color = to_rgba(method_colors[method])    
    plt.plot(grouped_data['rho'], grouped_data['waiting_time'], marker='', linestyle='-', label=method, color=method_colors[method])
    plt.fill_between(grouped_data['rho'], 
                     grouped_data['waiting_time'] - std_deviation['waiting_time'], 
                     grouped_data['waiting_time'] + std_deviation['waiting_time'], 
                     color=method_colors[method], alpha=0.2)

plt.xlabel(r"$\rho$", fontsize = 16)
plt.xticks(fontsize = 18)
plt.ylabel('Average Waiting Time', fontsize = 16)
plt.yticks(fontsize = 18)
plt.title("Average Waiting Time of M/M/1, M/M/2, M/M/4 Based on Rho", fontsize = 18)
plt.legend(fontsize = 16)
plt.savefig('./simulation_results/rho.png')
plt.close()

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
plt.show()
plt.savefig('./simulation_results/variances.png')
plt.close()

f = pd.read_csv('simulation_results/ttest.csv',header=0)
data1 = f.loc[f['n_server'] == 1]['waiting_time'].to_numpy()
data2 = f.loc[f['n_server'] == 2]['waiting_time'].to_numpy()
data4 = f.loc[f['n_server'] == 4]['waiting_time'].to_numpy()

def get_means(data):
    split_data = np.array_split(data, 1000)
    means = np.array([np.mean(arr) for arr in split_data])  # Calculate the mean for each split
    return means

def get_waiting_time(lam, mu, rho, n):
    if n == 1:
        return rho / (mu-lam)

    if n == 2:
        PI = 2 * rho**2 / (1+ rho)
        return PI / (n*mu-lam)
    else:
        PI = 32*rho**4 / (8*rho**3 + 12*rho**2 + 9*rho + 3)
        return PI / (n*mu-lam)
    
means1 = get_means(data1)
means2 = get_means(data2)
means4 = get_means(data4)

cumulative_mean1 = np.cumsum(means1) / np.arange(1, len(means1) + 1)
cumulative_mean2 = np.cumsum(means2) / np.arange(1, len(means2) + 1)
cumulative_mean4 = np.cumsum(means4) / np.arange(1, len(means4) + 1)

analytical1 = get_waiting_time(0.98, 1, 0.98, 1)
analytical2 = get_waiting_time(0.98*2, 1, 0.98, 2)
analytical4 = get_waiting_time(0.98*4, 1, 0.98, 4)

#plotting using matplotlib
plt.plot(range(0,100000,100), cumulative_mean1, label='1 Server',  color=method_colors['M/M/1'])
plt.plot(range(0,100000,100),cumulative_mean2, label='2 Servers', color=method_colors['M/M/2'])
plt.plot(range(0,100000,100),cumulative_mean4, label='4 Servers', color=method_colors['M/M/4'])

# horizontal line showing the analytical value
plt.axhline(analytical1, label='Analytical 1 Server', linestyle='--', color=method_colors['M/M/1'])
plt.axhline(analytical2, label='Analytical 2 Servers', linestyle='--', color=method_colors['M/M/2'])
plt.axhline(analytical4, label='Analytical 4 Servers', linestyle='--', color=method_colors['M/M/4'])
plt.legend(loc='upper right' ,fontsize=10, borderaxespad = 2)
plt.title('Samples mean for different starting points', fontsize=14)
plt.xlabel('starting point', fontsize=12)
plt.ylabel('mean', fontsize=12)
plt.tight_layout()
plt.savefig('./simulation_results/.png')