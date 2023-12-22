from map import *
from map_config import *
import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import ast
from matplotlib.colors import to_rgba


def general_stats(data):
    df = pd.DataFrame(data)
    alpha = 0.01

    point_estimated = df.groupby(["method"]).aggregate(
        ["mean", "std", "min", "max"])["best_length"]
    point_estimated["Confidence Interval"] = [stats.t.interval(
        1-alpha, 99, mean, std/33) for mean, std in zip(point_estimated["mean"], point_estimated["std"])]
    print(point_estimated.to_latex(float_format="%.3f"))

    array_like = df.groupby("method")["best_length"].apply(
        list).reset_index()["best_length"].values
    methods = df["method"].unique()
    
    shapiro_stat_group1, shapiro_pvalue_group1 = stats.shapiro(array_like[0])
    shapiro_stat_group2, shapiro_pvalue_group2 = stats.shapiro(array_like[1])
    shapiro_stat_group3, shapiro_pvalue_group3 = stats.shapiro(array_like[2])

    print(
        f"Shapiro-Wilk Test - {methods[0]}: Statistic: {shapiro_stat_group1}, p-value: {shapiro_pvalue_group1}")
    print(
        f"Shapiro-Wilk Test - {methods[1]}: Statistic: {shapiro_stat_group2}, p-value: {shapiro_pvalue_group2}")
    print(
        f"Shapiro-Wilk Test - {methods[2]}: Statistic: {shapiro_stat_group3}, p-value: {shapiro_pvalue_group3}")

    kruskal_stat, kruskal_pvalue = stats.kruskal(*array_like)
    print(
        f"Kruskal-Wallis Test - Statistic: {kruskal_stat}, p-value: {kruskal_pvalue}")
    print()

    dunn_results = sp.posthoc_dunn(df, val_col='best_length', group_col='method', p_adjust='bonferroni')
    print("Dunn's Test with Bonferroni Correction:")
    print(dunn_results)

def error_plot_methods(csv_data):
    nodes = read_csv(MEDIUM_MAP)
    paths = add_paths(MEDIUM_OPT)
    known_best_length = calculate_path_length(paths, nodes)
    
    df = pd.DataFrame(csv_data)
    grouped = df.groupby('method')
    
    method_colors = {
        "sim_annealing": "#8BBF9F",
        "fast_annealing": "#83BCFF",
        "list_sim_annealing": "#124559",
    }

    titles = {
        "sim_annealing": "Classic Simulated Annealing",
        "fast_annealing": "Fast Simulated Annealing",
        "list_sim_annealing": "List-based Simulated Annealing",
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (sim_type, sim_data) in enumerate(grouped):
        sa_t_lists = sim_data['t_list']
        sa_length_lists = sim_data['length_list']

        t_lists = [np.array(ast.literal_eval(t_list)) for t_list in sa_t_lists]
        length_lists = [np.array(ast.literal_eval(length_list)) for length_list in sa_length_lists]
    
        mean_t_list = np.mean(t_lists, axis=0)
        mean_length_list = np.mean(length_lists, axis=0)
        std_length_list = np.std(length_lists, axis=0)
    
        error_list = mean_length_list - known_best_length
    
        color = to_rgba(method_colors[sim_type])
    
        axes[i].loglog(mean_t_list, error_list, label=f'{sim_type} Error', color=color)
        axes[i].fill_between(mean_t_list, error_list - std_length_list, error_list + std_length_list, color=color, alpha=0.3)
        axes[i].invert_xaxis()
        axes[i].tick_params(axis='both', which='both', labelsize=14)  
        axes[i].set_title(f'{titles[sim_type]}', fontsize=16)
        best_length = min(sim_data["best_length"].values)
        axes[i].text(0.5, 0.9, f'Best Length: {round(best_length, 3)}', transform=axes[i].transAxes, fontsize=12, color='black')
    
    plt.subplots_adjust(left=0.1)
    axes[1].set_xlabel('Temperature (log-scale)', fontsize=16)
    axes[0].set_ylabel('Difference from Optimal Path Length (log-scale)', fontsize=16)
    plt.suptitle('Convergence Plots for Simulated Annealing Methods', fontsize=18)
    plt.tight_layout()
    plt.show()

    
def error_plot_cooling_factors(csv_data):
    nodes = read_csv(MEDIUM_MAP)
    paths = add_paths(MEDIUM_OPT)
    known_best_length = calculate_path_length(paths, nodes)
    
    df = pd.DataFrame(csv_data)
    grouped = df.groupby('cooling_factor')
    
    plt.figure(figsize=(10, 6))
    
    for sim_type, sim_data in grouped:
        sa_t_lists = sim_data['t_list']
        sa_length_lists = sim_data['length_list']

        t_lists = [np.array(ast.literal_eval(t_list)) for t_list in sa_t_lists]
        length_lists = [np.array(ast.literal_eval(length_list)) for length_list in sa_length_lists]
    
        mean_t_list = np.mean(t_lists, axis=0)
        mean_length_list = np.mean(length_lists, axis=0)
        std_length_list = np.std(length_lists, axis=0)
    
        error_list = mean_length_list - known_best_length    
        plt.loglog(mean_t_list, error_list, label=f'{sim_type}')
        plt.gca().invert_xaxis()
        plt.fill_between(mean_t_list, error_list - std_length_list, error_list + std_length_list, alpha=0.3)

    plt.xlabel('Temperature (log-scale)', fontsize=16)
    plt.ylabel('Difference from Optimal Path Length (log-scale)', fontsize=16)
    plt.title('Convergence Plots for Varying Cooling Factors', fontsize=18)
    plt.legend(title='Cooling Factor', fontsize=12)

    plt.tight_layout()
    plt.show()

def error_plot_chain_lengths(csv_data):
    nodes = read_csv(MEDIUM_MAP)
    paths = add_paths(MEDIUM_OPT)
    known_best_length = calculate_path_length(paths, nodes)

    df = pd.DataFrame(csv_data)
    grouped = df.groupby('chain_length')

    plt.figure(figsize=(10, 6))

    max_temp_for_chain_length_50 = 0  
    for sim_type, sim_data in grouped:
        sa_t_lists = sim_data['t_list']
        sa_length_lists = sim_data['length_list']

        t_lists = [np.array(ast.literal_eval(t_list)) for t_list in sa_t_lists]
        length_lists = [np.array(ast.literal_eval(length_list)) for length_list in sa_length_lists]

        mean_t_list = np.mean(t_lists, axis=0)
        mean_length_list = np.mean(length_lists, axis=0)
        std_length_list = np.std(length_lists, axis=0)

        error_list = mean_length_list - known_best_length
        if 50 in sim_data['chain_length'].values:
            max_temp_for_chain_length_50 = max(max_temp_for_chain_length_50, np.max(mean_t_list))

        plt.loglog(mean_t_list, error_list, label=f'{sim_type}')
        plt.fill_between(mean_t_list, error_list - std_length_list, error_list + std_length_list, alpha=0.3)

    plt.gca().invert_xaxis()
    plt.xlabel('Temperature (log-scale)', fontsize=16)
    plt.ylabel('Difference from Optimal Path Length (log-scale)', fontsize=16)
    plt.title('Convergence Plots for Varying Chain Lengths', fontsize=18)
    plt.legend(title='Chain Length', fontsize=12)
    plt.xlim(max_temp_for_chain_length_50, 0.01)

    plt.tight_layout()
    plt.show()

method_data = pd.read_csv("./method_results_medium.csv", header=0)
cooling_factor_data = pd.read_csv("./cooling_factor_results.csv", header=0)
chain_length_data = pd.read_csv("./chain_length_results.csv", header=0)

#general_stats(method_data)
# error_plot_methods(method_data)
# error_plot_cooling_factors(cooling_factor_data)
# error_plot_chain_lengths(chain_length_data)
