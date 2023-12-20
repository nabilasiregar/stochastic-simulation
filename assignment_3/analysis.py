from map import *
from map_config import *
import numpy as np
import pandas as pd
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
import ast
from matplotlib.colors import to_rgba


def general_stats(csv_data):
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

    bartlett_stat, bartlett_pvalue = stats.bartlett(*array_like)
    print(
        f"Bartlett's Test - Statistic: {bartlett_stat}, p-value: {bartlett_pvalue}")
    print()

    welch_result = pg.welch_anova(data=data, dv="best_length", between="method")
    print(f"Welch's ANOVA statistic: {welch_result['F'][0]}    p-value: {welch_result['p-unc'][0]}")

    posthoc_result = pg.pairwise_gameshowell(data=data, dv="best_length", between="method")
    print(posthoc_result)


def bar_plot(csv_data):
    df = pd.DataFrame(data)
    alpha = 0.01
    methods = ["Classic Simulated Annealing", "Fast Simulated Annealing", "List-Based Simulated Annealing"]
    
    sa_data = data.groupby("method").get_group("sim_annealing")["best_length"].iloc[0]
    fast_sa_data = data.groupby("method").get_group("fast_annealing")["best_length"].iloc[0]
    list_sa_data = data.groupby("method").get_group("list_sim_annealing")["best_length"].iloc[0]
    
    mean_sa = np.mean(sa_data)
    mean_fast_sa = np.mean(fast_sa_data)
    mean_list_sa = np.mean(list_sa_data)
    
    std_sa = np.std(sa_data)
    std_fast_sa = np.std(fast_sa_data)
    std_list_sa = np.std(list_sa_data)
    
    means = [mean_sa, mean_fast_sa, mean_list_sa]
    stds = [std_sa, std_fast_sa, std_list_sa]
    
    method_colors = {
    "Classic Simulated Annealing": "#8BBF9F",
    "Fast Simulated Annealing": "#83BCFF",
    "List-Based Simulated Annealing": "#124559",
    }
    
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        alpha = 0.05
        standard_error = stds[i]/ np.sqrt(20)
        df = 19
        color = to_rgba(method_colors[method])
        plt.bar(i, means[i], yerr=standard_error, capsize=5, color=color, label=method)      
    plt.ylabel('Expected Waiting Time', fontsize = 16)
    plt.yticks(fontsize = 14)
    plt.legend(["Classic", "Fast", "List-based"], loc='upper right', fontsize = 14)
    plt.title('Mean Best Path Lengths', fontsize = 16)
    plt.tight_layout()
    plt.show()


def error_plot_methods(csv_data):
    nodes = read_csv(MEDIUM_MAP)
    paths = add_paths(MEDIUM_OPT)
    known_best_length = calculate_path_length(paths, nodes)
    
    df = pd.DataFrame(data)
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
        axes[i].fill_between(mean_t_list, error_list - std_length_list, error_list + std_length_list, color = color, alpha=0.3)
        axes[i].invert_xaxis()
        axes[i].tick_params(axis='both', which='both', labelsize=16)  
        axes[i].set_title(f'{titles[sim_type]}', fontsize=16)
        best_length = min(sim_data["best_length"].values)
        axes[i].text(0.5, 0.9, f'Best Length: {round(best_length, 3)}', transform=axes[i].transAxes, fontsize=12, color='black')

    
    axes[1].set_xlabel('Temperature (log-scale)', fontsize=16)
    axes[0].set_ylabel(f'Difference from Optimal Path Length (log-scale)', fontsize=16)
    plt.suptitle('Convergence Plots for Simulated Annealing Methods', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    
        #color = to_rgba(method_colors[sim_type])
    
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
    
    for sim_type, sim_data in grouped:
        sa_t_lists = sim_data['t_list']
        sa_length_lists = sim_data['length_list']

        t_lists = [np.array(ast.literal_eval(t_list)) for t_list in sa_t_lists]
        length_lists = [np.array(ast.literal_eval(length_list)) for length_list in sa_length_lists]
    
        mean_t_list = np.mean(t_lists, axis=0)
        mean_length_list = np.mean(length_lists, axis=0)
        std_length_list = np.std(length_lists, axis=0)
    
        error_list = mean_length_list - known_best_length
    
        #color = to_rgba(method_colors[sim_type])
    
        plt.loglog(mean_t_list, error_list, label=f'{sim_type}')
        plt.fill_between(mean_t_list, error_list - std_length_list, error_list + std_length_list, alpha=0.3)

    plt.gca().invert_xaxis()
    plt.xlabel('Temperature (log-scale)', fontsize=16)
    plt.ylabel('Difference from Optimal Path Length (log-scale)', fontsize=16)
    plt.title('Convergence Plots for Varying Chain Lengths', fontsize=18)
    plt.legend(title='Chain Length', fontsize=12)

    plt.tight_layout()
    plt.show()


method_data = pd.read_csv("./method_results_medium.csv", header=0)
cooling_factor_data = pd.read_csv("./cooling_factor_results.csv", header=0)
chain_length_data = pd.read_csv("./chain_length_results.csv", header=0)

# general_stats(data)
# bar_plot(data)
#error_plot_methods(data)
error_plot_cooling_factors(cooling_factor_data)
#error_plot_chain_lengths(chain_length_data)


