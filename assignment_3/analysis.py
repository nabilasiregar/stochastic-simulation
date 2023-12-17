from map import *
from map_config import *
import numpy as np
import pandas as pd
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
import ast
from matplotlib.colors import to_rgba

data = pd.read_csv("./results.csv", header=0)

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


def error_plot(csv_data, sim_type):
    nodes = read_csv(SMALL_MAP)
    paths = add_paths(SMALL_OPT)
    known_best_length = calculate_path_length(paths, nodes)
    
    df = pd.DataFrame(data)
    filtered_df = df[df['method'] == sim_type]

    t_lists = filtered_df['t_list']
    length_lists = filtered_df['length_list']
    
    t_lists = [np.array(ast.literal_eval(t_list)) for t_list in t_lists]
    length_lists = [np.array(ast.literal_eval(length_list)) for length_list in length_lists]
    mean_t_list = np.mean(t_lists, axis=0)
    mean_length_list = np.mean(length_lists, axis=0)
    std_length_list = np.std(length_lists, axis=0)
    
    error_list = mean_length_list - known_best_length
    
    method_colors = {
    "sim_anealing": "#8BBF9F",
    "fast_annealing": "#83BCFF",
    "list_sim_annealing": "#124559",
    }
    color = to_rgba(method_colors[sim_type])
    
    plt.figure(figsize=(10, 6))
    plt.loglog(mean_t_list, error_list, label=f'{sim_type} Error', color=color)
    plt.fill_between(mean_t_list, error_list - std_length_list, error_list + std_length_list, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.xlim([100, mean_t_list[-1]])
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlabel('Temperature', fontsize = 16)
    plt.ylabel(f'Difference from Optimal Path Length', fontsize = 16)
    titles = {
    "sim_anealing": "Classic Simulated Annealing",
    "fast_annealing": "Fast Simulated Annealing",
    "list_sim_annealing": "List-based Simulated Annealing",
    }
    plt.title(f'{titles[sim_type]} Convergence Plot', fontsize = 16)
    plt.show()

# general_stats(data)
# bar_plot(data)
error_plot(data, "list_sim_annealing")


