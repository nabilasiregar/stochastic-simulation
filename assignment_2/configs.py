import random
import numpy as np

def lognormal(x):
    """Accepts the lambda from an exponential distribution and returns a lognormal distribution with the same mean and variance"""
    s = np.sqrt(1/x**2)
    m = 1/x
    block = np.log((s/m)**2 + 1)
    scaled_variance = np.sqrt(block)
    scaled_mean = np.log(1/x) - block/2
    return np.random.lognormal(scaled_mean, scaled_variance)

def hyperexponential(x):
    """Accepts the lambda from an exponential distribution and returns a hyperexponential distribution"""
    return random.expovariate(x) if random.random() < 0.75 else random.expovariate(x*1.5)

def configs():
    """Configuration to handle passing different arguments to the Simulation function

        Argument values can be updated/re-assigned during the simulation

        Example usage:
        experiment_config = configs()['experiment_1']['kwargs']
        experiment_config['lam'] = 0.90
    """
    return {
        f"experiment_{i}{runs}":
        {"name": "M/M/n",
        "kwargs": {
            "lam": 0.98,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": random.expovariate,
            "priority": False,
            "preempt": False,
            "debug": False,
            "runtime": 30000},
        },
        "experiment_2":
        {"name": "M/M/1P",
        "kwargs": {
            "lam": 0.98,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": random.expovariate,
            "priority": True,
            "preempt": False,
            "debug": False,
            "runtime": 30000}
        },
        "experiment_3":
        {"name": "M/G/n",
        "kwargs": {
            "lam": 0.98,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": lognormal,
            "priority": False,
            "preempt": False,
            "debug": False,
            "runtime": 30000}
        },
        "experiment_4":
        {"name": "M/G/n",
        "kwargs": {
            "lam": 0.98,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": hyperexponential,
            "priority": False,
            "preempt": False,
            "debug": False,
            "runtime": 30000}
        },
        "experiment_5":
        {"name": "M/M/nPreempt",
        "kwargs": {
            "lam": 0.98,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": random.expovariate,
            "priority": True,
            "preempt": True,
            "debug": False,
            "runtime": 30000}
        },
        "experiment_6":
        {"name": "M/M/1Preempt",
        "kwargs": {
            "lam": 0.98,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": lognormal,
            "priority": True,
            "preempt": True,
            "debug": False,
            "runtime": 30000}
        },
    }
