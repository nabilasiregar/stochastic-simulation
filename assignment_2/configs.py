import random
import numpy as np

def random_norm(x):
    return abs(random.normalvariate(1/x, 1/x**2))

def hyperexponential(x):
    return random.expovariate(x) if random.random() < 0.75 else random.expovariate(x*1.5)

def configs():
    return {
        "experiment_1":
        {"name": "M/M/n",
        "kwargs": {
            "lam": 0.99,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": random.expovariate,
            "priority": False,
            "preempt": False,
            "debug": False,
            "runtime": 20000},
        },
        "experiment_2":
        {"name": "M/M/1P",
        "kwargs": {
            "lam": 0.99,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": random.expovariate,
            "priority": True,
            "preempt": False,
            "debug": False,
            "runtime": 20000}
        },
        "experiment_3":
        {"name": "M/G/n",
        "kwargs": {
            "lam": 0.99,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": np.random.lognormal,
            "priority": False,
            "preempt": False,
            "debug": False,
            "runtime": 20000}
        },
        "experiment_4":
        {"name": "M/G/n",
        "kwargs": {
            "lam": 0.99,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": hyperexponential,
            "priority": False,
            "preempt": False,
            "debug": False,
            "runtime": 20000}
        },
        "experiment_5":
        {"name": "M/M/nPreempt",
        "kwargs": {
            "lam": 0.99,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": random.expovariate,
            "priority": True,
            "preempt": True,
            "debug": False,
            "runtime": 20000}
        },
        "experiment_6":
        {"name": "M/M/1Preempt",
        "kwargs": {
            "lam": 0.99,
            "mu": 1,
            "dist_wait": random.expovariate,
            "dist_serve": np.random.lognormal,
            "priority": True,
            "preempt": True,
            "debug": False,
            "runtime": 20000}
        },
    }