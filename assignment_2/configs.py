import random
def random_norm(x):
    return random.normalvariate(x, 0.1)
def configs():
    return {
        "experiment_1":
        {"name": "M/M/n",
        "kwargs": {
            "lam": 0.28,
            "mu": 0.8,
            "dist_wait": random.expovariate,
            "dist_serve": random.expovariate,
            "priority": False,
            "debug": False,
            "runtime": 1000},
        },
        "experiment_2":
        {"name": "M/M/1P",
        "kwargs": {
            "lam": 0.28,
            "mu": 0.8,
            "dist_wait": random.expovariate,
            "dist_serve": random.expovariate,
            "priority": True,
            "debug": False,
            "runtime": 1000}
        },
        "experiment_3":
        {"name": "M/G/n",
        "kwargs": {
            "lam": 0.28,
            "mu": 0.8,
            "dist_wait": random_norm,
            "dist_serve": random_norm,
            "priority": False,
            "debug": False,
            "runtime": 1000}
        },
    }