import simpy
import random
import pandas as pd

DEBUG = False
PRIORITY = True
mu = 0.8
lam = 0.28
n = 2


class Customer():
    def __init__(self, env, name, duration):
        self.env = env
        self.name = name
        self.duration = duration
        if PRIORITY:
            self.priority = duration
        else:
            self.priority = 0


def add_customers(env, lam, mu, servers, results):
    i = 1
    while True:
        yield env.timeout(random.expovariate(lam))
        i += 1
        customer = Customer(env, f'customer {i}', random.expovariate(mu))
        if DEBUG:
            print(f'Customer {i} arrived at {env.now:.2f}')

        env.process(serve_customers(env, customer, servers, results))


def serve_customers(env, customer, servers, results):
    with servers.request(priority=customer.priority) as req:
        yield req

        #print(f'{customer.name} started service at {env.now:.2f}')
        start_time = env.now
        yield env.timeout(customer.duration)
        end_time = env.now
        #print(f'{customer.name} finished service at {env.now:.2f}')
        
        results.append({
            'N': servers.capacity,
            'PriorityQueue': PRIORITY,
            'CustomerName': customer.name,
            'CustomerDuration': customer.duration
        })

def simulate(n, lam, mu):
    results = []
    env = simpy.Environment()
    servers = simpy.PriorityResource(env, capacity=n)
    env.process(add_customers(env, lam, mu, servers, results))
    env.run(until=1000)

    return results

def simulation_results():
    results = []
    for n in [1, 2, 4]:
        results.extend(simulate(n, lam, mu))

    df = pd.DataFrame(results)
    df.to_csv(f"./assignment_2/customer_durations.csv", index=False)
    