import simpy
import random

DEBUG = True
PRIOTITY = True
mu = 0.8
lam = 0.28
n = 2


class Customer():
    def __init__(self, env, name, duration):
        self.env = env
        self.name = name
        self.duration = duration
        if PRIOTITY:
            self.priority = duration
        else:
            self.priority = 0


def add_customers(env, lam, mu, servers):
    i = 1
    while True:
        yield env.timeout(random.expovariate(lam))
        i += 1
        customer = Customer(env, f'customer {i}', random.expovariate(mu))
        if DEBUG:
            print(f'Customer {i} arrived at {env.now:.2f}')

        env.process(serve_customers(env, customer, servers))


def serve_customers(env, customer, servers):
    with servers.request(priority=customer.priority) as req:
        yield req

        print(f'{customer.name} started service at {env.now:.2f}')
        yield env.timeout(customer.duration)
        print(f'{customer.name} finished service at {env.now:.2f}')


env = simpy.Environment()
servers = simpy.PriorityResource(env, capacity=n)
env.process(add_customers(env, lam, mu, servers))

env.run(until=1000)
