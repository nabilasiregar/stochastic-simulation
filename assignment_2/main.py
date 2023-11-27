import simpy
import random

class Customer():
    def __init__(self, env, name, duration, priority=False):
        self.env = env
        self.name = name
        self.duration = duration
        self.arrival_time = env.now
        self.priority = self.duration if priority else 0

class Server:
    def __init__(self, env, capacity):
        self.resource = simpy.PriorityResource(env, capacity=capacity)

class Simulation:
    def __init__(self, lam, mu, n_servers, priority, debug, runtime):
        self.lam = lam
        self.mu = mu
        self.n_servers = n_servers
        self.priority = priority
        self.debug = debug
        self.runtime = runtime
        self.env = simpy.Environment()
        self.server = Server(self.env, capacity=self.n_servers)
        self.results = {'waiting_times': [], 'system_times': [], 'utilization': []}
        self.env.process(self.add_customers())

    def add_customers(self):
        i = 0
        while True:
            yield self.env.timeout(random.expovariate(self.lam))
            i += 1
            customer = Customer(self.env, f'customer {i}', random.expovariate(self.mu), self.priority)
            if self.debug:
                print(f'Customer {i} arrived at {self.env.now:.2f}')
            self.env.process(self.serve_customer(customer))

    def serve_customer(self, customer):
        arrival_time = self.env.now
        with self.server.resource.request(priority=customer.priority) as req:
            yield req
            waiting_time = self.env.now - arrival_time
            self.results['waiting_times'].append(waiting_time)
            if self.debug:
                print(f'{customer.name} started service at {self.env.now:.2f}')
            yield self.env.timeout(customer.duration)
            if self.debug:
                print(f'{customer.name} finished service at {self.env.now:.2f}')
            system_time = self.env.now - customer.arrival_time
            self.results['system_times'].append(system_time)
            busy_time = customer.duration
            self.results['utilization'].append(busy_time / self.env.now)

    def run(self):
        self.env.run(until=self.runtime)
        return self.results
