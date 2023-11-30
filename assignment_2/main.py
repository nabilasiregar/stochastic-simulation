import simpy
import random
import argparse

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
    def __init__(self, lam, mu, dist_wait, dist_serve, n_servers, priority, debug, runtime):
        self.lam = lam
        self.mu = mu
        self.dist_wait = dist_wait
        self.dist_serve = dist_serve
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
    
def main(debug, n_servers):
    mu = 0.8
    lam = 0.28
    priority = True
    runtime = 1000

    simulation = Simulation(lam, mu, n_servers, priority, debug, runtime)
    simulation.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the simulation in debug mode.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--servers', type=int, default=2, help='Number of servers')
    args = parser.parse_args()

    main(args.debug, args.servers)
