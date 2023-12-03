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
        self.resource = simpy.PreemptiveResource(env, capacity=capacity)

class Simulation:
    def __init__(self, lam, mu, dist_wait, dist_serve, n_servers, priority, preempt, debug, runtime):
        self.lam = lam
        self.mu = mu
        self.dist_wait = dist_wait
        self.dist_serve = dist_serve
        self.n_servers = n_servers
        self.priority = priority
        self.preempt = preempt
        self.debug = debug
        self.runtime = runtime
        self.env = simpy.Environment()
        self.server = Server(self.env, capacity=self.n_servers)
        self.results = {'waiting_times': [], 'system_times': [], 'utilization': []}
        self.env.process(self.add_customers())
        self.queue = []

    def add_customers(self):
        i = 0
        while True:
            yield self.env.timeout(self.dist_wait(self.lam))
            i += 1
            customer = Customer(self.env, f'customer {i}', self.dist_serve(self.mu), self.priority)
            self.queue.append(customer)
            if self.debug:
                print(f'Customer {i} arrived at {self.env.now:.2f}, duration {customer.duration:.2f}')
            self.env.process(self.serve_customer(customer))

    def serve_customer(self, customer):
        arrival_time = self.env.now
        while True:
            with self.server.resource.request(priority=customer.priority, preempt=self.preempt) as req:
                yield req
                try:
                    if self.debug:
                        print(f'{customer.name} started service at {self.env.now:.2f}')
                    yield self.env.timeout(customer.duration)
                    if self.debug:
                        print(f'{customer.name} finished service at {self.env.now:.2f}')
                except simpy.Interrupt as interrupt:
                    by = interrupt.cause.by
                    usage = self.env.now - interrupt.cause.usage_since
                    customer.duration -= usage
                    yield self.env.timeout(customer.duration)

                    if self.debug:
                        print(f'{customer.name} finished service after preemption at {self.env.now:.2f}')



            waiting_time = self.env.now - arrival_time - customer.duration
            system_time = self.env.now - customer.arrival_time
            self.queue.remove(customer)
            busy_time = customer.duration
            self.results['waiting_times'].append(waiting_time)
            self.results['system_times'].append(system_time)
            self.results['utilization'].append(busy_time / self.env.now)
            break

    def run(self):
        self.env.run(until=self.runtime)
        # printing all the unterminated customers
        if self.debug:
            for customer in self.queue:
                print(f'Customer {customer.name} did not finish service, arrived at {customer.arrival_time:.2f}')
        return self.results
    
def main(debug, n_servers):
    mu = 0.8
    lam = 0.28
    priority = True
    debug = True
    n_servers = 2
    preempt = True
    runtime = 100

    simulation = Simulation(lam, mu, random.expovariate, random.expovariate, n_servers, priority, preempt, debug, runtime)
    simulation.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the simulation in debug mode.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--servers', type=int, default=2, help='Number of servers')
    args = parser.parse_args()

    main(args.debug, args.servers)
