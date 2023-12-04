"""Module to perform discrete-event simulation"""
import simpy

class Customer():
    """Customer arrives, is served and leaves."""
    def __init__(self, env, name, duration, priority=False):
        self.env = env
        self.name = name
        self.duration = duration
        self.arrival_time = env.now
        self.priority = self.duration if priority else 0

class Server:
    """Server preempt less important tasks. The server continues serving the costumer when the server is done
  serving customer with the shortest job"""
    def __init__(self, env, capacity):
        self.resource = simpy.PreemptiveResource(env, capacity=capacity)

class Simulation:
    """
        Input parameters
        lam: the arrival rate to the system
        mu: the capacity of each of n equal servers
        dist_wait: the arrival rate distribution
        dist_serve: the service rate distribution
        n_servers: the number of servers/resources
        priority: Give priority to the smallest jobs (True/False)
        preempt: Abandoning current customer being served (True/False)
        debug: Run in debug mode (True/False)
        runtime: An integer to limit how long the queue keeps going
    """
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

    def add_customers(self):
        i = 0
        while True:
            yield self.env.timeout(self.dist_wait(self.lam))
            i += 1
            customer = Customer(self.env, f'customer {i}', self.dist_serve(self.mu), self.priority)
            if self.debug:
                print(f'Customer {i} arrived at {self.env.now:.2f}')
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
                    usage = self.env.now - interrupt.cause.usage_since
                    customer.duration -= usage
                    yield self.env.timeout(customer.duration)

                    if self.debug:
                        print(f'{customer.name} finished service after preemption at {self.env.now:.2f}')

            waiting_time = self.env.now - arrival_time - customer.duration
            system_time = self.env.now - customer.arrival_time
            busy_time = customer.duration
            self.results['waiting_times'].append(waiting_time)
            self.results['system_times'].append(system_time)
            self.results['utilization'].append(busy_time / self.env.now)
            break

    def run(self):
        self.env.run(until=self.runtime)
        return self.results
