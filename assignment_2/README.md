# Assignment 2
Queue-tie Py: Discrete event simulation of average waiting times for varying queue types with SimPy

## How to Run
📁 Go to the project directory
```bash
  cd stochastic-simulation/assignment_2
```

💻  Run the simulation
```bash
  python simulation.py
```
You will be prompted to choose what kind of simulation to run from your terminal: `general`, `test system load` or `individual`.
General is the main simulation to produce **results.csv**, test system load is when we vary the arrival rates to produce **results_with_rho.csv**, individual is when the simulation produce **results_individual.csv**

All csv files generated from simulation run will be saved under the simulation_results folder.

📊 Reproduce plots and stats that are presented in our report (❗ You need to run the simulation first)
```bash
  python analysis.py
```

```bash
  python plotting.py
```
All plots will be saved under 'simulation_results' folder.
