# Assignment 2
Discrete event simulation for queuing system using SimPy

## Prerequisite
Python >3.10

## How to Run
💻  Run the simulation
```bash
  python simulation.py
```
You will be prompted to choose what kind of simulation to run from your terminal: `general` or `test system load`.
General is the main simulation to produce results.csv and test system load is when we vary the arrival rates to produce results_with_rho.csv

If you run the simulation, the csv file be saved under simulation_results folder.

📊 Reproduce plots and stats that are presented in our report
```bash
  python analysis.py
```

```bash
  python plotting.py
```
❗ You need to run the simulation first to have the csv files in the specified file path.
All plots will be saved under 'simulation_results' folder.
