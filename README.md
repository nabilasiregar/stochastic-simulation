# Stochastic Simulation

## Projects
- [Estimating the Area of Mandelbrot Set](#assignment1)
- [Discrete Event Simulation for Queuing System](#assignment2)

## Prerequisite
Python >3.10

## Installation
📋 Clone the project
```bash
  git clone https://github.com/nabilasiregar/stochastic-simulation
```

## Assignment 1
Estimating the area of Mandelbrot set with monte carlo integration

### How to run
📁 Go to the project directory
```bash
  cd stochastic-simulation/assignment_1
```

💻 Run the main file
```bash
  python main.py
```

⏳ Run the simulation (optional)
```bash
  python simulation.py
```
❗Running the simulation is computationally expensive and will take **at least** one hour. For convenience, we provide the pre-generated datasets under the 'data' directory. 

📈 Reproduce plots that are presented in our report
```bash
  python plotting.py
```
All plots can be found in the 'assets' folder. If you run the simulation, the csv file will also be saved under assets.

📊 Run the statistical tests
```bash
  python analysis.py
```

## Assignment 2
Discrete event simulation for queuing system using SimPy
### How to run
💻 Run the main file
```bash
  python main.py
```

⏳ Run the simulation (optional)
```bash
  python simulation.py
```
❗ You will be prompted to choose what kind of simulation to run from your terminal: `general` or `test system load`.
General is the main simulation to produce results.csv and test system load is when we vary the arrival rates to produce results_with_rho.csv
If you run the simulation, the csv file be saved under simulation_results folder.

📊 Reproduce plots and stats that are presented in our report
```bash
  python analysis.py
```

```bash
  python plotting.py
```
All plots will be saved under 'simulation_results' folder.
