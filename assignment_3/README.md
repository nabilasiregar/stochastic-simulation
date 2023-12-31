# Assignment 3
Solving Traveling Salesman Problem using Simulated Annealing

## How to Run
📁 Go to the project directory
```bash
  cd stochastic-simulation/assignment_3
```

⚙️ Tuning simulated annealing parameters: initial temperature, Markov chain length, cooling factor 
```bash
  python train.py --multirun
```
For convenience, we provide our training results under the 'data' folder. 

⏳ Run the simulation (optional)
```bash
  python simulation.py
```
You will be prompted to choose what kind of simulation to run from your terminal: `general`, `cooling_factor`, `fast_cooling_factor` or `markov chain`.

+ `general`: Run all simulated annealing types (Simulated Annealing, List-Based Simulated Annealing, Fast Simulated Annealing)
+ `cooling_factor`: Run regular simulated annealing with varying cooling factors
+ `fast_cooling_factor`: Run fast simulated annealing with varying cooling factors
+ `markov_chain`: Run regular simulated annealing with varying Markov chain length

All csv files generated from simulation run will be saved under the root of assignment_3 folder.
For convenience, we provide the pre-generated datasets under the 'data' folder. 

📊 Reproduce figures and stats that are presented in our report 
```bash
  python analysis.py
```
