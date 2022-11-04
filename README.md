# On the Hybridization of Geometric Semantic GP with Gradient-based Optimizers

Julia implementation for paper "On the Hybridization of Geometric Semantic GP with Gradient-based Optimizers", Gloria Pietropolli, Alessia Paoletti, Mauro Castelli, Luca Manzoni, 2022.

### Abstract
Geometric Semantic Genetic Programming (GSGP) is a popular form of GP where the effect of crossover and mutation can be expressed as geometric operations on a semantic space. A recent study showed that GSGP can be hybridized with a standard gradient-based optimized, Adam, commonly used in training artificial neural networks.

We expand upon that work by considering more gradient-based optimizers, a deeper investigation of their parameters, how the hybridization is performed, and a more comprehensive set of benchmark problems. With the correct choice of hyperparameters, this hybridization improves the performances of GSGP and allows it to reach the same fitness values with fewer fitness evaluations.

## Instructions

Code runs with python 3.8.5 and Julia 1.4.1 on Ubuntu 20.04.
To run the code with Gradient Descent optimizer, enter the following command:

```bash
julia gsgp.jl --problem --optimizer-- lr --p1 --p2
```
To run the code with ADAM optimizer, enter the following command:

```bash
julia gsgp-adam.jl --problem --optimizer-- lr --p1 --p2
```

where the inputs arguments stands for: 
* `--problem` is the benchmark considered
* `--optimizer` is the optimizer considered
* `--lr` is the learning rate
* `--p1` is the number of steps of GSGP
* `--p2` is the number of steps of grdient-based optimizer

This run will return the fitness results for the 30 runs performed and save them in the `results` folder.

The considered benchmark problems are: 
- _human oral bioavaibility_
- _median lethal dose_
- _protein-plasma binding level_
- _yacht hydrodynamics_
- _concrete slump_
- _concrete compressive strenght_
- _airfoil self-noise_
- _parkinson_

that are saved in the `dataset` folder.

The code to reproduce the box-plot of the paper, it is sufficient to run:
```bash
python3 boxplot.py 
```
