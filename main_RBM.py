import numpy as np
from numpy.random import default_rng
from src import vmc
import matplotlib.pyplot as plt

nsamples = 1000
training_iterations = 200

N = 2
dim = 2
nhidden = 4

wf = vmc.RBMWF(N, dim, nhidden, scale=1.0, rng=default_rng, seed=2123)
rng = default_rng(0)
rbm = vmc.samplers.RWMRBM(wf)
initial_positions = rng.normal(loc=0.0, scale=1.0, size=(N, dim))
print("Initial positions: ", initial_positions)
energies = rbm.train(training_iterations, nsamples, initial_positions, eta=0.05)


for i, energy in enumerate(energies):
    #if energy < 2.5:
    plt.scatter(i, energy)
    plt.hlines(2.0, 0, 200, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
plt.savefig("test.pdf")
