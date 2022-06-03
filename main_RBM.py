import numpy as np
from numpy.random import default_rng
from src import vmc

nsamples = 10000
training_iterations = 100

N = 2
dim = 1
nhidden = 2

wf = vmc.RBMWF(N, dim, nhidden, scale=0.5, rng=default_rng, seed=2123)
rng = default_rng(0)
net = vmc.samplers.RWMRBM(wf)
initial_positions = rng.normal(loc=0.0, scale=1.0, size=(N, dim))
print("Initial positions: ", initial_positions)
energies = net.train(training_iterations, nsamples, initial_positions, eta=0.1)
