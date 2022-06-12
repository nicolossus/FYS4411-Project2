#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import default_rng



N = 4
d = 3
hidden = 2

rng = default_rng(0)
positions = rng.normal(loc=0.0, scale=1.0, size=(N, d))
print(positions)
print(positions.reshape(N*d))

weights = rng.normal(loc=0.0, scale=1.0, size=(N, d, hidden))


print(weights)
print(weights.reshape(N*d, hidden))
#print(matrix_product)



print("Flattened positions times flattend weights")
print(positions.reshape(N*d).T@weights.reshape(N*d, hidden))
print("Einsum: ")
print(np.einsum("ij, ijk->k", positions, weights))


denom = rng.normal(loc=0.0, scale=1.0, size=(hidden))
r = positions.reshape(N, d, 1)
test = r/denom
print(test.shape)


print(1 + np.exp(1))
print(np.exp(np.logaddexp(1, 0)))
