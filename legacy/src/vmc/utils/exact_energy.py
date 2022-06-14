import numpy as np


def exact_energy(N, dim, omega=1.):
    return (omega * dim * N) / 2


def exact_energy_scaled(dim, omega=1.):
    return (omega * dim) / 2
