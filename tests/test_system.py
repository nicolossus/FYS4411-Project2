#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from src import vmc

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(("N", "alpha"),
                         [(1, 0.4),
                          (10, 0.6),
                          (100, 0.49),
                          (500, 0.8), ])
def test_logprob(N, alpha):
    """Verify that System class properly computes the log density of
    the wave function by comparing with the analytical implementation.
    Verify for several values of variational parameter alpha and number of
    particles in the system.
    """
    dim = 3                      # Dimensionality
    omega = 1.                   # Oscillator frequency
    r = onp.random.rand(N, dim)  # Positions
    r = jnp.array(r)

    analytical = vmc.ANIB(N, dim, omega)
    numerical = vmc.LogNIB(omega)

    lp_a = analytical.logprob(r, alpha)
    lp_n = numerical.logprob(r, alpha)

    assert jnp.allclose(lp_a, lp_n)


@pytest.mark.parametrize(("N", "dim"),
                         [(1, 1),
                          (10, 1),
                          (100, 1),
                          (500, 1),
                          (1, 2),
                          (10, 2),
                          (100, 2),
                          (500, 2),
                          (1, 3),
                          (10, 3),
                          (100, 3),
                          (500, 3),
                          ])
def test_exact_energy_analytical(N, dim):
    """Verify that the analytical (variational) implementation computes the
    correct local energy by comparing with the closed form of the exact energy.
    Verify for several number of particles and across all spatial dimensions.
    """
    omega = 1.                   # Oscillator frequency
    alpha = 0.5                  # Optimal variational parameter
    r = onp.random.rand(N, dim)  # Positions
    r = jnp.array(r)

    exact_energy = 0.5 * omega * dim * N
    analytical = vmc.ANIB(N, dim, omega)

    locE = analytical.local_energy(r, alpha)
    assert jnp.allclose(exact_energy, locE)


@pytest.mark.parametrize(("N", "dim"),
                         [(1, 1),
                          (10, 1),
                          (100, 1),
                          (500, 1),
                          (1, 2),
                          (10, 2),
                          (100, 2),
                          (500, 2),
                          (1, 3),
                          (10, 3),
                          (100, 3),
                          (500, 3),
                          ])
def test_exact_energy_numerical(N, dim):
    """Verify that the numerical implementation computes the correct
    local energy by comparing with the closed form of the exact energy.
    Verify for several number of particles and across all spatial dimensions.
    """
    omega = 1.                   # Oscillator frequency
    alpha = 0.5                  # Optimal variational parameter
    r = onp.random.rand(N, dim)  # Positions
    r = jnp.array(r)

    exact_energy = 0.5 * omega * dim * N
    numerical = vmc.LogNIB(omega)

    locE = numerical.local_energy(r, alpha)
    assert jnp.allclose(exact_energy, locE)


@pytest.mark.parametrize(("N", "dim"),
                         [(1, 1),
                          (10, 1),
                          (100, 1),
                          (500, 1),
                          (1, 2),
                          (10, 2),
                          (100, 2),
                          (500, 2),
                          (1, 3),
                          (10, 3),
                          (100, 3),
                          (500, 3),
                          ])
def test_drift_force(N, dim):
    """
    Verify that System class properly computes the drift force by comparing
    with the analytical implementation Verify for several number of particles
    and across all spatial dimensions.
    """

    omega = 1.                   # Oscillator frequency
    alpha = 0.5                  # Optimal variational parameter
    r = onp.random.rand(N, dim)  # Positions
    r = jnp.array(r)

    analytical = vmc.ANIB(N, dim, omega)
    numerical = vmc.LogNIB(omega)

    F_a = analytical.drift_force(r, alpha)
    F_n = numerical.drift_force(r, alpha)

    assert jnp.allclose(F_a, F_n)
