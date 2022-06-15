#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import System, WaveFunction


class ASHONIB(WaveFunction):
    """Single particle wave function with Gaussian kernel for a Non-Interacting
    Boson (NIB) system in a spherical harmonic oscillator.

    Analytical Spherical Harmonic Oscillator Non-Interacting Bosons (ASHONIB).

    Parameters
    ----------
    N : int
        Number of particles in system
    dim : int
        Dimensionality of system
    omega : float
        Harmonic oscillator frequency
    """

    def __init__(self, N, dim, omega=1.):
        super().__init__(N, dim)
        self._omega = omega

        # precompute
        self._Nd = N * dim
        self._halfomega2 = 0.5 * omega * omega

    def wf(self, r, alpha):
        """Scalar evaluation of the trial wave function"""

        return -alpha * np.sum(r * r)

    def local_energy(self, r, alpha):
        """Compute the local energy.

        Parameters
        ----------
        r : np.ndarray, shape=(n_particles, dim)
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        float
            Computed local energy
        """

        locE = self._Nd * alpha + \
            (self._halfomega2 - 2 * alpha * alpha) * np.sum(r * r)
        return locE

    def drift_force(self, r, alpha):
        """Drift force"""

        return -4 * alpha * r

    def grad_alpha(self, r, alpha):
        """Gradient of wave function w.r.t. variational parameter alpha"""

        return -np.sum(r * r)


class SHONIB(System):
    """
    Spherical HO, Non-Interacting Boson (NIB) system in log domain using JAX.

    Spherical Harmonic Oscillator Non-Interacting Bosons (SHONIB).

    Trial wave function:
                psi = -alpha * r**2
    """

    def __init__(self, omega=1.):
        super().__init__()
        self._omega2 = omega * omega

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):
        return -alpha * jnp.sum(r * r)

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        return 0.5 * self._omega2 * jnp.sum(r * r)


class EHONIB(System):
    """
    Elliptical Harmonic Oscillator Non-Interacting Bosons (EHONIB).
    """

    def __init__(self):
        super().__init__()
        self._beta = 2.82843
        self._gamma2 = self._beta * self._beta

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):
        r2 = jnp.square(r)
        r2 = r2.at[-1].multiply(self._beta)
        return -alpha * jnp.sum(r2)

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        r2 = jnp.square(r)
        r2 = r2.at[:, 2].multiply(self._gamma2)
        return 0.5 * jnp.sum(r2)


class AEHONIB(WaveFunction):
    """
    Analytical Elliptical Harmonic Oscillator Non-Interacting Bosons (AEHONIB)
    """

    def __init__(self, N, dim, omega, beta=2.82843):
        super().__init__(N, dim)
        self._omega = omega
        self._halfomega2 = 0.5 * omega * omega
        self._beta = beta
        self._gamma2 = beta * beta

    def wf(self, r, alpha):
        return self._single(r, alpha)

    def _single(self, r, alpha):
        r2 = r * r
        r2[:, 2] = r2[:, 2] * self._beta
        return -alpha * np.sum(r2)

    def _gradient(self, r, alpha):
        # Single particle gradient
        r_ = r.copy()
        r_[:, 2] = r_[:, 2] * self._beta
        return - 2 * alpha * r_

    def _laplacian_spf(self, r, alpha):
        N, d = r.shape
        return -2 * (d - 1 + self._beta) * alpha * N

    def _laplacian(self, r, alpha):
        grad2 = np.sum(self._laplacian_spf(r, alpha))
        grad = self._gradient(r, alpha)
        laplacian = grad2 + np.sum(grad * grad)
        return laplacian

    def _kinetic_energy(self, r, alpha):
        return -0.5 * self._laplacian(r, alpha)

    def _potential_energy(self, r):
        r2 = r * r
        r2[:, 2] *= self._gamma2
        return self._halfomega2 * np.sum(r2)

    def grad_alpha(self, r, alpha):
        """Gradient of wave function w.r.t. variational parameter alpha"""
        r_ = r.copy()
        r_[:, 2] = r_[:, 2] * self._beta
        return -np.sum(r_ * r_)

    def local_energy(self, r, alpha):
        ke = self._kinetic_energy(r, alpha)
        pe = self._potential_energy(r)
        return ke + pe

    def drift_force(self, r, alpha):
        return 2 * self._gradient(r, alpha)
