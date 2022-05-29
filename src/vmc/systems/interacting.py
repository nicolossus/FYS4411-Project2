#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import System, WaveFunction


class SHOIB(System):

    def __init__(self, omega=1., a=0.00433):
        super().__init__()
        self._omega2 = omega * omega
        self._a = a

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):
        return self._single(r, alpha) + self._correlation(r)

    @partial(jax.jit, static_argnums=(0,))
    def _single(self, r, alpha):
        return -alpha * jnp.sum(r * r)

    @partial(jax.jit, static_argnums=(0,))
    def _correlation(self, r):
        N = r.shape[0]
        i, j = jnp.triu_indices(N, 1)
        axis = r.ndim - 1
        rij = jnp.linalg.norm(r[i] - r[j], ord=2, axis=axis)
        f = 1 - self._a / rij * (rij > self._a)
        return jnp.sum(jnp.log(f))

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        return 0.5 * self._omega2 * jnp.sum(r * r)


class EHOIB(System):

    def __init__(self, a=0.00433):
        super().__init__()
        self._beta = 2.82843
        self._gamma2 = self._beta * self._beta
        self._a = a

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):
        return self._single(r, alpha) + self._correlation(r)

    @partial(jax.jit, static_argnums=(0,))
    def _single(self, r, alpha):
        r2 = jnp.square(r)
        r2 = r2.at[-1].multiply(self._beta)
        return -alpha * jnp.sum(r2)

    @partial(jax.jit, static_argnums=(0,))
    def _correlation(self, r):
        N = r.shape[0]
        i, j = jnp.triu_indices(N, 1)
        axis = r.ndim - 1
        rij = jnp.linalg.norm(r[i] - r[j], ord=2, axis=axis)
        f = 1 - self._a / rij * (rij > self._a)
        return jnp.sum(jnp.log(f))

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        r2 = jnp.square(r)
        r2 = r2.at[:, 2].multiply(self._gamma2)
        return 0.5 * jnp.sum(r2)


class ASHOIB(WaveFunction):

    def __init__(self, N, dim, omega=1., a=0.00433):
        super().__init__(N, dim)
        self._omega2 = omega * omega
        self._a = a

    def wf(self, r, alpha):
        return self._single(r, alpha) + self._correlation(r)

    def wf_vectorized(self, r, alpha):
        return self._single_vectorized(r, alpha) + self._correlation(r)

    def _single(self, r, alpha):
        return -alpha * np.sum(r * r)

    def _single_vectorized(self, r, alpha):
        return -alpha * np.sum(r * r, axis=1)

    def PDF_vectorized(self, r, alpha):
        return np.exp(2 * self.wf_vectorized(r, alpha))

    def _correlation(self, r):
        i, j = np.triu_indices(r.shape[0], 1)
        axis = r.ndim - 1
        rij = np.linalg.norm(r[i] - r[j], ord=2, axis=axis)
        f = 1 - self._a / rij * (rij > self._a)

        return np.sum(np.log(f))

    def _gradient_spf(self, r, alpha):
        # Single particle gradient
        return - 2 * alpha * r

    def _gradient_jastrow(self, r, alpha):
        # Correlation gradient
        N = r.shape[0]
        axis = r.ndim - 1

        # Generate indices
        ii, jj = np.meshgrid(range(N), range(N), indexing='ij')
        i, j = (ii != jj).nonzero()
        #print("Jastrow indices: ", i)
        # print(j)
        # Compute quantities
        rij = r[i] - r[j]
        dij = np.linalg.norm(rij, ord=2, axis=axis)
        du_tmp = self._a / (dij * dij * (dij - self._a))
        du_dij = rij * du_tmp[:, np.newaxis]

        # Sum contributions
        _, indices = np.unique(i, return_counts=True)
        row_summing = np.append([0], np.cumsum(indices))[:-1]
        # print(row_summing)
        grad_jastrow = np.add.reduceat(du_dij, row_summing, axis=0)

        return grad_jastrow

    def _gradient(self, r, alpha):
        # Gather gradients
        grad_spf = self._gradient_spf(r, alpha)
        grad_jastrow = self._gradient_jastrow(r, alpha)
        gradient = grad_spf + grad_jastrow
        return gradient

    def _laplacian_spf(self, r, alpha):
        N, d = r.shape
        return -2 * d * alpha * N

    def _laplacian_jastrow(self, r, alpha):
        # Correlation gradient
        N = r.shape[0]
        axis = r.ndim - 1

        # Generate indices
        ii, jj = np.meshgrid(range(N), range(N), indexing='ij')
        i, j = (ii != jj).nonzero()

        # Compute quantities
        rij = r[i] - r[j]
        dij = np.linalg.norm(rij, ord=2, axis=axis)
        dij_a = dij - self._a
        du_dr = 2 * self._a / (dij * dij * dij_a)
        d2u_dr2 = self._a * (self._a - 2 * dij) / (dij * dij * dij_a * dij_a)
        grad2_tmp = du_dr + d2u_dr2

        # Sum contributions
        _, indices = np.unique(i, return_counts=True)
        row_summing = np.append([0], np.cumsum(indices))[:-1]
        grad2_jastrow = np.add.reduceat(grad2_tmp, row_summing, axis=0)
        return grad2_jastrow

    def _laplacian(self, r, alpha):
        grad2_spf = self._laplacian_spf(r, alpha)
        grad2_jastrow = self._laplacian_jastrow(r, alpha)
        grad2 = np.sum(grad2_spf) + np.sum(grad2_jastrow)
        grad = self._gradient(r, alpha)
        laplacian = grad2 + np.sum(grad * grad)
        return laplacian

    def _kinetic_energy(self, r, alpha):
        return -0.5 * self._laplacian(r, alpha)

    def _potential_energy(self, r):
        return 0.5 * self._omega2 * np.sum(r * r)

    def local_energy(self, r, alpha):
        ke = self._kinetic_energy(r, alpha)
        pe = self._potential_energy(r)
        return ke + pe

    def drift_force(self, r, alpha):
        return 2 * self._gradient(r, alpha)

    def grad_alpha(self, r, alpha):
        """Gradient of wave function w.r.t. variational parameter alpha"""

        return -np.sum(r * r)


class AEHOIB(WaveFunction):

    def __init__(self, N, dim, omega=1., a=0.00433):
        super().__init__(N, dim)
        self._beta = 2.82843
        self._gamma2 = self._beta * self._beta
        self._omega2 = omega * omega
        self._a = a

    def wf(self, r, alpha):
        return self._single(r, alpha) + self._correlation(r)

    def _single(self, r, alpha):
        r2 = r * r
        r2[:, 2] = r2[:, 2] * self._beta
        return -alpha * np.sum(r2)

    def _correlation(self, r):
        i, j = np.triu_indices(r.shape[0], 1)
        axis = r.ndim - 1
        rij = np.linalg.norm(r[i] - r[j], ord=2, axis=axis)
        f = 1 - self._a / rij * (rij > self._a)

        return np.sum(np.log(f))

    def _gradient_spf(self, r, alpha):
        # Single particle gradient
        r_ = r.copy()
        r_[:, 2] = r_[:, 2] * self._beta
        return - 2 * alpha * r_

    def _gradient_jastrow(self, r, alpha):
        # Correlation gradient
        N = r.shape[0]
        axis = r.ndim - 1

        # Generate indices
        ii, jj = np.meshgrid(range(N), range(N), indexing='ij')
        i, j = (ii != jj).nonzero()
        #print("Jastrow indices: ", i)
        # print(j)
        # Compute quantities
        rij = r[i] - r[j]
        dij = np.linalg.norm(rij, ord=2, axis=axis)
        du_tmp = self._a / (dij * dij * (dij - self._a))
        du_dij = rij * du_tmp[:, np.newaxis]

        # Sum contributions
        _, indices = np.unique(i, return_counts=True)
        row_summing = np.append([0], np.cumsum(indices))[:-1]
        # print(row_summing)
        grad_jastrow = np.add.reduceat(du_dij, row_summing, axis=0)

        return grad_jastrow

    def _gradient(self, r, alpha):
        # Gather gradients
        grad_spf = self._gradient_spf(r, alpha)
        grad_jastrow = self._gradient_jastrow(r, alpha)
        gradient = grad_spf + grad_jastrow
        return gradient

    def _laplacian_spf(self, r, alpha):
        N, d = r.shape
        return -2 * (d - 1 + self._beta) * alpha * N
        # grad = -2a*(x + y + b*z)
        # lap = -2a(1 + 1 + b)

    def _laplacian_jastrow(self, r, alpha):
        # Correlation gradient
        N = r.shape[0]
        axis = r.ndim - 1

        # Generate indices
        ii, jj = np.meshgrid(range(N), range(N), indexing='ij')
        i, j = (ii != jj).nonzero()

        # Compute quantities
        rij = r[i] - r[j]
        dij = np.linalg.norm(rij, ord=2, axis=axis)
        dij_a = dij - self._a
        du_dr = 2 * self._a / (dij * dij * dij_a)
        d2u_dr2 = self._a * (self._a - 2 * dij) / (dij * dij * dij_a * dij_a)
        grad2_tmp = du_dr + d2u_dr2

        # Sum contributions
        _, indices = np.unique(i, return_counts=True)
        row_summing = np.append([0], np.cumsum(indices))[:-1]
        grad2_jastrow = np.add.reduceat(grad2_tmp, row_summing, axis=0)
        return grad2_jastrow

    def _laplacian(self, r, alpha):
        grad2_spf = self._laplacian_spf(r, alpha)
        grad2_jastrow = self._laplacian_jastrow(r, alpha)
        grad2 = np.sum(grad2_spf) + np.sum(grad2_jastrow)
        grad = self._gradient(r, alpha)
        laplacian = grad2 + np.sum(grad * grad)
        return laplacian

    def _kinetic_energy(self, r, alpha):
        return -0.5 * self._laplacian(r, alpha)

    def _potential_energy(self, r):
        r2 = r * r
        r2[:, 2] *= self._gamma2
        return 0.5 * self._omega2 * np.sum(r2)

    def local_energy(self, r, alpha):
        ke = self._kinetic_energy(r, alpha)
        pe = self._potential_energy(r)
        return ke + pe

    def drift_force(self, r, alpha):
        return 2 * self._gradient(r, alpha)

    def grad_alpha(self, r, alpha):
        """Gradient of wave function w.r.t. variational parameter alpha"""
        r_ = r.copy()
        r_[:, 2] = r_[:, 2] * self._beta
        return -np.sum(r_ * r_)


if __name__ == "__main__":

    N = 10       # Number of particles
    dim = 3      # Dimensionality
    omega = 1.   # Oscillator frequency

    alpha = 0.5
    r = np.random.rand(N, dim) * 2.0

    wf_a = ASHOIB(N, dim, omega)
    wf_n = SHOIB(omega)

    print("ASHOIB drift force=", wf_a.drift_force(r, alpha))
    print("SHOIB drift force=", wf_n.drift_force(r, alpha))
    print("ASHOIB local_energy=", wf_a.local_energy(r, alpha))
    print("SHOIB local energy=", wf_n.local_energy(r, alpha))
