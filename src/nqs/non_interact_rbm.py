#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


class NonInteractRBM:
    """Base class for creating a quantum system where the wave function is
    represented by a gaussian-binary restricted Boltzmann machine.

    The implementation assumes a logarithmic wave function.
    """

    def __init__(self, sigma2=1.):
        self._sigma2 = sigma2

    @partial(jax.jit, static_argnums=(0,))
    def _softplus(self, x):
        """Softplus activation function.

        Computes the element-wise function
                softplus(x) = log(1 + e^x)
        """
        return jnp.logaddexp(x, 0)

    @partial(jax.jit, static_argnums=(0,))
    def _log_rbm(self, r, v_bias, h_bias, kernel):
        """Logarithmic gaussian-binary RBM"""

        # visible layer
        x_v = jnp.linalg.norm(r - v_bias)
        x_v *= -x_v / (2 * self._sigma2)

        # hidden layer
        x_h = self._softplus(h_bias + (r.T @ kernel) / self._sigma2)
        x_h = jnp.sum(x_h, axis=-1)

        return x_v + x_h

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, v_bias, h_bias, kernel):
        """Evaluate the wave function"""
        return 0.5 * self._log_rbm(r, v_bias, h_bias, kernel).sum()

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        """Potential energy function"""
        return 0.5 * jnp.sum(r * r)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, r, v_bias, h_bias, kernel):
        """Probability amplitude"""
        return jnp.exp(self.logprob(r, v_bias, h_bias, kernel))

    @partial(jax.jit, static_argnums=(0,))
    def logprob(self, r, v_bias, h_bias, kernel):
        """Log probability amplitude"""
        return self._log_rbm(r, v_bias, h_bias, kernel).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _local_kinetic_energy(self, r, v_bias, h_bias, kernel):
        """Evaluate the local kinetic energy"""

        n = r.shape[0]
        eye = jnp.eye(n)

        grad_wf = jax.grad(self.wf, argnums=0)

        def grad_wf_closure(r): return grad_wf(r, v_bias, h_bias, kernel)
        primal, dgrad_f = jax.linearize(grad_wf_closure, r)

        _, diagonal = lax.scan(
            lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n)

        return -0.5 * jnp.sum(diagonal) - 0.5 * jnp.sum(primal**2)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, r, v_bias, h_bias, kernel):
        """Local energy of the system"""

        def ke_closure(r): return self._local_kinetic_energy(
            r, v_bias, h_bias, kernel)
        ke = jnp.sum(jax.vmap(ke_closure)(r))
        pe = self.potential(r)

        return ke + pe

    @partial(jax.jit, static_argnums=(0,))
    def drift_force(self, r, v_bias, h_bias, kernel):
        """Drift force at each particle's location"""
        grad_wf = jax.grad(self.wf, argnums=0)
        F = 2 * grad_wf(r, v_bias, h_bias, kernel)
        return F

    @partial(jax.jit, static_argnums=(0,))
    def grad_v_bias(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. visible bias"""
        grad_wf_v_bias = jax.grad(self.wf, argnums=1)
        return grad_wf_v_bias(r, v_bias, h_bias, kernel)

    @partial(jax.jit, static_argnums=(0,))
    def grad_h_bias(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. hidden bias"""
        grad_wf_h_bias = jax.grad(self.wf, argnums=2)
        return grad_wf_h_bias(r, v_bias, h_bias, kernel)

    @partial(jax.jit, static_argnums=(0,))
    def grad_kernel(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. weight matrix"""
        grad_wf_kernel = jax.grad(self.wf, argnums=3)
        return grad_wf_kernel(r, v_bias, h_bias, kernel)


if __name__ == "__main__":

    import numpy as np
    from numpy.random import default_rng

    '''
    # [0.27273297 0.20895334]
    v_bias = jnp.array([-0.02033396, -0.01811685])
    h_bias = jnp.array([-0.0033904, -0.00575564])
    kernel = jnp.array([[1.555187, -1.9820777],
                        [-1.4803927,  0.00444124]])
    r = jnp.array([[0.2, 0.5], [0.3, 0.7]])
    '''

    P = 1
    dim = 1
    nhidden = 2


    rng = default_rng(2113)
    r = rng.standard_normal(size=(P, dim))
    v_bias = rng.standard_normal(size=(P, dim))
    h_bias = rng.standard_normal(size=(nhidden))
    kernel = rng.standard_normal(size=(P, dim, nhidden))

    system = NonInteractRBM()
    print(system.wf(r, v_bias, h_bias, kernel))
    print(system._log_rbm(r, v_bias, h_bias, kernel).sum())
    print(system.logprob(r, v_bias, h_bias, kernel))
    print("")
    print(system.local_energy(r, v_bias, h_bias, kernel))
