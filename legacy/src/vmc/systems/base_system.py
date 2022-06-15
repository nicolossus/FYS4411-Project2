#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


class System:
    """Base class for creating a system.

    The implementation requires the wave function to be in the log domain.
    """

    def __init__(self):
        pass

    @abstractmethod
    def wf(self, r, alpha):
        """Evaluate the wave function.

        Must be in the log domain. To be overwritten by subclass.
        """

        raise NotImplementedError

    @abstractmethod
    def potential(self):
        """Potential energy function.

        To be overwritten by subclass.
        """

        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, r, alpha):
        """Probability density for each particle in the system.

        Arguments
        ---------
        r : array_like
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        array_like
            Probability density for each particle
        """

        return jnp.exp(self.logprob(r, alpha))

    @partial(jax.jit, static_argnums=(0,))
    def logprob(self, r, alpha):
        """Log probability density for each particle in the system.

        Given a trial wave function f in the log domain, the function
        calculates the drift force as

                logpdf = log(|f|^2) = 2 log|f|


        Arguments
        ---------
        r : array_like
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        array_like
            Log density for each particle
        """

        return 2. * self.wf(r, alpha)
        #wf = self.wf(r, alpha)
        # return wf * wf
        # return 2 * jnp.abs(wf)

    @partial(jax.jit, static_argnums=(0,))
    def _local_kinetic_energy(self, r, alpha):
        """Evaluate the local kinetic energy.

        Inspired by and almost identical to the implementation in FermiNet [1].

        The local kinetic energy is evaluted in the log domain via

            -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2),

        where f is the wave function.

        References
        ----------
        [1] https://github.com/deepmind/ferminet/blob/02a601268d5dc48cb3cd7846311f25ccc43087e8/ferminet/hamiltonian.py#L62
        """

        n = r.shape[0]
        eye = jnp.eye(n)

        grad_wf = jax.grad(self.wf, argnums=0)
        def grad_wf_closure(r): return grad_wf(r, alpha)
        primal, dgrad_f = jax.linearize(grad_wf_closure, r)

        _, diagonal = lax.scan(
            lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n)

        return -0.5 * jnp.sum(diagonal) - 0.5 * jnp.sum(primal**2)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, r, alpha):
        """Local energy of the system.

        Arguments
        ---------
        r : array_like
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        float
            Local energy
        """

        def ke_closure(r): return self._local_kinetic_energy(r, alpha)
        ke = jnp.sum(jax.vmap(ke_closure)(r))
        #ke = jnp.sum(ke_closure(r))
        #ke = ke_closure(r)

        pe = self.potential(r)

        return ke + pe

    @partial(jax.jit, static_argnums=(0,))
    def drift_force(self, r, alpha):
        """Drift force at each particle's location.

        Given a trial wave function f in the log domain, the function
        calculates the drift force as

                F = 1/f 2 nabla f = 2 nabla log|f|

        F = 1 / psi * 2 * grad psi = 2 * grad log(|psi|)

        Arguments
        ---------
        r : array_like
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        array_like
            Drift force at each particle's position
        """

        grad_wf = jax.grad(self.wf, argnums=0)
        F = 2 * grad_wf(r, alpha)

        return F

    @partial(jax.jit, static_argnums=(0,))
    def grad_alpha(self, r, alpha):
        """Gradient of wave function w.r.t. variational parameter alpha.

        DEV NOTE: When calling this function, use vmap wrapped by jnp.sum

        Arguments
        ---------
        r : array_like
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        float
            Evaluated gradient
        """

        grad_wf_alpha = jax.grad(self.wf, argnums=1)
        return grad_wf_alpha(r, alpha)
