#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import distance

from . import System, WaveFunction


class NIBWF(WaveFunction):
    """Single particle wave function with Gaussian kernel for a Non-Interacting
    Boson (NIB) system in a spherical harmonic oscillator.

    Parameters
    ----------
    N : int
        Number of particles in system
    dim : int
        Dimensionality of system
    omega : float
        Harmonic oscillator frequency
    """

    def __init__(self, N, dim, omega):
        super().__init__(N, dim)
        self._omega = omega

        # precompute
        self._Nd = N * dim
        self._halfomega2 = 0.5 * omega * omega

    def wf(self, r, alpha):
        """Evaluate the trial wave function.

        Parameters
        ----------
        r : np.ndarray, shape=(n_particles, dim)
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        array_like
            Evaluated trial wave function
        """

        return np.exp(-alpha * r * r)

    def wf_scalar(self, r, alpha):
        """Scalar evaluation of the trial wave function"""

        return np.exp(-alpha * np.sum(r * r))

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


class ANIB:
    """Analytical Non-Interacting Boson (ANIB) system.

    Trial wave function:
                psi = exp(-alpha * r**2)
    """

    def __init__(self, N, dim, omega):
        self._N = N
        self._d = dim
        self._omega = omega

        # precompute
        self._Nd = self._N * self._d
        self._halfomega2 = 0.5 * omega * omega

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, r, alpha):
        return jnp.exp(-alpha * r * r)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, r, alpha):
        wf = self(r, alpha)
        return wf * wf

    @partial(jax.jit, static_argnums=(0,))
    def logprob(self, r, alpha):
        wf = self(r, alpha)
        return jnp.log(wf * wf)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, r, alpha):
        locE = self._Nd * alpha + \
            (self._halfomega2 - 2 * alpha * alpha) * jnp.sum(r * r)
        return locE

    @partial(jax.jit, static_argnums=(0,))
    def drift_force(self, r, alpha):
        return -4 * alpha * r

    @partial(jax.jit, static_argnums=(0,))
    def grad_alpha(self, r, alpha):
        return -jnp.sum(r * r)

    @property
    def N(self):
        return self._N

    @property
    def dim(self):
        return self._d


class AIB(WaveFunction):
    """
    Analytical Interacting Boson (AIB) system.

    Trial wave function:
            psi = exp(-alpha * r**2) * exp(sum(u(rij)))
    """

    def __init__(self, N, dim, omega):
        super().__init__(N, dim)
        self._omega2 = omega * omega
        self._Nd = N * dim
        self._triu_indices = np.triu_indices(N, 1)

    def prepare_handy_values(self, r):
        self.dist_mat = self.distance_matrix(r)
        dist_vec = []
        for i, j in zip(*self._triu_indices):
            dist_vec.append(self.dist_mat[i, j])
        self.dist_vec = np.array(dist_vec)

    def wf(self, r, alpha):
        """Finds value of wave function.
        Parameters
        ----------
        r           :   np.ndarray, shape=(n_particles, dim)
        alpha       :   float

        Returns
        ---------
        array_like  :   np.ndarray, shape=(n_particles, dim)
                    single particle wfs order in dim-dimensional arrays,
                    all scaled by the Jastrow factor.
        """
        f = self.f(r)
        return np.exp(-alpha * np.sum(r * r)) * f

    def wf_scalar(self, r, alpha):
        """Finds scalar value of the wave function.
        Parameters:
        -----------
        r          : np.ndarray, shape=(n_particles, dim)
        alpha      : float
        Returns:
        -----------
        wf_scalar  : float
                    sum of all single particle wfs.
        """
        distance_vector = self.distance_vector(r)
        u_vector = self.u(distance_vector)
        return np.exp(-alpha * np.sum(r * r) + np.sum(u_vector))

    def distance_matrix(self, r, a=0.00433):
        """Finds distances between particles in n_particles times n_particles matrix.
        Parameters:
        ----------
        r               :   np.ndarray, shape=(n_particles, dim)

        Returns:
        ----------
        distance_matrix : np.ndarray, shape=(n_particles, n_particles)
        """
        distance_matrix = distance.cdist(r, r, metric="Euclidean")
        distance_matrix = np.where(distance_matrix < a, 0, distance_matrix)
        return distance_matrix

    def distance_vector(self, r):
        """Orders all distances between the particles in a vector of length
        len(self._triu_indices[0])
        Parameters:
        -----------
        r           : np.ndarray, shape=(n_particles, dim)
        Returns:
        -----------
        distance_vector : np.ndarray, shape=(len(_triu_indices[0]),)
        """
        distance_matrix = self.distance_matrix(r)
        distance_vector = []
        for i, j in zip(*self._triu_indices):
            distance_vector.append(distance_matrix[i, j])
        return np.array(distance_vector)

    def unit_matrix(self, r):
        """Orders distances between particles in a 3-dimensional matrix.

        Parameters
        ----------
        r : np.ndarray, shape=(n_particles, dim)

        Returns:
        ----------
        unit_matrix : np.ndarray, shape=(n_particles, n_particles, dim)
            Matrix filled with the unit vectors between all particles
        """
        N = self._N
        d = self._d
        unit_matrix = np.zeros((N, N, d))
        for i, j in zip(*self._triu_indices):
            rij = np.linalg.norm(r[i] - r[j])
            #print("Ri: ", r[i])
            #print("Rj: ", r[j])
            #print("rij: ", rij)
            upper_unit_vector = (r[i] - r[j]) / rij
            unit_matrix[i, j, :] = upper_unit_vector
            unit_matrix[j, i, :] = -upper_unit_vector
            rij = np.linalg.norm(r[i] - r[j])
            upper_unit_vector = (r[i] - r[j]) / rij
            unit_matrix[i, j, :] = upper_unit_vector
            unit_matrix[j, i, :] = -upper_unit_vector
        return unit_matrix

    def unit_matrix_faster(self, r):

        N = self._N
        d = self._d
        unit_matrix = np.zeros((N, N, d))
        axis = 1
        i, j = np.triu_indices(N, 1)
        M = len(i)
        distance_matrix = self.distance_matrix(r)
        distance_matrix_select = np.reshape(distance_matrix[i, j], (M, 1))
        unit_matrix[i, j, :] = (r[i, :] - r[j, :]) / distance_matrix_select
        unit_matrix[j, i, :] = (r[j, :] - r[i, :]) / distance_matrix_select
        return unit_matrix, distance_matrix

    def _gradient_jastrow(self, r, alpha, a=0.00433):
        # Correlation gradient
        N = r.shape[0]
        axis = r.ndim - 1

        # Generate indices
        ii, jj = np.meshgrid(range(N), range(N), indexing='ij')
        i, j = (ii != jj).nonzero()

        # Compute quantities
        rij = r[i] - r[j]
        dij = np.linalg.norm(rij, ord=2, axis=axis)
        du_tmp = a / (dij * dij * (dij - a))
        du_dij = rij * du_tmp[:, np.newaxis]

        # Sum contributions
        _, indices = np.unique(i, return_counts=True)
        row_summing = np.append([0], np.cumsum(indices))[:-1]
        grad_jastrow = np.add.reduceat(du_dij, row_summing, axis=0)

        return du_dij, grad_jastrow

    def u(self, r, a=0.00433):
        """Vector of exponent values.
        Parameters:
        -----------
        distance_vector         :   shape = (len(self._triu_indices[0]),)
        a                       :   float
                    hard sphere diameter
        Returns:
        -----------
        u_vec                   :   shape = (len(self._triu_indices[0]),)
                    vector containing values of u, (ln(f))
        """
        distance_vector = self.distance_vector(r)
        u = np.where(distance_vector < a, -1e20,
                     np.log(1 - a / distance_vector))
        return u

    def f(self, r, a=0.00433):
        """Jastrow factor
        Parameters:
        -----------
        r           : np.ndarray, shape=(n_particles, dim)
        a           : float,
                    hard sphere diameter
        Returns:
        -----------
        float,
                    product of all interactions (1-a/r)

        """
        N = self._N
        i, j = np.triu_indices(N, 1)
        axis = 1
        q = r[i] - r[j]
        rij = np.linalg.norm(q, ord=2, axis=axis)
        f = 1 - a / rij * (rij > a)
        return np.sum(np.log(f))

    def dudr(self, r, a=0.00433):
        """Derivative of u w.r.t. distances between particles.
        Parameters:
        -----------
        r           : np.ndarray, shape=(n_particles, dim)
        a           : float,
                    hard sphere diameter
        Returns:
        ---------
        dudr        : np.ndarray, shape=(n_particles, n_particles, dim)
        """
        N = self._N
        d = self._d
        distance_matrix = self.distance_matrix(r)
        scaler = np.zeros((N, N, 1))
        for i, j in zip(*self._triu_indices):
            rij = distance_matrix[i, j]
            scaler[i, j] = a / (rij * rij - a * rij)
            scaler[j, i] = scaler[i, j]
        unit_matrix = self.unit_matrix(r)
        dudr = unit_matrix * scaler
        return dudr

    def dudr_faster(self, r, a=0.00433):
        """Derivative of u w.r.t. distances between particles.
        Parameters:
        -----------
        r           : np.ndarray, shape=(n_particles, dim)
        a           : float,
                    hard sphere diameter
        Returns:
        ---------
        dudr        : np.ndarray, shape=(n_particles, n_particles, dim)
        """
        N = self._N
        d = self._d
        unit_matrix, distance_matrix = self.unit_matrix_faster(r)
        scaler = np.zeros((N, N, 1))
        i, j = np.triu_indices(N, 1)
        M = len(np.triu_indices(N, 1)[0])
        # for i, j in zip(*self._triu_indices):
        rij = np.reshape(distance_matrix[i, j], (M, 1))
        scaler[i, j] = a / (rij * rij - a * rij)
        scaler[j, i] = scaler[i, j]

        dudr = unit_matrix * scaler
        return dudr

    def d2udr2(self, r, a=0.00433):
        """Returns second derivative of u wrt all distances.
        Parameters:
        -----------
        r           : np.ndarray, shape=(n_particles, dim)
        a           : float,
                    hard sphere diameter

        Returns:
        -----------
        d2udr2      : np.ndarray, shape=(n_particles, n_particles)
        """
        N = self._N
        distance_matrix = self.distance_matrix(r)
        d2udr2 = np.zeros((N, N))
        for i, j in zip(*self._triu_indices):
            rij = distance_matrix[i, j]
            d2udr2[i, j] = a * (a - 2 * rij) / \
                (rij * rij * (a - rij) * (a - rij))
        return d2udr2

    def fourth_term(self, r, a=0.00433):
        """Compute terms u'' + 2/r*u'
        Parameters:
        -----------
        r           : np.ndarray, shape=(n_particle, dim)
        a           : float,
                    hard sphere diameter
        Returns:
        -----------
        fourth_term : np.ndarray, shape=(n_particles, n_particles)
        """
        N = self._N
        distance_matrix = self.distance_matrix(r)
        fourth_term = np.zeros((N, N))
        for i, j in zip(*self._triu_indices):
            rij = distance_matrix[i, j]
            fourth_term[i, j] = -a**2 / (rij**2 * (rij - a)**2)
            fourth_term[j, i] = fourth_term[i, j]
        return fourth_term

    def u_der(self, r, a=0.00433):
        distance_matrix = self.distance_matrix(r)
        N = self._N
        u_der = np.zeros((N, N, 1))

        for i, j in zip(*np.triu_indices(N, 1)):
            rij = distance_matrix[i, j]
            u_der[i, j, 0] = a / (rij**2 - a * rij)
        return u_der

    def fourth_term_alternative(self, r, a=0.004333):
        N = self._N
        distance_matrix = self.distance_matrix(r)
        u_der = self.u_der(r)
        u_der2 = self.d2udr2(r)
        fourth_term = np.zeros((N, N))
        for i, j in zip(*np.triu_indices(N, 1)):
            rij = distance_matrix[i, j]
            fourth_term[i, j] = u_der2[i, j] + 2 * u_der[i, j] / rij
            fourth_term[j, i] = fourth_term[i, j]
        return fourth_term

    def fourth_term_last(self, r, alpha, a=0.00433):
        fourth_term = 0
        N = self._N
        for i in range(N):
            for j in range(N):
                pass  # fourth_term += (a**2-2*)
        return 0

    def local_energy(self, r, alpha, a=0.00433):
        """Compute the local energy.

        Parameters
        ----------
        r               :   np.ndarray, shape=(n_particles, dim)
            Particle positions
        alpha           :   float
            Variational parameter

        Returns
        ----------
        local_energy    :   float
            Computed local energy

        Intermediates
        -------------
        non_interacting_part : float
            laplacian in non-interacting case
        second_term          : float
            dot product between all r[i, :] and dudr[i,:] where dudr has had its
            first axis summed out.
        third_term           : float
            dot product between all dudr[i,j, :] and dudr[i, :, :]
        fourth_term          : float
            sum of all u'' + 2/r*u'
        """
        dudr = self.dudr_faster(r)
        fourth_term_vals = self.fourth_term_alternative(r)

        non_interacting_part = self._Nd * alpha + \
            (0.5 * self._omega2 - 2 * alpha * alpha) * np.sum(r * r)
        print("NI part: ", non_interacting_part)

        second_term = -0.5 * (-4) * alpha * \
            np.sum(np.diag(np.inner(r, np.sum(dudr, axis=1))))
        #second_term_test = -4*alpha*np.sum(np.diag(np.inner(r, np.sum(dudr, axis=0))))
        print("Second term: ", second_term)
        #assert abs(second_term-second_term_test) <1e-12
        # if (abs(second_term)>abs(non_interacting_part)):
        #    print("Second term fuckup: ", second_term)
        #print("Second term: ", second_term)

        third_term = -0.5 * \
            np.einsum("ijk, ajk -> ", dudr, dudr, optimize="greedy")
        print("Third term: ", third_term)
        # if (abs(third_term)>abs(non_interacting_part)):
        #    print("Third term fuckup: ", third_term)
        #    print("log|wf|^2: ", self.logprob(r, alpha))
        #print("Third term: ", third_term)

        fourth_term = -0.5 * np.sum(fourth_term_vals)
        print("Fourth term: ", fourth_term)
        # if (abs(fourth_term)>abs(non_interacting_part)):
        #    print("Fourth term fuckup: ", fourth_term)
        #    print("log|wf|^2: ", self.logprob(r, alpha))
        #print("Fourth term: ", fourth_term)
        #print("Third and fourth term added: ", non_interacting_part-third_term-fourth_term)
        local_energy = non_interacting_part + second_term + third_term + \
            fourth_term  # + second_term + third_term + fourth_term
        # print(local_energy)
        return local_energy

    def drift_force(self, r, alpha):
        dudr = self.dudr_faster(r)
        drift_force = -4 * alpha * r + np.sum(dudr, axis=0)
        return drift_force

    def grad_alpha(self, r, alpha):
        return -np.sum(r * r)

    def second_inner(self, r, alpha):
        dudr = self.dudr(r)
        start = time.time()
        second_term = -4 * alpha * \
            np.sum(np.diag(np.inner(r, np.sum(dudr, axis=0))))
        end = time.time()
        return second_term, end - start

    def second_term_for(self, r, alpha):
        second_term = 0
        scaled_unit_matrix = self.dudr(r)
        start = time.time()
        scaled_unit_matrix = np.sum(scaled_unit_matrix, axis=0)
        for i in range(self._N):
            second_term += np.dot(r[i, :], scaled_unit_matrix[i, :])
        second_term = second_term * (-4 * alpha)
        end = time.time()
        return second_term, end - start

    def second_term_for_double(self, r, alpha, a=0.00433):
        second_term = 0
        N = self._N
        d = self._d
        for k in range(N):
            rk = r[k, :]
            sum_jastrow = np.zeros(d)
            for j in range(N):
                if j == k:
                    pass
                else:
                    rj = r[j, :]
                    rkj = np.linalg.norm(rk - rj)
                    sum_jastrow += (rk - rj) * a / (rkj**2 * (rkj - a))
            second_term += -4 * alpha * np.dot(rk, sum_jastrow)
        return second_term

    def third_term_double_for(self, scaled_unit_matrix):
        val = 0
        N = self._N
        start = time.time()
        for i in range(N):
            row = scaled_unit_matrix[i, :, :]
            for element in row:
                val += np.sum(element * row)
        end = time.time()
        return val, end - start

    def third_term_triple_for(self, scaled_unit_matrix):
        third_term = 0
        start = time.time()
        for i in range(self._N):
            for j in range(self._N):
                for k in range(self._N):
                    third_term += np.dot(
                        scaled_unit_matrix[i, j, :], scaled_unit_matrix[i, k, :])
        end = time.time()
        return third_term, end - start

    def test_terms_in_lap(self, r, scaled_unit_matrix, alpha):
        dudr_jastrow, _gradient_jastrow = self._gradient_jastrow(r, alpha)
        print("Second terms: ")
        second_for = self.second_term_for(r, alpha)
        print("Second for val: {}, time: {}".format(
            second_for[0], second_for[1]))
        second_inner = self.second_inner(r, alpha)
        print("Second inner val: {}, time: {}".format(
            second_inner[0], second_inner[1]))
        dudr = np.sum(scaled_unit_matrix, axis=0)
        start = time.time()
        second_einsum = -4 * alpha * \
            np.einsum("nd,nd->", r, dudr, optimize="greedy")
        end = time.time()
        print("Second einsum val: {}, time: {}".format(
            second_einsum, end - start))
        start = time.time()
        second_double_for = self.second_term_for_double(r, alpha)
        end = time.time()
        print(
            f"Second term double: {second_double_for} with time: {end-start}")
        print("Third terms: ")
        triple_for = self.third_term_triple_for(scaled_unit_matrix)
        print("Triple for val: {}, time: {}".format(
            triple_for[0], triple_for[1]))
        double_for = self.third_term_double_for(scaled_unit_matrix)
        print("Double for val: {}, time: {}".format(
            triple_for[0], triple_for[1]))
        start = time.time()
        third_einsum = np.einsum(
            "ijk, ajk -> ", scaled_unit_matrix, scaled_unit_matrix, optimize="greedy")
        end = time.time()
        print("Einsum val: {}, time: {}".format(third_einsum, end - start))

        fourth_term = self.fourth_term(r)
        fourth_term_alternative = self.fourth_term_alternative(r)

        print("Fourth term: ", np.sum(fourth_term))
        print("Fourth term alternative: ", np.sum(fourth_term_alternative))
        #start = time.time()
        #third_jastrow = np.einsum()
        #end = time.time()


class LogNIB(System):
    """
    Non-Interacting Boson (NIB) system in log domain.

    Trial wave function:
                psi = -alpha * r**2
    """

    def __init__(self, omega):
        super().__init__()
        self._omega2 = omega * omega

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):

        return -alpha * r * r

    '''
    @partial(jax.jit, static_argnums=(0,))
    def wf_scalar(self, r, alpha):

        return -alpha * jnp.sum(r * r)
    '''

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        return 0.5 * self._omega2 * jnp.sum(r * r)


class LogIB(System):

    def __init__(self, omega):
        super().__init__()
        self._omega2 = omega * omega

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):

        return self._single(r, alpha) + self._correlation(r)

    @partial(jax.jit, static_argnums=(0,))
    def _single(self, r, alpha):

        return -alpha * jnp.sum(r * r)

    @partial(jax.jit, static_argnums=(0,))
    def f(self, r, a=0.0043):
        N = r.shape[0]
        i, j = np.triu_indices(N, 1)
        axis = r.ndim - 1

        rij = jnp.linalg.norm(r[i] - r[j], ord=2, axis=axis)
        f = 1 - self._a / rij * (rij > self._a)
        return jnp.sum(jnp.log(f))

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        return 0.5 * self._omega2 * jnp.sum(r * r)
