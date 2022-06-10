#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import expit


class AniRBMwf:
    """Base class for creating a quantum system where the wave function is
    represented by a gaussian-binary restricted Boltzmann machine.

    The implementation assumes a logarithmic wave function.
    """

    def __init__(self, nparticles, dim, hard_shell_diameter=0.0043, sigma2=1.):

        self._N = nparticles
        self._d = dim
        self._a = hard_shell_diameter
        self._sigma2 = sigma2
        self._sigma4 = sigma2 * sigma2
        self._sigma2_factor = 1 / self._sigma2
        self._sigma2_factor2 = 0.5 * 1 / self._sigma2
        self._sigma4_factor2 = 0.5 * 1 / self._sigma4
        self._sigma4_factor = 1 / self._sigma4

    def _softplus(self, x):
        """Softplus activation function.

        Computes the element-wise function
                softplus(x) = log(1 + e^x)
        """
        return np.logaddexp(x, 0)

    def _log_rbm(self, r, v_bias, h_bias, kernel):
        """Logarithmic gaussian-binary RBM"""

        # visible layer
        x_v = np.linalg.norm(r - v_bias)
        #x_v = r-v_bias
        x_v *= -x_v * self._sigma2_factor2

        # hidden layer
        x_h = self._softplus(h_bias + (r.T @ kernel) * self._sigma2_factor)
        x_h = np.sum(x_h, axis=-1)

        return x_v + x_h

    def wf(self, r, v_bias, h_bias, kernel):
        """Evaluate the wave function"""
        return self._log_rbm(r, v_bias, h_bias, kernel)

    def potential(self, r):
        """Potential energy function"""
        # HO trap
        v_trap = 0.5 * np.sum(r * r)
        """
        # Interaction
        r_cpy = r.copy().reshape(self._N, self._d)
        r_dist = np.linalg.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)
        #print(r_dist)
        triu_idxs = np.triu_indices(self._N, k=1)
        distances = r_dist[triu_idxs]
        f = 1 / distances * (distances > self._a)
        #v_int = np.sum(np.triu(1 - self._a / r_dist, k=1))
        v_int = np.sum(f)
        """
        #print(v_int)

        return v_trap #+ v_int

    def pdf(self, r, v_bias, h_bias, kernel):
        """Probability amplitude"""
        return np.exp(self.logprob(r, v_bias, h_bias, kernel))

    def logprob(self, r, v_bias, h_bias, kernel):
        """Log probability amplitude"""
        return 2*self._log_rbm(r, v_bias, h_bias, kernel)

    def _grad_wf(self, r, v_bias, h_bias, kernel):
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = -(r - v_bias) + kernel @ _expit
        #print("SHape gr: ", gr.shape)
        gr *= self._sigma2_factor
        return gr

    def _laplace_wf(self, r, v_bias, h_bias, kernel):
        """
        Trial 1
        """
        _expit_pos = expit(-h_bias - (r @ kernel) * self._sigma2_factor)
        _exp = np.exp(h_bias + (r @ kernel) * self._sigma2_factor)
        kernel2 = kernel * kernel
        _expit_pos2 = _expit_pos * _expit_pos
        _exp_prod = _exp[:, np.newaxis] @ _expit_pos2[:, np.newaxis].T
        gr = -self._sigma2_factor + self._sigma4_factor * kernel2 @ _exp_prod
        return gr

    def _laplace_wf2(self, r, v_bias, h_bias, kernel):
        """
        Trial 2
        """
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        _expos = expit(-h_bias - (r @ kernel) * self._sigma2_factor)
        kernel2 = kernel * kernel
        exp_prod = _expos[:, np.newaxis] @ _expit[:, np.newaxis].T
        exp_prod = _expos * _expit
        gr = -self._sigma2_factor + self._sigma4_factor * kernel2 @ exp_prod
        #print("Shape laplace: ", gr.shape)

        #print(kernel2 @ exp_prod)
        return gr

    def _local_kinetic_energy(self, r, v_bias, h_bias, kernel):
        """Evaluate the local kinetic energy"""

        # where to sum?

        #_laplace = self._laplace_wf2(r, v_bias, h_bias, kernel)
        _laplace = self._laplace_wf2(r, v_bias, h_bias, kernel).sum()
        _grad = self._grad_wf(r, v_bias, h_bias, kernel)
        _grad2 = np.sum(_grad * _grad)
        #print("Laplcae: ", -0.5*_laplace)
        #print("grad2: ",-0.5* _grad2)
        #print("Laplace + grad2 : ", -0.5*(_laplace + _grad2))
        #_grad2 = _grad * _grad

        return -0.5 * (_laplace + _grad2)
        # return -0.5 * (_laplace + _grad2)

    def local_energy(self, r, v_bias, h_bias, kernel):
        """Local energy of the system"""
        ke = self._local_kinetic_energy(r, v_bias, h_bias, kernel)
        pe = self.potential(r)
        return ke + pe

    def drift_force(self, r, v_bias, h_bias, kernel):
        """Drift force at each particle's location"""
        F = 2 * self._grad_wf(r, v_bias, h_bias, kernel)
        return F

    def grad_v_bias(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. visible bias"""
        gr = (r - v_bias) * self._sigma2_factor
        return gr#.sum()

    def grad_h_bias(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. hidden bias"""
        gr = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        return gr#.sum()

    def grad_kernel(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. weight matrix"""
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = self._sigma2_factor * r[:, np.newaxis] @ _expit[:, np.newaxis].T
        return gr#.sum()


if __name__ == "__main__":
    r = np.array([0.30471707975443135, -1.0399841062404955,
                 0.7504511958064572, 0.9405647163912139])
    v_bias = np.array([-0.00631753,  0.01129719, -0.001397, -0.01849913])
    h_bias = np.array([0.00869276, -0.00643394])
    kernel = np.array([[-0.40775875,  0.08298116],
                       [-0.36875534,  0.03443719],
                       [0.40923255, -0.04661963],
                       [-1.21311022,  0.80609878]])

    system = AniRBMwf()
    print("wf eval:", system.wf(r, v_bias, h_bias, kernel))
    print("logprob:", system.logprob(r, v_bias, h_bias, kernel))
    print("grad v_bias", system.grad_v_bias(r, v_bias, h_bias, kernel))
    print("grad h_bias", system.grad_h_bias(r, v_bias, h_bias, kernel))
    print("grad kernel", system.grad_kernel(r, v_bias, h_bias, kernel))
    print("drift force:", system.drift_force(r, v_bias, h_bias, kernel))
    print("local energy:", system.local_energy(r, v_bias, h_bias, kernel))

    '''
    JAX:
    wf eval: 0.10676813669683652
    logprob: 0.21353627339367304
    grad v_bias 0.4853326778558034
    grad h_bias 0.5158698886876459
    grad kernel 0.4930420712853883
    drift force: [-0.4031509   0.94078729 -0.63485155 -0.85867653]
    local energy: 1.8890100437817727

    Analytical:
    wf eval: 0.10676813669683638
    logprob: 0.21353627339367276
    grad v_bias 0.4853326778558035
    grad h_bias 0.5158698886876458
    grad kernel 0.49304207128538846
    drift force: [-0.4031509   0.94078729 -0.63485155 -0.85867653]
    local energy: 1.8890100437817727

    energy is right
    '''
