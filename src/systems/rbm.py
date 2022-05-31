import numpy as np
from numpy.random import default_rng

from . import BaseRBM



class RBM(BaseRBM):
    def __init__(n_particles, dim, n_hidden, rng=None, seed=0, scale=1.0, loc=0.0, omega=1.0):
        super(RBM, )__init__(n_particles, dim, n_hidden, rng=rng, seed=seed, scale=scale, loc=loc)
        self.sigma2 = scale*scale
        self.omega2 = omega*omega

    def wf(self, r):
        """
        NQS wave function (in log domain) FRBM = |Psi|^2
        Normalization term is left out.
        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Return
        ---------
        A scalar value representing the NQS wave function.
        """
        gaussian = np.sum((r-self.a)/(4*self.sigma2))
        Q = self.Q(r)

        return -gaussian + np.sum(np.log(Q))


    def _gradient(self, r):
        gaussian_grad = -0.5*(r-self._a)/self._sigma2
        denom = self.denominator(r)
        W_denom = self._W/denom
        prod_term = 0.5*np.sum(W_denom, axis=2)/self._sigma2
        return gaussian_grad + prod_term

    def _log_laplacian(self, r):
        gaussian_grad2 = -0.5
        Q = self.Q(r)
        numerator = Q-1
        Wsquared = self._W*self._W # (n, d, h)
        # Summing over all dimensions for the
        prod_term2 = 0.5/(self._sigma2)*np.sum(Wsquared*numerator/(Q*Q))
        return (gaussian_grad2 + prod_term2)/self._sigma2

    def _laplacian(self, r):
        grad = self._gradient(r)
        grad2 = self._log_laplacian(r)
        gradient_term = np.sum(grad*grad)
        return gradient_term + grad2

    def _kinetic_energy(self, r):
        kinetic_energy = -0.5*self._laplacian(r)
        return kinetic_energy

    def _correlation(self, r):
        """
        Jastrow correlation factor in log domain
        Hard-sphere interactions
        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        Scalar corresponding to Jastrow correlation term
        """

        i, j = np.triu_indices(r.shape[0], 1)
        axis = r.ndim - 1
        rij = np.linalg.norm(r[i] - r[j], ord=2, axis=axis)
        f = 1 - self._a / rij * (rij > self._a)

        return np.sum(np.log(f))

    def _potential_energy(self, r):
        """
        Potential of the trap with Jastrow interactions
        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))
        Returns
        ---------
        Scalar corresponding to potential
        """
        Vint = self._correlation(r)
        Vtrap = 0.5 * self._omega2 * np.sum(r*r)
        return Vtrap + Vint


    def _Q(self, r):
        """
        Product term

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Return
        ---------
        An np.ndarray(shape=(n_hidden)) value corresponding to
                    (1 + e^{b+rW/s^2})
        """
        Q = 1 + np.exp(self.b + np.einsum("ij,ijk->k", r, self.W)/self.sigma2)
        return Q

    def _denominator(self, r):
        """
        Denominator in gradient terms
        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Return
        ---------
        np.ndarray(shape=(n_hidden))
        """
        denominator = np.exp(-(self.b + np.einsum("ij,ijk->k", r, self.W)/self.sigma2)) + 1
        return denominator

    def local_energy(self, r):
        """
        Calculates the Hamiltonian of the wave function at positions r

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        Scalar representing the energy of the system
        """
        kinetic_energy = self._kinetic_energy(r)
        potential_energy = self._potential_energy(r)
        return kinetic_energy + potential_energy

    def grad_a(self, r):
        """
        Calculates the gradient with respect to the bias
        of the visible layer.

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        np.ndarray(shape=(n_particles, dim)) containing the gradients
        wrt a.
        """
        grad_a = (r-self._a)/(2*self._sigma2)
        return grad_a

    def grad_b(self, r):
        """
        Calculates the gradient with respect to the bias
        of the hidden layer.

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        np.ndarray(shape=(n_hidden)) containing the gradient wrt b.
        """
        denom = self._denominator(r)
        grad_b = 0.5/denom
        return grad_b

    def grad_weights(self, r):
        """
        Calculates the gradient with respect to the weights
        of the interactions between the visible and hidden
        layers.

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        np.ndarray(shape=(n_particles, dim, n_hidden)) containing all
        the gradients wrt to W.
        """
        N, dim = r.shape
        denom = self._denominator(r)
        n_hidden = len(denom)
        r = r.reshape(N, dim, 1)
        grad_weights = r/denom
        return grad_weights

    def update_parameters(self, gradients):
        """
        Update trainable parameters

        Parameters
        ---------
        gradients : [grad_a, grad_b, grad_weights]

        Updates a, b, weights
        """
        return 0

    def sample(self):
        """
        Perform sampling
        """
        return 0
