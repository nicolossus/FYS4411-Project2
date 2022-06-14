import numpy as np
from numpy.random import default_rng

from . import BaseRBM



class RBMWF(BaseRBM):
    def __init__(self, n_particles, dim, n_hidden, rng=None, seed=0, scale=0.5, loc=0.0, omega=1.0, hard_sphere_diameter=0.00433):
        """
        Initiates biases and weights of one visible and one hidden layer,
        in addition to making some variables.

        Parameters
        ---------
        n_particles : int, number of particles
        dim         : int, number of dimensions
        n_hidden    : int, number of hidden nodes
        rng         : PRNG
        seed        : seed number to give the PRNG
        scale       : standard deviation of the initialization of the biases
                        and weights. (Normal distribution)
        loc         : mean of initialization (Normal distribution)
        omega       : angular frequency of HO potential
        """

        super().__init__(n_particles, dim, n_hidden, rng=rng, seed=seed, scale=scale, loc=loc)
        self._sigma2 = 1.0
        self._omega2 = omega*omega
        self._hard_sphere_diameter = hard_sphere_diameter

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
        gaussian = 0.25*(r-self._a)**2
        Q = self._Q(r)
        return -np.sum(gaussian)+0.5*np.sum(np.log(Q))


    def _gradient(self, r):
        """
        Calculates the gradient of the log domain wave function.

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        np.ndarray(shape=(n_particles, dim)) containing the gradients
        wrt the positions array, r.
        """
        gaussian_grad = -0.5*(r-self._a)/self._sigma2
        #print("Gaussain grad: ", gaussian_grad)
        denom = self._denominator(r)
        #print("denom: ", denom)
        W_denom = (self._W ) / denom
        #print("Shape W_denom: ", W_denom.shape)
        #print("0.5sum(W_denom)", 0.5*np.sum(W_denom, axis=2))
        prod_term = 0.5*np.sum(W_denom, axis=2) #/self._sigma2
        #print("Shape prod_term: ", prod_term.shape)
        return gaussian_grad + prod_term

    def _log_laplacian(self, r):
        """
        Calculates the laplacian of the log domain wave function.

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        scalar containing the sum the laplacian of the log domain wave
        function.
        """
        gaussian_grad2 = -0.5
        Q = self._Q(r)
        numerator = np.exp(self._b + np.einsum("ij, ijk->k", r, self._W)/self._sigma2)
        Wsquared = self._W**2 # (n, d, h)
        # Summing over all dimensions for the
        #num = np.transpose((numerator/(Q*Q)), axes=(0, 2, 1))
        #print("Shape num: ", num.shape)
        prod_term2 = 0.5*Wsquared*((numerator)/(Q**2))
        prod_term2 = np.sum(prod_term2, axis=2)
        return np.sum((gaussian_grad2 + prod_term2))

    def _laplacian(self, r):
        """
        Calculates the laplacian of the wave function in linear domain.

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        Float
        """
        grad = self._gradient(r)
        grad2 = self._log_laplacian(r)
        gradient_term = grad*grad
        return np.sum(gradient_term) + np.sum(grad2)

    def _kinetic_energy(self, r):
        """
        Calculates the kinetic energy.

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        Scalar
        """
        kinetic_energy = -0.5*self._laplacian(r)
        #print("Kinetic energy: ", kinetic_energy)
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
        f = 1 - self._hard_sphere_diameter / rij * (rij > self._hard_sphere_diameter)

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
        #Vint = self._correlation(r)
        Vtrap = 0.5 * self._omega2 * np.sum(r*r)
        #print("Potential energy: ", Vtrap)
        return Vtrap #+ Vint


    def _softplus(self, x):
        """Softplus activation function.

        Computes the element-wise function
                softplus(x) = log(1 + e^x)
        """
        return np.logaddexp(x, 0)


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
        Q =  self._softplus(self._b + np.einsum("ij, ijk->k", r, self._W)/self._sigma2)
        print("New Q: ", np.exp(Q))
        Q = 1 + np.exp(self._b + np.einsum("ij, ijk->k", r, self._W)/self._sigma2)
        print("Old Q: ", Q)

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
        denominator = self._softplus(-self._b - np.einsum("ij, ijk->k", r, self._W)/self._sigma2)
        print("New denom: ", np.exp(denominator))
        denominator = 1 + np.exp(-self._b - np.einsum("ij, ijk->k", r, self._W)/self._sigma2)
        print("Old denom: ", denominator)
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
        #print("Potential E: ", potential_energy)
        #print("Kinetic energy: ", kinetic_energy)
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
        grad_a = (r-self._a)*0.5
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
        grad_b = 0.5 / (denom)
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
        r = r.reshape(N, dim, 1)
        grad_weights = (0.5*r/denom)
        return grad_weights

    def drift_force(self, r):
        """
        Calculates the drift force of the neural network wave function.

        Parameters
        ---------
        r       : np.ndarray(shape=(n_particles, dim))

        Returns
        ---------
        np.ndarray(shape=(n_particles, dim)) containing the elements
        of the drift force.
        """

        return 2*self._gradient(r)
