#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import numpy as np


class WaveFunction(metaclass=ABCMeta):
    """Base class for constructing analytical trial wave functions.

    Parameters
    ----------
    N : int
        Number of particles in system
    dim : int
        Dimensionality of system
    """

    def __init__(self, N, dim):

        self._verify_constructor_input(N, dim)

        self._N = N
        self._d = dim

    @abstractmethod
    def wf(self):
        """Evaluate the many body trial wave function.

        Must be overwritten by sub-class.
        """
        raise NotImplementedError

    def _verify_constructor_input(self, N, dim):
        # Check for valid dtype
        if not isinstance(N, int):
            msg = "The number of particles in the system must be passed as int"
            raise TypeError(msg)
        if not isinstance(dim, int):
            msg = "The dimensionality of the system must be passed as int"
            raise TypeError(msg)

        # Check for valid value
        if not N > 0:
            msg = "The number of particles must be > 0"
            raise ValueError(msg)
        if not 1 <= dim <= 3:
            msg = "Dimensionality must be between 1D, 2D or 3D"
            raise ValueError(msg)

    def pdf(self, *args, **kwargs):
        """Compute the square of the many body trial wave function

        Parameters
        ----------
        *args
            args are passed to the call method
        **kwargs
            kwargs are passed to the call method

        Returns
        -------
        array_like
            The squared trial wave function
        """
        return np.exp(self.logprob(*args, **kwargs))

    def logprob(self, *args, **kwargs):

        return 2. * self.wf(*args, **kwargs) #np.log(self.wf(*args, **kwargs)*self.wf(*args, **kwargs))
        

    @abstractmethod
    def local_energy(self):
        """Compute the local energy.

        Must be overwritten by sub-class.
        """
        raise NotImplementedError

    @abstractmethod
    def drift_force(self):
        """Compute the local energy.

        Must be overwritten by sub-class.
        """
        raise NotImplementedError

    @abstractmethod
    def grad_alpha(self):
        """Compute the local energy.

        Must be overwritten by sub-class.
        """
        raise NotImplementedError

    @property
    def N(self):
        return self._N

    @property
    def dim(self):
        return self._d
