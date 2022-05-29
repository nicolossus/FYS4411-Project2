#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from abc import abstractmethod

import numpy as np

from ..utils import advance_PRNG_state
from .base_sampler import BaseVMC
from .state import State


class RWM(BaseVMC):

    def __init__(self, wavefunction, rng=None):
        super().__init__(wavefunction,
                         inference_scheme='Random Walk Metropolis',
                         rng=rng)

    def step(self, state, alpha, seed, scale=1.0):
        """One step of the random walk Metropolis algorithm

        Parameters
        ----------
        state : vmc.State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        scale : float
            Scale of proposal distribution. Default: 1.0

        Returns
        -------
        new_state : vmc.State
            The updated state of the system.
        """

        N, dim = state.positions.shape

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Sample proposal positions, i.e., move walkers
        proposals = rng.normal(loc=state.positions, scale=scale)

        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Compute proposal log density
        logp_proposal = self._logp_fn(proposals, alpha)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = self._logp_fn(new_positions, alpha)

        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state
