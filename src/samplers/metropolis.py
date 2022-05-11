#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from abc import abstractmethod

import numpy as np

from .base_sampler import BaseVMC
from .pool_tools import advance_PRNG_state
from .state import State


class Metropolis(BaseVMC):

    def __init__(self, wavefunction, rng=None):
        super().__init__(wavefunction, inference_scheme='metropolis', rng=rng)

    def step(self, state, alpha, seed, scale=1.0):
        """One step of the random walk Metropolis algorithm

        Parameters
        ----------
        state : State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        scale : float
            Scale of proposal distribution. Default: 1.0
        """
        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)
        N, dim = state.positions.shape
        # Sample proposal positions, i.e., move walkers
        proposals = rng.normal(loc=state.positions, scale=scale)
        #new_positions = state.positions
        # Sample log uniform rvs

        log_unif = np.log(rng.random())
        #log_unif = np.log(rng.random(size=state.positions.shape))

        # Compute proposal log density
        logp_proposal = self._logp_fn(proposals, alpha)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp
        # print(accept)
        # print(log_unif)
        # print(logp_proposal)
        # print(state.logp)
        # Where accept is True, yield proposal, otherwise keep old state
        #new_positions = np.where(accept, proposals, state.positions)
        new_positions = proposals if accept else state.positions

        new_logp = self._logp_fn(new_positions, alpha)

        #new_n_accepted = state.n_accepted + np.sum(accept)
        new_n_accepted = state.n_accepted + accept

        new_logp = self._logp_fn(new_positions, alpha)

        new_delta = state.delta + 1

        # Create new state

        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state
