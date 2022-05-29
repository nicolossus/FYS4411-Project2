#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as scipy
from jax import grad, jit, lax, pmap, random, vmap

from .state_jax import Config, State

jax.config.update("jax_enable_x64", True)


def _init(initial_positions, wf, alpha, scale):
    logp_fn = wf.logpdf
    state = State(initial_positions, logp_fn(initial_positions, alpha), 0)
    config = Config(alpha, scale, logp_fn, wf.locE)  # wf.locE
    return state, config


def _metropolis_sampler(rng_key, n_samples, initial_state, config):
    """Metroplis sampling loop"""

    @jit
    def metropolis_step(state, rng_key):
        """One step of the Metroplis algorithm"""
        # Sample proposal positions and log uniform rvs
        key_proposal, key_accept = random.split(rng_key)
        move = random.normal(key_proposal,
                             shape=state.positions.shape)
        log_u = jnp.log(random.uniform(key_accept,
                                       shape=state.positions.shape)
                        )

        # Move particles and compute log density
        proposals = state.positions + move * config.scale
        logp_proposal = config.logp_fn(proposals, config.alpha)

        # Metroplis acceptance criterion
        do_accept = log_u < logp_proposal - state.logp

        # where do_accept is True yield proposal, otherwise keep old state
        new_positions = jnp.where(do_accept, proposals, state.positions)
        new_logp = jnp.where(do_accept, logp_proposal, state.logp)
        new_n_accepted = state.n_accepted + jnp.sum(do_accept)

        new_state = State(new_positions, new_logp, new_n_accepted)
        # Compute local energy
        local_energy = config.locE(new_positions, config.alpha)

        return new_state, local_energy

    rng_keys = random.split(rng_key, n_samples)
    final_state, energies = lax.scan(metropolis_step, initial_state, rng_keys)

    acc_rate = final_state.n_accepted / n_samples
    #print("Acceptance rate:", acc_rate)

    return energies


def vmc_jax(seed, n_samples, n_chains, initial_positions, wf, scale, alpha):
    rng_seed = random.PRNGKey(seed)
    base_keys = random.split(rng_seed, n_chains)

    initial_state, config = _init(initial_positions, wf, alpha, scale)

    run_fn = vmap(_metropolis_sampler,
                  in_axes=(0, None, None, None),
                  out_axes=0
                  )

    energies = run_fn(base_keys,
                      n_samples,
                      initial_state,
                      config
                      )

    energy = jnp.mean(energies)
    return energy, energies
