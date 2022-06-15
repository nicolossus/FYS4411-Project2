#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import warnings
from abc import abstractmethod
from functools import partial

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pathos.pools import ProcessPool

from .blocking import block
from .pool_tools import check_and_set_jobs, generate_seed_sequence
from .sampler_utils import early_stopping, tune_dt_table, tune_scale_table
from .state import State

warnings.filterwarnings("ignore", message="divide by zero encountered")


class SamplingNotPerformed(Exception):
    r"""Failed attempt at accessing posterior samples.
    A call to the sample method must be carried out first.
    """
    pass


class BaseVMC:
    """

    Arguments
    ---------
    wavefunction : vmc_numpy.WaveFunction
        Wave function with methods dictated by vmc_numpy.WaveFunction
    inference_scheme : str
        Name of inference scheme. Can be used to condition for algorithm
        specific methods.
    rng : generator
        Random number generator. Default: numpy.random.default_rng
    """

    def __init__(self, wavefunction, inference_scheme=None, rng=None):

        self._check_inference_scheme(inference_scheme)

        self._wf = wavefunction
        self._inference_scheme = inference_scheme

        if rng is None:
            rng = default_rng
        self._rng = rng

        # Retrieve callables
        self._wf_fn = self._wf.wf
        self._logp_fn = self._wf.logprob
        self._locE_fn = self._wf.local_energy
        self._driftF_fn = self._wf.drift_force
        self._grad_alpha_fn = self._wf.grad_alpha
        self._pdf = self._wf.PDF_vectorized

    def _check_inference_scheme(self, inference_scheme):
        if inference_scheme is None:
            msg = 'inference_scheme must be passed to the base vmc constructor'
            raise ValueError(msg)

        if not isinstance(inference_scheme, str):
            msg = 'inference_scheme must be passed as str'
            raise TypeError(msg)

    @abstractmethod
    def step(self, state, alpha, **kwargs):
        """Single sampling step.

        To be overwritten by subclass. Signature must be as shown in the
        abstract method, meaning that algorithm specific parameters must be
        set via keyword arguments.
        """

        raise NotImplementedError

    def sample(
        self,
        nsamples,
        initial_positions,
        alpha,
        nchains=1,
        seed=None,
        warm=True,
        warmup_iter=500,
        tune=True,
        tune_iter=5000,
        tune_interval=250,
        tol_tune=1e-5,
        optimize=True,
        max_iter=50000,
        batch_size=500,
        gradient_method='adam',
        eta=0.01,
        tol_optim=1e-5,
        early_stop=True,
        **kwargs
    ):
        """Sampling procedure"""
        self._reject_batch = False
        # Settings for warm-up
        self._warm = warm
        self._warmup_iter = warmup_iter

        # Settings for tuning
        self._tune = tune
        self._tune_iter = tune_iter
        self._tune_interval = tune_interval
        self._tol_tune = tol_tune

        # Settings for optimize
        self._optimize = optimize
        self._max_iter = max_iter
        self._batch_size = batch_size
        self._gradient_method = gradient_method
        self._tol_optim = tol_optim

        # Flag for early stopping
        self._early_stop = early_stop

        # Set and run chains
        nchains = check_and_set_jobs(nchains)
        seeds = generate_seed_sequence(seed, nchains)

        if nchains == 1:
            self._final_state, results, self._energies, self._distances, self._pdfs = self._sample(nsamples,
                                                                                                   initial_positions,
                                                                                                   alpha,
                                                                                                   eta,
                                                                                                   seeds[0],
                                                                                                   **kwargs
                                                                                                   )

            self._results_full = pd.DataFrame([results])

        else:
            # kwargs
            # iterables
            nsamples = (nsamples,) * nchains
            initial_positions = (initial_positions,) * nchains
            alpha = (alpha,) * nchains
            eta = [eta] * nchains
            kwargs = (kwargs,) * nchains

            # nsamples, initial_positions, alpha, eta, **kwargs

            with ProcessPool(nchains) as pool:
                # , self._distances
                self._final_state, results, self._energies, self._distances, self._pdfs = zip(*pool.map(self._sample,
                                                                                              nsamples,
                                                                                              initial_positions,
                                                                                              alpha,
                                                                                              eta,
                                                                                              seeds,
                                                                                              kwargs,
                                                                                                        ))

                self._results_full = pd.DataFrame(results)

        #print("Shape of pdfs within sample: ", self._pdfs.shape)
        return self.results

    def _sample(self, *args, **kwargs):
        """To be called by process"""

        # Some trickery to get this to work with multiprocessing
        if not kwargs:
            nsamples, initial_positions, alpha, eta, seed, kwargs = args
        else:
            nsamples, initial_positions, alpha, eta, seed = args

        # Set some flags and counters
        retune = False
        rewarm = True
        actual_warm_iter = 0
        actual_tune_iter = 0
        actual_optim_iter = 0
        subtract_iter = 0

        # Set initial state
        state = self.initial_state(initial_positions, alpha)

        # Warm-up?
        if self._warm:
            state = self.warmup_chain(state, alpha, seed, **kwargs)
            actual_warm_iter += state.delta
            subtract_iter = actual_warm_iter

            print("Warm done")

        # Tune?
        if self._tune:
            state, kwargs = self.tune_selector(state, alpha, seed, **kwargs)
            actual_tune_iter += state.delta - subtract_iter
            subtract_iter = actual_tune_iter + actual_warm_iter

            print("Tune done")

        # Optimize?
        if self._optimize:
            state, alpha = self.optimizer(state, alpha, eta, seed, **kwargs)
            actual_optim_iter += state.delta - subtract_iter
            subtract_iter = actual_optim_iter + actual_tune_iter + actual_warm_iter
            # retune = True
            """
            ^ TURNED OFF FOR DEBUG
            """

            print(f"Optimize done, final alpha: {alpha}")
        # Retune for good measure
        if retune:
            state, kwargs = self.tune_selector(state, alpha, seed, **kwargs)
            actual_tune_iter += state.delta - subtract_iter
            print("Retune done")

        if rewarm:
            state = self.warmup_chain(state, alpha, seed, **kwargs)
            actual_warm_iter += state.delta
            subtract_iter = actual_warm_iter
            print("Warm after tune done")

        print("Sampling energy")
        # Sample energy
        # , distances
        state, energies, distances, pdfs = self.sample_energy(nsamples,
                                                              state,
                                                              alpha,
                                                              seed,
                                                              **kwargs
                                                              )

        results = self._accumulate_results(state,
                                           energies,
                                           distances,
                                           nsamples,
                                           alpha,
                                           eta,
                                           actual_warm_iter,
                                           actual_tune_iter,
                                           actual_optim_iter,
                                           **kwargs
                                           )
        #print("Shape pdfs within _sample: ", pdfs.shape)
        return state, results, energies, distances, pdfs

    def _accumulate_results(
        self,
        state,
        energies,
        distances,
        nsamples,
        alpha,
        eta,
        warm_cycles,
        tune_cycles,
        optim_cycles,
        **kwargs
    ):
        """
        Gather results
        """

        N, d = state.positions.shape

        # total_moves = nsamples * N * d
        #total_moves = nsamples
        # total_moves = nsamples*N
        total_moves = nsamples  # *N*d
        acc_rate = state.n_accepted / total_moves
        energy = np.mean(energies)
        mean_distance = np.mean(distances)
        # blocking
        error = block(energies)
        # effective samples?
        if self._inference_scheme == "metropolis":
            scale_name = "scale"
            scale_val = kwargs[scale_name]
        elif self._inference_scheme == "metropolis-hastings":
            scale_name = "dt"
            scale_val = kwargs[scale_name]

        results = {"nparticles": N,
                   "dim": d,
                   scale_name: scale_val,
                   "eta": eta,
                   "alpha": alpha,
                   "energy": energy,
                   "mean_distance": mean_distance,
                   "standard_error": error,
                   "accept_rate": acc_rate,
                   "nsamples": nsamples,
                   "total_cycles": state.delta,
                   "warmup_cycles": warm_cycles,
                   "tuning_cycles": tune_cycles,
                   "optimize_cycles": optim_cycles
                   }

        return results

    def initial_state(self, initial_positions, alpha):
        state = State(initial_positions,
                      self._logp_fn(initial_positions, alpha),
                      0,
                      0
                      )
        return state

    def warmup_chain(self, state, alpha, seed, **kwargs):
        """Warm-up the chain for warmup_iter cycles.

        Arguments
        ---------
        warmup_iter : int
            Number of cycles to warm-up the chain
        State : vmc_numpy.State
            Current state of the system
        alpha : float
            Variational parameter
        **kwargs
            Arbitrary keyword arguments are passed to the step method

        Returns
        -------
        State
            The state after warm-up
        """
        for i in range(self._warmup_iter):
            state = self.step(state, alpha, seed, **kwargs)
        return state

    def tune_selector(self, state, alpha, seed, **kwargs):
        """Select appropriate tuning procedure"""

        if self._inference_scheme == "metropolis":
            scale = kwargs.pop("scale")
            state, new_scale = self.tune_scale(state,
                                               alpha,
                                               seed,
                                               scale,
                                               **kwargs)
            kwargs = dict(kwargs, scale=new_scale)
        elif self._inference_scheme == "metropolis-hastings":
            dt = kwargs.pop("dt")

            state, new_dt = self.tune_dt(state,
                                         alpha,
                                         seed,
                                         dt,
                                         **kwargs)
            kwargs = dict(kwargs, dt=new_dt)
        else:
            msg = (f"Tuning of {self._inference_scheme} currently not "
                   "available, set tune=False")
            raise ValueError(msg)

        return state, kwargs

    def tune_scale(self, state, alpha, seed, scale, **kwargs):
        """For samplers with scale parameter."""

        steps_before_tune = self._tune_interval
        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)
        N, d = state.positions.shape
        #total_moves = self._tune_interval * N * d
        total_moves = self._tune_interval
        #total_moves = self._tune_interval * N
        count = 0
        for i in range(self._tune_iter):
            state = self.step(state, alpha, seed, scale=scale, **kwargs)
            steps_before_tune -= 1

            if steps_before_tune == 0:
                old_scale = scale
                accept_rate = state.n_accepted / total_moves
                scale = tune_scale_table(old_scale, accept_rate)
                if scale == old_scale:
                    count += 1
                else:
                    count = 0
                #print(f"Acceptance rate {accept_rate} and scale {scale}")
                # Reset
                steps_before_tune = self._tune_interval
                state = State(state.positions, state.logp, 0, state.delta)

                # Early stopping?
                if count > 2:
                    break
                """
                if self._early_stop:
                    if early_stopping(scale, old_scale, tolerance=self._tol_tune):
                        break
                """
        #print(f"Final acceptance rate {accept_rate} and scale {scale}")
        return state, scale

    def tune_dt(self, state, alpha, seed, dt, **kwargs):
        """For samplers with dt parameter."""

        steps_before_tune = self._tune_interval
        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)
        N, d = state.positions.shape
        # total_moves = self._tune_interval * N * d
        total_moves = self._tune_interval
        # print("Tuning..")
        for i in range(self._tune_iter):
            state = self.step(state, alpha, seed, dt=dt, **kwargs)
            steps_before_tune -= 1

            if steps_before_tune == 0:
                old_dt = dt
                accept_rate = state.n_accepted / total_moves
                dt = tune_dt_table(old_dt, accept_rate)
                # print(f'Accept rate: {accept_rate}, dt: {dt}')

                # Reset
                steps_before_tune = self._tune_interval
                state = State(state.positions, state.logp, 0, state.delta)

                # Early stopping?

                if self._early_stop:
                    if early_stopping(dt, old_dt, tolerance=self._tol_tune):
                        break

        #print(f"Final dt val: {dt}, with accept rate: {accept_rate}")
        return state, dt

    def optimizer(self, state, alpha, eta, seed, **kwargs):
        """Optimize alpha
        """

        # Set hyperparameters for Adam
        if self._gradient_method == "adam":
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            m = 0
            v = 0
            t = 0

        # Set initial config
        steps_before_optimize = self._batch_size
        energies = []
        grad_alpha = []

        for i in range(self._max_iter):
            state = self.step(state, alpha, seed, **kwargs)
            energies.append(self._locE_fn(state.positions, alpha))
            grad_alpha.append(self._grad_alpha_fn(state.positions, alpha))
            steps_before_optimize -= 1

            if steps_before_optimize == 0:
                old_alpha = alpha

                # Expectation values
                energies = np.array(energies)
                grad_alpha = np.array(grad_alpha)
                expect1 = np.mean(grad_alpha * energies)
                expect2 = np.mean(grad_alpha)
                expect3 = np.mean(energies)
                gradient = 2 * (expect1 - expect2 * expect3)

                # Gradient descent
                if self._gradient_method == "gd":
                    alpha -= eta * gradient
                elif self._gradient_method == "adam":
                    t += 1
                    m = beta1 * m + (1 - beta1) * gradient
                    v = beta2 * v + (1 - beta2) * gradient**2
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    alpha -= eta * m_hat / (np.sqrt(v_hat) - epsilon)

                # Reset
                energies = []
                grad_alpha = []
                steps_before_optimize = self._batch_size

                # Early stopping?
                if self._early_stop:
                    if early_stopping(alpha, old_alpha, tolerance=self._tol_optim):
                        break

        return state, alpha

    def sample_energy(self, nsamples, state, alpha, seed, **kwargs):

        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)
        # For one body density calculation
        nparticles = state.positions.shape[0]
        #distances = np.zeros(nsamples, nparticles)
        #pdfs = np.zeros((nsamples, nparticles))
        #distances = np.zeros((nsamples, nparticles))
        energies = np.zeros(nsamples)

        for i in range(nsamples):
            state = self.step(state, alpha, seed, **kwargs)
            energies[i] = self._locE_fn(state.positions, alpha)
            distances[i, :] = np.linalg.norm(state.positions, axis=1)
            pdfs[i, :] = self._pdf(state.positions, alpha)

        #print("Shape pdfs inside sample_energy: ", pdfs.shape)
        return state, energies  # , distances, pdfs
    """
    def sample_distance(self, nsamples, state, alpha, seed, **kwargs):

        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)

        nparticles = state.positions.shape[0]
        distances = np.zeros(nsamples, nparticles)

        for i in range(nsamples):
            state = self.step(state, alpha, seed, **kwargs)
    """
    @property
    def results_all(self):
        try:
            return self._results_full
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

    @property
    def results(self):
        df = self.results_all[["nparticles", "dim", "alpha",
                               "energy",  "mean_distance", "standard_error", "accept_rate"]]
        return df

    @property
    def energy_samples(self):
        try:
            return self._energies
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

    @property
    def final_state(self):
        try:
            return self._final_state
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

    @property
    def distance_samples(self):
        try:
            return self._distances
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

    @property
    def pdf_samples(self):
        try:
            """
            pdfs = self._pdfs
            num_mc_cycles, N = pdfs.shape
            pdf_data = {}
            for i in range(num_mc_cycles):
                pdf_data[f"Cycle_{i+1}"] = []
            for i in range(num_mc_cycles):
                for particle in range(N):
                    pdf_data[f"Cycle_{i+1}"].append(pdfs[i, particle])
            """
            return self._pdfs  # pd.DataFrame(pdf_data)

        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

    @property
    def alpha(self):
        return self.results["alpha"]

    @property
    def energy(self):
        return self.results["energy"]

    @property
    def standard_error(self):
        return self.results["standard_error"]

    @property
    def accept_rate(self):
        return self.results["accept_rate"]
