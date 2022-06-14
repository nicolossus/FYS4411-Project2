#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import warnings
from abc import abstractmethod
from functools import partial
from multiprocessing import Lock, RLock
from threading import RLock as TRLock

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pathos.pools import ProcessPool
from tqdm.auto import tqdm

from ..utils import (block, check_and_set_nchains, early_stopping,
                     generate_seed_sequence, setup_logger, tune_dt_table,
                     tune_scale_table)
from .state import State

warnings.filterwarnings("ignore", message="divide by zero encountered")


class SamplingNotPerformed(Exception):
    r"""Failed attempt at accessing posterior samples.
    A call to the sample method must be carried out first.
    """
    pass


class BaseRBMVMC:
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

    def __init__(self, wavefunction, inference_scheme=None, rng=None, update_method="gd"):

        #self._check_inference_scheme(inference_scheme)

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
        # Needed derivatives for updating the RBM
        self._grad_a_fn = self._wf.grad_a
        self._grad_b_fn = self._wf.grad_b
        self._grad_W_fn = self._wf.grad_weights
        self._gradient_method = update_method

    @abstractmethod
    def step(self, state, scale=1.0):
        """Single sampling step.

        To be overwritten by subclass. Signature must be as shown in the
        abstract method, meaning that algorithm specific parameters must be
        set via keyword arguments.
        """

        raise NotImplementedError

    def train(self, ntrains, nsamples, initial_positions, eta=0.01, seed=0):
        """
        Train RBM

        Parameters
        ----------
        ntrains     : int, number of training rounds
        nsamples    : int, number of MC cycles
        initial_positions : np.ndarray(shape=(N, dim))
        eta         : float, learning rate
        seed        : number, seed for PRNG
        """
        #print("Shape eta: ", eta.shape)
        state = self.initial_state(initial_positions)
        training_energies = np.zeros(ntrains)

        # Initialising momentums and second momentums
        m_a = np.zeros_like(self._wf._a); v_a = np.zeros_like(self._wf._a)
        m_b = np.zeros_like(self._wf._b); v_b = np.zeros_like(self._wf._b)
        m_W = np.zeros_like(self._wf._W); v_W = np.zeros_like(self._wf._W)

        for i in range(ntrains):
            energies = []
            grad_a = []
            grad_b = []
            grad_W = []

            for _ in range(nsamples):
                state  = self.step(state, seed)
                #print("State positions: ", state.positions)
                energies.append(self._locE_fn(state.positions))
                grad_a.append(self._grad_a_fn(state.positions))
                grad_b.append(self._grad_b_fn(state.positions))
                grad_W.append(self._grad_W_fn(state.positions))

            energies = np.array(energies)
            grad_a = np.array(grad_a)
            grad_b = np.array(grad_b)
            grad_W = np.array(grad_W)

            expect_energy = np.mean(energies)
            expect_grad_a = np.mean(grad_a, axis=0)
            expect_grad_b = np.mean(grad_b, axis=0)
            expect_grad_W = np.mean(grad_W, axis=0)
            '''
            expect_grad_a_E = np.mean(energies*grad_a, axis=0)
            expect_grad_b_E = np.mean(energies*grad_b, axis=0)
            expect_grad_W_E = np.mean(energies*grad_W, axis=0)
            '''
            expect_grad_a_E = np.mean(energies.reshape(nsamples, 1, 1)*grad_a, axis=0)
            expect_grad_b_E = np.mean(energies.reshape(nsamples, 1)*grad_b, axis=0)
            expect_grad_W_E = np.mean(energies.reshape(nsamples, 1, 1, 1)*grad_W, axis=0)
            '''
            expect_grad_a_E = 0.0
            expect_grad_b_E = 0.0
            expect_grad_W_E = 0.0
            for j, energy in enumerate(energies):
                expect_grad_a_E += energy*grad_a[j, :, :]/float(nsamples)
                expect_grad_b_E += energy*grad_b[j, :]/float(nsamples)
                expect_grad_W_E += energy*grad_W[j, :, :, :]/float(nsamples)
            '''

            gradient_a = 2 * (expect_grad_a_E - expect_grad_a * expect_energy)
            gradient_b = 2 * (expect_grad_b_E - expect_grad_b * expect_energy)
            gradient_W = 2 * (expect_grad_W_E - expect_grad_W * expect_energy)
            self.update_parameters(gradient_a, gradient_b, gradient_W, m_a, v_a, m_b, v_b, m_W, v_W, eta=eta)
            #if (i%100 == 0):
            print(f"At iteration {i}: Energy={expect_energy}.")
            training_energies[i] = expect_energy
        print("Accepted: ", state.n_accepted)
        return training_energies

    def update_parameters(self, grad_a, grad_b, grad_W, m_a, v_a, m_b, v_b, m_W, v_W, eta=0.1):
        """
        Updates the biases and weights of the RBM

        Parameters
        ---------
        grad_a      :  np.ndarray(shape = a.shape)
        grad_b      :  np.ndarray(shape = b.shape)
        grad_W      :  np.ndarray(shape = W.shape)
        eta         :  float, learning rate of the optimizing scheme
        """


        if self._gradient_method == "adam":
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8


        if self._gradient_method == "gd":
            self._wf._a -= eta * grad_a
            self._wf._b -= eta * grad_b
            self._wf._W -= eta * grad_W

        elif self._gradient_method == "adam":
            # Update bias visible layer
            m_a = beta1 * m_a + (1 - beta1) * grad_a
            v_a = beta2 * v_a + (1 - beta2) * grad_a*grad_a
            m_a_hat = m_a / (1 - beta1)
            v_a_hat = v_a / (1 - beta2)
            self._wf._a -= eta * m_a_hat / (np.sqrt(v_a_hat) - epsilon)
            # Update bias hidden layer
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b*grad_b
            m_b_hat = m_b / (1 - beta1)
            v_b_hat = v_b / (1 - beta2)
            self._wf._b -= eta * m_b_hat / (np.sqrt(v_b_hat) - epsilon)
            # Update weights
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W*grad_W
            m_W_hat = m_W / (1 - beta1)
            v_W_hat = v_W / (1 - beta2)
            self._wf._W -= eta * m_W_hat / (np.sqrt(v_W_hat) - epsilon)


    def initial_state(self, initial_positions):
        state = State(initial_positions,
                      self._logp_fn(initial_positions),
                      0,
                      0
                      )
        return state
