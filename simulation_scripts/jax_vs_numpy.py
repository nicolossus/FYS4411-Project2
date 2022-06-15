#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time

import jax
import matplotlib.pyplot as plt
#import nqs
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import nqs package
sys.path.insert(0, '../nqs/')
import nqs  # noqa

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


output_filename = "../data/runtimes.csv"
nparticles = 1    # particles
dim = 1            # dimensionality
nhidden = 1       # hidden neurons


training_cycles = [10_000, 20_000, 30_000, 40_000]
data = {"max_iter": training_cycles,
        "t_rwm_numpy": [],
        "t_lmh_numpy": [],
        "t_rwm_jax": [],
        "t_lmh_jax": []
        }

# Numpy, RWM
for max_iter in training_cycles:
    system = nqs.NQS(nparticles,
                     dim,
                     nhidden=nhidden,
                     interaction=False,
                     mcmc_alg='rwm',
                     nqs_repr='psi',
                     backend='numpy',
                     log=False
                     )

    system.init(sigma2=1.0, scale=3.0)

    t_start = time.time()

    system.train(max_iter=max_iter,
                 batch_size=1_000,
                 gradient_method='adam',
                 eta=0.05,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 early_stop=False
                 )
    t_end = time.time()
    data['t_rwm_numpy'].append(t_end - t_start)


# Numpy, LMH
for max_iter in training_cycles:
    system = nqs.NQS(nparticles,
                     dim,
                     nhidden=nhidden,
                     interaction=False,
                     mcmc_alg='lmh',
                     nqs_repr='psi',
                     backend='numpy',
                     log=False
                     )

    system.init(sigma2=1.0, scale=1.3)

    t_start = time.time()

    system.train(max_iter=max_iter,
                 batch_size=1_000,
                 gradient_method='adam',
                 eta=0.05,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 early_stop=False
                 )
    t_end = time.time()
    data['t_lmh_numpy'].append(t_end - t_start)


# JAX, RWM
for max_iter in training_cycles:
    system = nqs.NQS(nparticles,
                     dim,
                     nhidden=nhidden,
                     interaction=False,
                     mcmc_alg='rwm',
                     nqs_repr='psi',
                     backend='jax',
                     log=False
                     )

    system.init(sigma2=1.0, scale=3.0)

    t_start = time.time()

    system.train(max_iter=max_iter,
                 batch_size=1_000,
                 gradient_method='adam',
                 eta=0.05,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 early_stop=False
                 )
    t_end = time.time()
    data['t_rwm_jax'].append(t_end - t_start)


# JAX, LMH
for max_iter in training_cycles:
    system = nqs.NQS(nparticles,
                     dim,
                     nhidden=nhidden,
                     interaction=False,
                     mcmc_alg='lmh',
                     nqs_repr='psi',
                     backend='jax',
                     log=False
                     )

    system.init(sigma2=1.0, scale=1.3)

    t_start = time.time()

    system.train(max_iter=max_iter,
                 batch_size=1_000,
                 gradient_method='adam',
                 eta=0.05,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 early_stop=False
                 )
    t_end = time.time()
    data['t_lmh_jax'].append(t_end - t_start)

print(data)
df = pd.DataFrame(data)
print(df)
df.to_csv(output_filename, index=False)
#print(df[["energy", "std_error", "variance", "accept_rate"]])
