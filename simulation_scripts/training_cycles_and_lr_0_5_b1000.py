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

# Config
output_filename = "../data/training_cycles_lr_0_5_b1000.csv"
nparticles = 1    # particles
dim = 1            # dimensionality
nhidden = 2       # hidden neurons
nsamples = int(2**18)
nchains = 8
eta = 0.5

training_cycles = [50_000, 100_000, 250_000, 500_000]

dfs = []

for max_iter in training_cycles:
    system = nqs.NQS(nparticles,
                     dim,
                     nhidden=nhidden,
                     interaction=False,
                     mcmc_alg='rwm',
                     nqs_repr='psi',
                     backend='numpy',
                     log=True
                     )

    system.init(sigma2=1.0, scale=3.0)

    system.train(max_iter=max_iter,
                 batch_size=1_000,
                 gradient_method='adam',
                 eta=eta,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 early_stop=False
                 )

    df = system.sample(nsamples,
                       nchains=nchains,
                       seed=None
                       )

    sem_factor = 1 / np.sqrt(len(df))
    mean_data = df[["energy",
                    "std_error",
                    "variance",
                    "accept_rate"]
                   ].mean().to_dict()
    mean_data["sem_energy"] = df["energy"].std() * sem_factor
    mean_data["sem_std_error"] = df["std_error"].std() * sem_factor
    mean_data["sem_variance"] = df["variance"].std() * sem_factor
    mean_data["sem_accept_rate"] = df["accept_rate"].std() * sem_factor
    info_data = df[["nparticles",
                    "dim",
                    "eta",
                    "scale",
                    "nvisible",
                    "nhidden",
                    "mcmc_alg",
                    "nsamples",
                    "training_cycles",
                    "training_batch"]
                   ].iloc[0].to_dict()
    data = {**mean_data, **info_data}
    df_mean = pd.DataFrame([data])
    dfs.append(df_mean)

df_final = pd.concat(dfs)
# Save results
df_final.to_csv(output_filename, index=False)
