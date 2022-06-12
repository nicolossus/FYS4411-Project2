#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time

import jax
import matplotlib.pyplot as plt
import nqs
import numpy as np
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

# Import code from src
# sys.path.insert(0, './src/')
#import src as nqs  # noqa

nparticles = 1     # particles
dim = 1            # dimensionality
nhidden = 2        # hidden neurons

system = nqs.NQS(nparticles,
                 dim,
                 nhidden=nhidden,
                 interaction=False,
                 mcmc_alg='rwm',
                 nqs_repr='psi',
                 backend='numpy',
                 log=True
                 )

system.init(sigma2=1.0, scale=0.5)

'''
system.tune(tune_iter=10_000,
            tune_interval=500,
            early_stop=False,  # set to True later
            rtol=1e-05,
            atol=1e-08,
            seed=None,
            mcmc_alg=None
            )
'''

system.train(max_iter=50_000,
             batch_size=1000,
             gradient_method='adam',
             eta=0.01,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-8,
             early_stop=False,  # set to True later
             rtol=1e-05,
             atol=1e-08,
             seed=None
             )


system.tune(tune_iter=50_000,
            tune_interval=1000,
            early_stop=False,  # set to True later
            rtol=1e-05,
            atol=1e-08,
            seed=None,
            mcmc_alg=None
            )


energies = system.sample(50_000,
                         nchains=1,
                         burn=1000,
                         seed=None,
                         mcmc_alg=None
                         )

plt.plot(energies)
plt.show()
