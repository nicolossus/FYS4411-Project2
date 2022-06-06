#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

'''
TODO:
- why does nn.initializers.he_normal fail?
- check against analytical (see printout of "variables" in main block for
initial values of biases and weight matrix)
'''

default_kernel_init = jax.nn.initializers.normal(stddev=0.01)


class RBM(nn.Module):
    """Gaussian-Binary Restricted Boltzmann Machine

    RBM with one visible and one hidden layer – equivalent to a 2-layer FFNN
    with a nonlinear activation function in between.

    Initializers: https://jax.readthedocs.io/en/latest/jax.nn.initializers.html
    """
    # Number of hidden neurons
    nhidden: int = 2
    # Variance
    sigma2: float = 1.0
    # Initializer for the Dense layer matrix
    kernel_init: Callable = default_kernel_init  # nn.initializers.he_normal
    # Initializer for the hidden bias
    hidden_bias_init: Callable = default_kernel_init  # nn.initializers.he_normal
    # Initializer for the visible bias
    visible_bias_init: Callable = default_kernel_init  # nn.initializers.he_normal

    @nn.compact
    def __call__(self, input):

        # Visible layer
        v_bias = self.param(
            "visible_bias",
            self.visible_bias_init,
            (input.shape[-1],),
            np.float64
        )

        x_v = jnp.linalg.norm(input - v_bias)
        x_v *= -x_v / (4 * self.sigma2)

        # Hidden layer
        layer = nn.Dense(
            name="hidden_layer",
            features=self.nhidden,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init
        )

        x_h = layer(input / self.sigma2)
        x_h = nn.log_sigmoid(-x_h)
        x_h = jnp.sum(x_h, axis=-1)
        x_h *= -0.5

        return x_v + x_h


if __name__ == "__main__":
    psi = RBM(nhidden=2, sigma2=1.0)
    r = jnp.array([[0.2, 0.5], [0.3, 0.7]])
    variables = psi.init(jax.random.PRNGKey(0), r)
    print(variables)
    output = psi.apply(variables, r)
    print(output)
