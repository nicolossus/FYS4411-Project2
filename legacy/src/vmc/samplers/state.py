#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# DISCLAIMER: Idea and code structure from blackjax

from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union

import jax.numpy as jnp
import numpy as onp

Array = Union[onp.ndarray, jnp.ndarray]
PyTree = Union[Array, Iterable[Array], Mapping[Any, Array]]


class State(NamedTuple):
    positions: PyTree
    logp: Union[float, PyTree]
    n_accepted: int
    delta: int
