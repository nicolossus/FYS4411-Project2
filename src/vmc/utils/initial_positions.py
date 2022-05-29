#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def safe_initial_positions(wavefunction, alpha, N, dim, nchains=1, interaction=True):

    if interaction:
        if nchains == 1:
            positions = _with_interactions(wavefunction, alpha, N, dim)
        else:
            positions = [_with_interactions(
                wavefunction, alpha, N, dim) for _ in range(nchains)]
    else:
        if nchains == 1:
            positions = _without_interactions(wavefunction, alpha, N, dim)
        else:
            positions = [_without_interactions(
                wavefunction, alpha, N, dim) for _ in range(nchains)]

    return positions


def _without_interactions(wavefunction, alpha, N, dim):
    rng = np.random.default_rng()
    positions = rng.random(size=(N, dim))

    wf2 = wavefunction.pdf(positions, alpha)
    while np.sum(wf2) <= 1e-14:
        positions *= 0.5
        wf2 = wavefunction.pdf(positions, alpha)

    return positions


def _with_interactions(wavefunction, alpha, N, dim, a=0.00433):
    rng = np.random.default_rng()

    def corr_factor(r1, r2):
        rij = np.linalg.norm(r1 - r2)
        if rij <= a:
            return 0.
        else:
            return 1 - (a / rij)

    scale = 2.
    r = np.random.randn(N, dim) * scale

    rerun = True
    while rerun:
        rerun = False
        for i in range(N):
            for j in range(i + 1, N):
                corr = corr_factor(r[i, :], r[j, :])
                if corr == 0.:
                    #print("corr=0 encountered")
                    rerun = True
                    r[i, :] = np.random.randn() * scale
                    r[j, :] = np.random.randn() * scale
        scale *= 1.5

    return r
