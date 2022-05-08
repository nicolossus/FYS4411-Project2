#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from setup import df_to_latex, fig_path


def example_figure():
    """
    Generate figure and save to latex folder
    """
    im = np.random.random([500, 500])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(im, cmap='gray')
    fig.savefig(fig_path("example.png"))


def example_table():
    """
    Generate dataframe, render to latex tabular env. and save to latex folder
    """
    # Dataframe for storing results
    df = pd.DataFrame(columns=['$x$', '$x^2$', '$x^3$'])
    for x in [0.25, 0.5, 0.75]:
        df = df.append(
            {'$x$': x,
             '$x^2$': x**2,
             '$x^3$': x**3},
            ignore_index=True
        )
    # write to file
    df_to_latex(df, "example.tex")


if __name__ == "__main__":
    example_figure()
    example_table()
