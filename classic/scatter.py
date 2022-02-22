from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

from classic.basic import get_color


def scattercomp(
    data: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    opacity: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    on_axis: Optional[plt.Axes] = None,
) -> Optional[tuple[plt.Figure, plt.Axes]]:
    """Creates a 2D scatter plot for the given data.

    Args:
        data (np.ndarray | tuple[np.ndarray, np.ndarray]): One or a
            tuple of two numpy arrays. If one is given, a histogram of
            the data is drawn.
        opacity (np.ndarray, optional): A numpy array with matching
            length as the ones supplied in ``data``. Points in the
            scatter plot will have corresponding opacity color values.
        labels (Sequence[str], optional): The labels of the two sets of
            data categories, e.g. the name of the classifiers used to
            create given accuracy results.
        on_axis (plt.Axes, optional): A matplotlib axis that the plot
            will be drawn on. If none is supplied, a new one will be
            created first.

    Returns:
        Pyplot figure and axis with the scatter plot.
    """
    fig, axs = plt.subplots(1, 1)
    if on_axis is not None:
        axs = on_axis
    if isinstance(data, tuple):
        n_datasets = data[0].shape[0]
        colors = np.zeros((n_datasets, 4))
        colors[:, :3] = get_color(0)
        colors[:, 3] = opacity

        # set up the axis
        axs.axis('square')
        axs.set_xlim([0, 1])
        axs.set_ylim([0, 1])
        axs.scatter(
            data[0], data[1],
            c=opacity,
            cmap="copper_r",
        )

        # draw auxiliary lines for equality and five percent difference
        axs.plot(
            [0, 1], [0, 1],
            transform=axs.transAxes,
            color=get_color(1),
            ls="--",
        )
        axs.plot(
            [0.05, 1], [0, 0.95],
            transform=axs.transAxes,
            color=get_color(1) + (0.3,),
            ls="--",
        )
        axs.plot(
            [0, 0.95], [0.05, 1],
            transform=axs.transAxes,
            color=get_color(1) + (0.3,),
            ls="--",
        )

        # draw lines for the mean values on each axis
        mean1 = data[0].mean()
        mean2 = data[1].mean()
        axs.axhline(
            mean1,
            xmin=0,
            xmax=mean1,
            color=get_color(3) + (0.5, ),
            ls="--",
        )
        axs.axvline(
            mean2,
            ymin=0,
            ymax=mean2,
            color=get_color(3) + (0.5, ),
            ls="--"
        )

        # place labels if given
        if labels is not None:
            if len(labels) != 2:
                raise ValueError(f"Expected two labels, got {len(labels)}")
            axs.text(
                x=0.02,
                y=0.98,
                s=labels[0],
                size="large",
                ha="left",
                va="top",
            )
            axs.text(
                x=0.98,
                y=0.02,
                s=labels[1],
                size="large",
                ha="right",
                va="bottom",
            )
    else:
        weights = np.ones_like(data)
        weights /= data.shape[0]
        axs.hist(
            data,
            weights=weights,
        )

    if on_axis is None:
        return fig, axs
    return None
