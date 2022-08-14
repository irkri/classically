from itertools import combinations, combinations_with_replacement
from typing import Union, Optional, Sequence, overload

import networkx as nx
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib import ticker


def critical_difference_graph(
    data: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    holm_bonferroni: bool = True,
) -> nx.Graph:
    """Creates and returns a graph needed by the critical difference
    diagram. Nodes in the graph are different categories or classifiers
    and two nodes are connected if they have a significant difference
    based on the values supplied. This is based on pairwise Wilcoxon
    signed rank tests with Holm-Bonferroni correction. The edge weights
    are set to the p-values of the tests.

    Args:
        data (np.ndarray): A numpy array with two dimensions. The first
            dimension iterates over different categories that are
            compared in this diagram, e.g. classifiers. The second
            dimension iterates over the values being used for the
            comparison, e.g. accuracy results.
        labels (Sequence[str], optional): The labels for categories of
            the data, e.g. the name of the classifiers used to create
            given accuracy results.
        alpha (float, optional): Significance level used for doing
            pairwise Wilcoxon signed rank tests. Defaults to 0.05.
        holm_bonferroni (bool, optional): Whether to use holm-bonferroni
            correction on the Wilcoxon signed rank tests. Defaults to
            True.

    Returns:
        A graph built with the package ``networkx``.
    """
    if data.ndim != 2:
        raise ValueError(
            f"Expected numpy array with two dimensions, got {data.ndim}"
        )
    n_classifiers = data.shape[0]
    if labels is None:
        labels = [f"Set {i+1}" for i in range(n_classifiers)]
    if len(labels) != n_classifiers:
        raise ValueError(
            "Number of labels must equal the number of classifiers to compare"
        )

    # paired Wilcoxon test for every (distinct) pair of classifiers
    p_values = np.zeros(
        shape=(int(n_classifiers * (n_classifiers-1) / 2), ),
        dtype=np.float32,
    )
    c = -1
    for i in range(n_classifiers-1):
        for j in range(i+1, n_classifiers):
            c += 1
            mode = "exact"
            if (nzeros := np.sum((data[i] - data[j]) == 0)) >= 1:
                if nzeros == len(data[i, :]):
                    p_values[c] = 1.0
                    continue
                mode = "approx"
            p_values[c] = stats.wilcoxon(
                data[i, :],
                data[j, :],
                zero_method='pratt',
                mode=mode,
            )[1]
    p_order = np.argsort(p_values)
    alpha_ = np.full_like(p_values, alpha)
    if holm_bonferroni:
        alpha_ = alpha_ / np.arange(p_values.shape[0], 0, -1)
    significant = (p_values[p_order] <= alpha_)[p_order.argsort()]

    # get cliques of significant test results by building an adjacency
    # matrix and using the networkx package
    adjacency_matrix = np.zeros((n_classifiers, n_classifiers))
    indexing = np.array(np.triu_indices(n_classifiers, k=1))
    for index in np.where(~significant):
        i, j = indexing[:, index]
        adjacency_matrix[i, j] = p_values[index]
    G = nx.Graph(adjacency_matrix)
    G = nx.relabel_nodes(G, {i: l for i, l in enumerate(labels)})
    return G


@overload
def critical_difference_diagram(
    data: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    holm_bonferroni: bool = True,
    axis: None = None,
    color_cliques: Optional[tuple[float, float, float, float]] = None,
    color_markings: Optional[tuple[float, float, float, float]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    ...


@overload
def critical_difference_diagram(
    data: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    holm_bonferroni: bool = True,
    axis: plt.Axes = None,
    color_cliques: Optional[tuple[float, float, float, float]] = None,
    color_markings: Optional[tuple[float, float, float, float]] = None,
) -> None:
    ...


def critical_difference_diagram(
    data: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    holm_bonferroni: bool = True,
    axis: Optional[plt.Axes] = None,
    color_cliques: Optional[tuple[float, float, float, float]] = None,
    color_markings: Optional[tuple[float, float, float, float]] = None,
) -> Optional[tuple[plt.Figure, plt.Axes]]:
    """Draws a critical difference diagram based on the given
    categorized values between 0 and 1. This type of plot was described
    in the paper
    'Statistical Comparison of Classifiers over Multiple Data Sets'
    by Janez Demsar, 2006.

    Args:
        data (np.ndarray): A numpy array with two dimensions. The first
            dimension iterates over different categories that are
            compared in this diagram, e.g. classifiers. The second
            dimension iterates over the values being used for the
            comparison, e.g. accuracy results.
        labels (Sequence[str], optional): The labels for categories of
            the data, e.g. the name of the classifiers used to create
            given accuracy results.
        alpha (float, optional): Significance level used for doing
            pairwise Wilcoxon signed rank tests. Defaults to 0.05.
        holm_bonferroni (bool, optional): Whether to use holm-bonferroni
            correction on the Wilcoxon signed rank tests. Defaults to
            True.
        axis (plt.Axes, optional): A matplotlib axis that the plot
            will be drawn on. If none is supplied, a new one will be
            created first and returned after finishing the diagram.
        color_cliques (tuple of 4 floats, optional): Color that will be
            used to mark classifiers with non-significant differences.
        color_markings (tuple of 4 floats, optional): Color for lines
            marking the different classifiers.

    Returns:
        Pyplot figure and axis with the diagram if ``axis`` is
        supplied, else None.
    """
    if data.ndim != 2:
        raise ValueError(
            f"Expected numpy array with two dimensions, got {data.ndim}"
        )
    n_classifiers = data.shape[0]
    if labels is None:
        labels = [f"Set {i+1}" for i in range(n_classifiers)]
    if len(labels) != n_classifiers:
        raise ValueError(
            "Number of labels must equal the number of classifiers to compare"
        )

    G = critical_difference_graph(data, labels, alpha, holm_bonferroni)
    cliques = [
        list(map(lambda l: labels.index(l), clq))
        for clq in nx.find_cliques(G) if len(clq) > 1
    ]

    # calculate average ranks of classifiers over all datasets
    avg_ranks = (n_classifiers - stats.rankdata(data, axis=0) + 1).mean(axis=1)
    avg_ranks_order = avg_ranks.argsort()[::-1]

    # initialize and configure plot
    width = 6 + 0.3 * max(map(len, labels))
    height = 1.0 + n_classifiers * 0.1
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    if axis is not None:
        ax = axis

    lowest_rank = min(1, int(np.floor(avg_ranks.min())))
    highest_rank = max(len(avg_ranks), int(np.ceil(avg_ranks.max())))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.spines['right'].set_color("none")
    ax.spines['left'].set_color("none")
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['bottom'].set_color("none")
    ax.spines['top'].set_linewidth(2.5)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(which='major', width=2.5, length=5, labelsize=12)
    ax.tick_params(which='minor', width=2.0, length=3, labelsize=12)
    ax.set_xlim(highest_rank, lowest_rank)
    ax.set_ylim(0.0, 1.0)

    fig.subplots_adjust(bottom=-0.6, top=0.7)

    # visual configurations
    half = int(np.ceil(n_classifiers / 2))
    label_xshift = 0.05 * (highest_rank-lowest_rank)
    rank_label_xshift = 0.02 * (highest_rank-lowest_rank)
    label_offset = 0.01 * (highest_rank-lowest_rank)
    lower_marking = 0.6
    markings_vspace = 0.35 / half
    color_markings = (
        (0.15, 0.15, 0.15, 1.0) if color_markings is None else
        color_markings
    )
    cliques_color = (
        (0.9, 0.5, 0.3, 0.9) if color_cliques is None else color_cliques
    )
    first_clique_line = 0.9 + (len(cliques) + 3) / 100
    if len(cliques) >= 4:
        first_clique_line = 0.96
    clique_line_vspace = (
        1 - (lower_marking + (half-1) * markings_vspace) - 0.001
    )
    if len(cliques) > 0:
        clique_line_vspace /= len(cliques)

    # draw left branching markings
    for i, index in enumerate(avg_ranks_order[:half]):
        ax.axvline(
            x=avg_ranks[index],
            ymin=lower_marking + (half-i-1)*markings_vspace,
            ymax=1.0,
            c=color_markings,
            lw=2.0,
        )
        ax.axhline(
            y=lower_marking + (half-i-1)*markings_vspace,
            xmin=(half-i-1) * label_xshift / (highest_rank-lowest_rank),
            xmax=(highest_rank-avg_ranks[index]) / (highest_rank-lowest_rank),
            c=color_markings,
            lw=2.0,
        )
        ax.text(
            x=highest_rank - rank_label_xshift - (half-i-1)*label_xshift,
            y=lower_marking + (half-i-1)*markings_vspace,
            s=f"{avg_ranks[index]:.2f}",
            ha="left",
            va="bottom",
            size=8,
        )
        ax.text(
            x=highest_rank - (half-i-1)*label_xshift + label_offset,
            y=lower_marking + (half-i-1)*markings_vspace,
            s=f"{labels[index]}",
            ha="right",
            va="center",
            size=14,
        )

    # draw right branching markings
    for i, index in enumerate(avg_ranks_order[half:]):
        ax.axvline(
            x=avg_ranks[index],
            ymin=lower_marking + i*markings_vspace,
            ymax=1.0,
            c=color_markings,
            lw=2.0,
        )
        ax.axhline(
            y=lower_marking + i*markings_vspace,
            xmin=(highest_rank-avg_ranks[index]) / (highest_rank-lowest_rank),
            xmax=1.0 - i * label_xshift / (highest_rank-lowest_rank),
            c=color_markings,
            lw=2.0,
        )
        ax.text(
            x=lowest_rank + rank_label_xshift + i*label_xshift,
            y=lower_marking + i*markings_vspace,
            s=f"{avg_ranks[index]:.2f}",
            ha="right",
            va="bottom",
            size=8,
        )
        ax.text(
            x=lowest_rank + i*label_xshift - label_offset,
            y=lower_marking + i*markings_vspace,
            s=f"{labels[index]}",
            ha="left",
            va="center",
            size=14,
        )

    # draw the cliques, i.e. connect classifiers that don't have a
    # significant difference
    clique_line_y = first_clique_line
    for clique in cliques:
        xmin = (
            highest_rank - avg_ranks[max(clique, key=lambda i: avg_ranks[i])]
        ) / (highest_rank - lowest_rank)
        xmax = (
            highest_rank - avg_ranks[min(clique, key=lambda i: avg_ranks[i])]
        ) / (highest_rank - lowest_rank)
        ax.axhline(
            y=clique_line_y,
            xmin=xmin,
            xmax=xmax,
            color=cliques_color,
            linewidth=4.0,
        )
        clique_line_y -= clique_line_vspace

    if axis is None:
        return fig, ax
    return None


def _hist(
    data: np.ndarray,
    axis: plt.Axes,
    label: str,
    color: Optional[tuple[float, float, float]],
) -> None:
    weights = np.ones_like(data)
    weights /= data.shape[0]
    axis.hist(
        data,
        bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        weights=weights,
        color=color,
    )
    axis.text(x=0.02, y=0.98, s=label, size="large", ha="left", va="top")


def _scattercomp(
    data: np.ndarray,
    axis: plt.Axes,
    labels: tuple[str, str],
    color: tuple[float, float, float],
    color_ml: tuple[float, float, float],
    color_dl: tuple[float, float, float],
    opacity: Optional[np.ndarray] = None,
) -> None:
    n_datasets = data[0].shape[0]
    colors = np.zeros((n_datasets, 4))
    colors[:, :3] = color
    colors[:, 3] = opacity if opacity is not None else 1.0

    # draw scatterplot
    axis.scatter(data[0, :], data[1, :], c=colors)

    # draw diagonal auxiliary lines
    axis.plot(
        [0, 1], [0, 1],
        transform=axis.transAxes,
        color=color_dl,
        ls="--",
    )
    axis.plot(
        [0.05, 1], [0, 0.95],
        [0, 0.95], [0.05, 1],
        transform=axis.transAxes,
        color=color_dl+(0.3, ),
        ls="--",
    )

    # draw lines for the mean values on each axis
    mean1 = data[0].mean()
    mean2 = data[1].mean()
    opacity1, ls1 = (0.8, "-") if mean1 > mean2 else (0.5, "--")
    opacity2, ls2 = (0.8, "-") if mean2 > mean1 else (0.5, "--")
    axis.axhline(
        mean2,
        xmin=0,
        xmax=mean2,
        color=color_ml+(opacity2, ),
        ls=ls2,
    )
    axis.axvline(
        mean1,
        ymin=0,
        ymax=mean1,
        color=color_ml+(opacity1, ),
        ls=ls1,
    )
    axis.text(0.02, 0.98, s=labels[1], size="large", ha="left", va="top")
    axis.text(0.98, 0.02, s=labels[0], size="large", ha="right", va="bottom")


@overload
def scatter_comparison(
    data: np.ndarray,
    opacity: Optional[np.ndarray] = None,
    labels: Optional[Union[str, Sequence[str]]] = None,
    axes: None = None,
    max_cols: int = 4,
    draw_hist: bool = False,
    color: Optional[tuple[float, float, float]] = None,
    color_ml: Optional[tuple[float, float, float]] = None,
    color_dl: Optional[tuple[float, float, float]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    ...


@overload
def scatter_comparison(
    data: np.ndarray,
    opacity: Optional[np.ndarray] = None,
    labels: Optional[Union[str, Sequence[str]]] = None,
    axes: Union[plt.Axes, np.ndarray] = None,
    max_cols: int = 4,
    draw_hist: bool = False,
    color: Optional[tuple[float, float, float]] = None,
    color_ml: Optional[tuple[float, float, float]] = None,
    color_dl: Optional[tuple[float, float, float]] = None,
) -> None:
    ...


def scatter_comparison(
    data: np.ndarray,
    opacity: Optional[np.ndarray] = None,
    labels: Optional[Union[str, Sequence[str]]] = None,
    axes: Optional[Union[plt.Axes, np.ndarray]] = None,
    max_cols: int = 4,
    draw_hist: bool = False,
    color: Optional[tuple[float, float, float]] = None,
    color_ml: Optional[tuple[float, float, float]] = None,
    color_dl: Optional[tuple[float, float, float]] = None,
) -> Optional[tuple[plt.Figure, plt.Axes]]:
    """Creates scatterplots comparing the data of different categories.
    This produces plots for each combination of rows in the given
    ``data`` array.
    The plots also contain a vertical and horizontal line indicating the
    mean accuracy values of the classifiers. The line corresponding to
    the lower mean is dashed.

    Args:
        data (np.ndarray): A two dimensional array, e.g. consisting of
            accuracy values from different classifiers.
        opacity (np.ndarray, optional): Opacity values that are used to
            set alpha color values to the points in the scatter plot.
            The length of this array has to be the same as the size of
            the second dimension in ``data``.
        labels (Sequence[str], optional): A number of strings matching
            names of the categories or classifiers that are being
            compared. The length of this sequence has to match
            ``data.shape[0]``.
        axes (plt.Axes | np.ndarray, optional): One or multiple axes in
            a numpy array. The plots will be drawn on these axes.
        max_cols (int, optional): If ``axes`` is not specified, this
            integer is the number of columns in a figure to fill before
            starting a new row of scatterplots. Defaults to 4.
        draw_hist (bool, optional): If True, also draws a histogram
            for each category. Defaults to False.
        color (tuple[float, float, float], optional): Color of points in
            the scatterplots.
        color_dl (tuple[float, float, float], optional): Color of the
            diagonal lines.
        color_ml (tuple[float, float, float], optional): Color of the
            lines for the mean values on each axis.

    Returns:
        Pyplot figure and axes containing all plots or None if ``axes``
        is specified.
    """
    if data.ndim != 2:
        raise ValueError("Expected two dimensional data array, "
                         f"got {data.ndim}")
    n_classifiers, n_datasets = data.shape
    if opacity is not None and len(opacity) != n_datasets:
        raise ValueError("Given number of opacity values do not correspond to "
                         "data.shape[1]")
    color = color if color is not None else (0.9, 0.5, 0.3)
    if n_classifiers == 1:
        if axes is not None:
            if not isinstance(axes, plt.Axes):
                raise ValueError(f"Expected a single axes, got {type(axes)}")
            return _hist(data, axis=axes, labels=labels, color=color)
        else:
            fig, axs = plt.subplots(1, 1)
            _hist(data, axis=axs, labels=labels, color=color)
            return fig, axs

    color_dl = color_dl if color_dl is not None else (0.2, 0.1, 0.7)
    color_ml = color_ml if color_ml is not None else (0.2, 0.4, 0.3)
    if axes is None:
        cols = n_plots = int(n_classifiers * (n_classifiers-1) / 2)
        if draw_hist:
            cols += n_classifiers
            n_plots = cols
        rows = 1
        if n_plots > max_cols:
            cols = max_cols
            rows = n_plots // max_cols
            if n_plots % max_cols != 0:
                rows += 1
        fig, axs = plt.subplots(rows, cols)
        axs = axs.reshape((rows, cols))
    else:
        if n_classifiers == 2 and isinstance(axes, plt.Axes):
            axs = np.ndarray([[axes]])
        elif n_classifiers > 2 and isinstance(axes, plt.Axes):
            raise ValueError("Expected an array of axes")
        axs = axes
    if draw_hist:
        indices = combinations_with_replacement(range(n_classifiers), r=2)
    else:
        indices = combinations(range(n_classifiers), r=2)
    i = j = 0
    for ii, jj in indices:
        if j == axs.shape[1]:
            j = 0
            i += 1
        axs[i, j].axis('square')
        axs[i, j].set_xlim([0, 1])
        axs[i, j].xaxis.set_ticks(
            ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["", "", "", "", "", ""],
        )
        axs[i, j].xaxis.set_ticks(
            ticks=[0.1, 0.3, 0.5, 0.7, 0.9],
            labels=["", "", "", "", ""],
            minor=True,
        )
        axs[i, j].yaxis.set_ticks(
            ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["", "", "", "", "", ""],
        )
        axs[i, j].yaxis.set_ticks(
            ticks=[0.1, 0.3, 0.5, 0.7, 0.9],
            labels=["", "", "", "", ""],
            minor=True,
        )
        axs[i, j].set_ylim([0, 1])
        axs[i, j].tick_params(which="both", direction="in")
        if ii == jj:
            _hist(
                data[ii],
                axis=axs[i, j],
                label=labels[ii],
                color=color,
            )
        else:
            _scattercomp(
                data[(ii, jj), :],
                axis=axs[i, j],
                opacity=opacity,
                labels=(labels[ii], labels[jj]),
                color=color,
                color_dl=color_dl,
                color_ml=color_ml,
            )
        j += 1
    while j < axs.shape[1]:
        axs[i, j].axis('off')
        j += 1
    if axes is None:
        return fig, axs
    return None
