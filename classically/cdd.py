from typing import Optional, Sequence, overload

import networkx as nx
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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
            mode = "auto"
            if (nzeros := np.sum((data[i] - data[j]) == 0)) >= 1:
                if nzeros == len(data[i, :]):
                    p_values[c] = 1.0
                    continue
                mode = "approx"
            p_values[c] = stats.wilcoxon(
                data[i, :],
                data[j, :],
                zero_method='pratt',
                method=mode,
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
    labels: Optional[Sequence[str]] = ...,
    alpha: float = ...,
    holm_bonferroni: bool = ...,
    axis: None = ...,
    color_cliques: Optional[tuple[float, float, float, float]] = ...,
    color_markings: Optional[tuple[float, float, float, float]] = ...,
) -> tuple[Figure, Axes]:
    ...


@overload
def critical_difference_diagram(
    data: np.ndarray,
    labels: Optional[Sequence[str]] = ...,
    alpha: float = ...,
    holm_bonferroni: bool = ...,
    axis: Axes = ...,
    color_cliques: Optional[tuple[float, float, float, float]] = ...,
    color_markings: Optional[tuple[float, float, float, float]] = ...,
) -> None:
    ...


def critical_difference_diagram(
    data: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    holm_bonferroni: bool = True,
    axis: Optional[Axes] = None,
    color_cliques: Optional[tuple[float, float, float, float]] = None,
    color_markings: Optional[tuple[float, float, float, float]] = None,
) -> Optional[tuple[Figure, Axes]]:
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
        axis (Axes, optional): A matplotlib axis that the plot
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

    # get cd graph and find cliques in it
    G = critical_difference_graph(data, labels, alpha, holm_bonferroni)
    cliques = []
    for clq in nx.find_cliques(G):
        if len(clq) > 1:
            cliques.append(list(map(lambda l: labels.index(l), clq)))

    # calculate average ranks of classifiers over all datasets
    avg_ranks = (n_classifiers - stats.rankdata(data, axis=0) + 1).mean(axis=1)
    avg_ranks_order = avg_ranks.argsort()[::-1]

    # initialize and configure plot
    width = 6 + 0.3 * max(map(len, labels))
    height = 1.0 + n_classifiers * 0.05
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    if axis is not None:
        ax = axis

    min_rank = int(np.floor(avg_ranks.min()))
    max_rank = int(np.ceil(avg_ranks.max()))
    rank_span = max_rank - min_rank

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
    ax.set_xlim(max_rank, min_rank)
    ax.set_ylim(0.0, 1.0)

    half = int(np.ceil(n_classifiers / 2))

    # visual configurations for markings
    label_left = avg_ranks[avg_ranks_order[0]] + .06*rank_span
    label_right = avg_ranks[avg_ranks_order[-1]] - .06*rank_span
    label_hshift = .03*rank_span
    marking_vspacing = 1 / (half + 1)

    color_markings = (
        (.15, .15, .15, 1.) if color_markings is None else color_markings
    )

    # draw left branching markings
    for i, index in enumerate(avg_ranks_order[:half]):
        rank = avg_ranks[index]
        ax.plot(
            (rank, rank),
            ((half-i-1)*marking_vspacing, 1.0),
            "-",
            c=color_markings,
            lw=2.0,
            clip_on=False,
        )
        ax.plot(
            (label_left + i * label_hshift, rank),
            ((half-i-1)*marking_vspacing, (half-i-1)*marking_vspacing),
            "-",
            c=color_markings,
            lw=2.0,
            clip_on=False,
        )
        ax.text(
            x=label_left + i * label_hshift,
            y=(half-i-1) * marking_vspacing,
            s=f"{avg_ranks[index]:.2f}",
            ha="left",
            va="bottom",
            size=8,
        )
        ax.text(
            x=label_left + i * label_hshift,
            y=(half-i-1) * marking_vspacing,
            s=f"{labels[index]}",
            ha="right",
            va="center",
            size=14,
        )

    # draw right branching markings
    for i, index in enumerate(avg_ranks_order[half:]):
        j = half-i
        rank = avg_ranks[index]
        ax.plot(
            (rank, rank),
            (i*marking_vspacing, 1.0),
            "-",
            c=color_markings,
            lw=2.0,
            clip_on=False,
        )
        ax.plot(
            (label_right - j * label_hshift, rank),
            (i*marking_vspacing, i*marking_vspacing),
            "-",
            c=color_markings,
            lw=2.0,
            clip_on=False,
        )
        ax.text(
            x=label_right - j * label_hshift,
            y=i * marking_vspacing,
            s=f"{rank:.2f}",
            ha="right",
            va="bottom",
            size=8,
        )
        ax.text(
            x=label_right - j * label_hshift,
            y=i * marking_vspacing,
            s=f"{labels[index]}",
            ha="left",
            va="center",
            size=14,
        )

    # visual configurations for cliques
    color_cliques = (
        (.9, .5, .3, .9) if color_cliques is None else color_cliques
    )

    first_clique_line = 0.9 + (len(cliques)) / 100
    if len(cliques) >= 4:
        first_clique_line = 0.95
    clique_line_vspace = 1. - ((half - 1) * marking_vspacing)
    if len(cliques) > 0:
        clique_line_vspace /= len(cliques)

    # draw the cliques, i.e. connect classifiers that don't have a
    # significant difference
    clique_line_y = first_clique_line
    for clique in cliques:
        xmin = (
            max_rank - avg_ranks[max(clique, key=lambda i: avg_ranks[i])]
        ) / (max_rank - min_rank)
        xmax = (
            max_rank - avg_ranks[min(clique, key=lambda i: avg_ranks[i])]
        ) / (max_rank - min_rank)
        ax.axhline(
            y=clique_line_y,
            xmin=xmin,
            xmax=xmax,
            color=color_cliques,
            linewidth=4.0,
        )
        clique_line_y -= clique_line_vspace

    fig.subplots_adjust(top=.75)

    if axis is None:
        return fig, ax
    return None
