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
