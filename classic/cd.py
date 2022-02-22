from typing import Optional, Sequence, overload

import networkx
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib import ticker

from classic.basic import get_color


@overload
def critical_difference_diagram(
    data: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    on_axis: None = None,
    alpha: float = 0.05,
) -> tuple[plt.Figure, plt.Axes]:
    ...


@overload
def critical_difference_diagram(
    data: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    on_axis: plt.Axes = None,
    alpha: float = 0.05,
) -> None:
    ...


def critical_difference_diagram(
    data: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    on_axis: Optional[plt.Axes] = None,
    alpha: float = 0.05,
) -> Optional[tuple[plt.Figure, plt.Axes]]:
    """Draws and returns a figure of a critical difference diagram based
    on the accuracies given to the class object. This type of plot was
    described in the paper
    'Statistical Comparison of Classifiers over Multiple Data Sets'
    by Janez Demsar, 2006.

    Args:
        data (np.ndarray): A numpy array with two dimensions. The first
            dimension iterates over different categories that are
            compared in this diagram, e.g. classifiers. The second
            dimension iterates over the values being used for the
            comparison, e.g. accuracy results.
        alpha (float, optional): Significance level used for doing
            pairwise Wilcoxon signed-rank tests. Defaults to 0.05.
        labels (Sequence[str], optional): The labels for categories of
            the data, e.g. the name of the classifiers used to create
            given accuracy results.
        on_axis (plt.Axes, optional): A matplotlib axis that the plot
            will be drawn on. If none is supplied, a new one will be
            created first and returned after finishing the diagram.

    Returns:
        Pyplot figure and axis with the diagram if ``on_axis`` is
        supplied, else None.
    """
    if data.ndim != 2:
        raise ValueError(
            f"Expected numpy array with two dimensions, got {data.ndim}"
        )
    n_classifiers = data.shape[0]
    if labels is None:
        labels = [f"Set {i+1}" for i in range(n_classifiers)]
    if data.shape[0] != len(labels):
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
    holm_bonferroni = alpha / np.arange(p_values.shape[0], 0, -1)
    significant = (p_values[p_order] <= holm_bonferroni)[p_order.argsort()]
    print(f"{significant=}")

    # calculate average ranks of classifiers over all datasets
    avg_ranks = (n_classifiers - stats.rankdata(data, axis=0) + 1).mean(axis=1)
    avg_ranks_order = avg_ranks.argsort()[::-1]
    print(f"{avg_ranks_order=}")

    # initialize and configure plot
    width = 6 + 0.3 * max(map(len, labels))
    height = 1.0 + n_classifiers * 0.1
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    if on_axis is not None:
        ax = on_axis

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

    half = int(np.ceil(n_classifiers / 2))

    # visual configurations
    rank_xshift = 0.02 * (highest_rank-lowest_rank)
    label_xshift = 0.05 * (highest_rank-lowest_rank)
    label_offset = 0.01 * (highest_rank-lowest_rank)
    first_marking = 0.6
    markings_vspace = 0.35 * 1/half
    markings_color = (0.15, 0.15, 0.15, 1.0)
    cliques_color = get_color(1) + (0.9, )

    # draw left branching markings
    for i, index in enumerate(avg_ranks_order[:half]):
        ax.axvline(
            x=avg_ranks[index],
            ymin=first_marking + (half-i-1)*markings_vspace,
            ymax=1.0,
            c=markings_color,
            lw=2.0,
        )
        ax.axhline(
            y=first_marking + (half-i-1)*markings_vspace,
            xmin=(half-i-1) * label_xshift / (highest_rank-lowest_rank),
            xmax=(highest_rank-avg_ranks[index]) / (highest_rank-lowest_rank),
            c=markings_color,
            lw=2.0,
        )
        ax.text(
            x=highest_rank - rank_xshift - (half-i-1)*label_xshift,
            y=first_marking + (half-i-1)*markings_vspace,
            s=f"{avg_ranks[index]:.2f}",
            ha="left",
            va="bottom",
            size=8,
        )
        ax.text(
            x=highest_rank - (half-i-1)*label_xshift + label_offset,
            y=first_marking + (half-i-1)*markings_vspace,
            s=f"{labels[index]}",
            ha="right",
            va="center",
            size=14,
        )

    # draw right branching markings
    for i, index in enumerate(avg_ranks_order[half:]):
        ax.axvline(
            x=avg_ranks[index],
            ymin=first_marking + i*markings_vspace,
            ymax=1.0,
            c=markings_color,
            lw=2.0,
        )
        ax.axhline(
            y=first_marking + i*markings_vspace,
            xmin=(highest_rank-avg_ranks[index]) / (highest_rank-lowest_rank),
            xmax=1.0 - i * label_xshift / (highest_rank-lowest_rank),
            c=markings_color,
            lw=2.0,
        )
        ax.text(
            x=lowest_rank + rank_xshift + i*label_xshift,
            y=first_marking + i*markings_vspace,
            s=f"{avg_ranks[index]:.2f}",
            ha="right",
            va="bottom",
            size=8,
        )
        ax.text(
            x=lowest_rank + i*label_xshift - label_offset,
            y=first_marking + i*markings_vspace,
            s=f"{labels[index]}",
            ha="left",
            va="center",
            size=14,
        )

    # get cliques of significant test results by building an adjacency
    # matrix and using the networkx package
    adjacency_matrix = np.zeros((n_classifiers, n_classifiers))
    connect_at = np.where(~significant)
    indexing = np.array(np.triu_indices(n_classifiers, k=1))
    for index in connect_at:
        i, j = indexing[:, index]
        adjacency_matrix[i, j] = 1
    print(adjacency_matrix)
    cliques = [
        clique for clique in networkx.find_cliques(
            networkx.Graph(adjacency_matrix)
        ) if len(clique) > 1
    ]
    fig, newax = plt.subplots()
    networkx.draw(networkx.Graph(adjacency_matrix), ax=newax)
    print(f"{cliques=}")

    # draw the cliques, i.e. connect classifiers that don't have a
    # significant difference
    i = 1
    if len(cliques) < 4:
        first_clique_line = 0.9 + (len(cliques) + 4) / 100
    else:
        first_clique_line = 0.97
    clique_line_diff = (1 - (first_marking + (half-1)*markings_vspace))
    clique_line_diff -= 0.001
    if len(cliques) > 0:
        clique_line_diff /= len(cliques)
    clique_line_y = first_clique_line
    for clique in cliques:
        xmin = (
            (highest_rank - avg_ranks[avg_ranks_order[min(clique)]])
            / (highest_rank - lowest_rank)
        )
        xmax = (
            (highest_rank - avg_ranks[avg_ranks_order[max(clique)]])
            / (highest_rank - lowest_rank)
        )
        ax.axhline(
            y=clique_line_y,
            xmin=xmin,
            xmax=xmax,
            color=cliques_color,
            linewidth=4.0,
        )
        clique_line_y -= clique_line_diff

    if on_axis is None:
        return fig, ax
    return None
