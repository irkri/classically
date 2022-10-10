from multiprocessing.sharedctypes import Value
from typing import Optional, Sequence, Union, overload

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from .types import FitScoreClassifier


def score_features(
    features: np.ndarray,
    pca_components: int = 1,
) -> np.ndarray:
    """Scores the given features based on a principal component
    analysis. The score of one feature is the sum of squared eigenvector
    values multiplied by the explained variance ratio of the single
    principal components.

    Args:
        features (np.ndarray): Two dimensional numpy array of shape
            ``(n_instances, n_features)``.
        pca_components (int, optional): The number of principal
            components to calculate.

    Returns:
        numpy.ndarray: A numpy array with the score of each feature.
    """
    pca = PCA(n_components=pca_components)
    pca.fit(features)
    return (
        pca.explained_variance_ratio_ @ pca.components_**2  # type: ignore
    )


@overload
def plot_feature_score(
    data: Union[
        np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    pca_components: int = ...,
    classifier: Optional[FitScoreClassifier] = ...,
    labels: Optional[Sequence[str]] = ...,
    restrict: Union[int, float] = ...,
    last: bool = ...,
    axis: None = ...,
    bar_color: Optional[tuple[float, float, float, float]] = ...,
    acc_color: Optional[tuple[float, float, float, float]] = ...,
) -> tuple[Figure, Axes]:
    ...


@overload
def plot_feature_score(
    data: Union[
        np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    pca_components: int = ...,
    classifier: Optional[FitScoreClassifier] = ...,
    labels: Optional[Sequence[str]] = ...,
    restrict: Union[int, float] = ...,
    last: bool = ...,
    axis: Axes = ...,
    bar_color: Optional[tuple[float, float, float, float]] = ...,
    acc_color: Optional[tuple[float, float, float, float]] = ...,
) -> None:
    ...


def plot_feature_score(
    data: Union[
        np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    pca_components: int = 1,
    classifier: Optional[FitScoreClassifier] = None,
    labels: Optional[Sequence[str]] = None,
    restrict: Union[int, float] = 20,
    last: bool = False,
    axis: Optional[Axes] = None,
    bar_color: Optional[tuple[float, float, float, float]] = None,
    acc_color: Optional[tuple[float, float, float, float]] = None,
) -> Optional[tuple[Figure, Axes]]:
    """Plots the :underline:`normalized` features scores calculated with
    :meth:`score_features` in a bar chart.
    If ``len(indices) > 20``, the method plots ``20`` features
    with the highest scores (see option 'restrict').
    Additionally the classification accuracies of the cumulative
    feature sets can be plotted by specifying a classifier.

    Args:
        data (tuple of arrays): A single array of features or a tuple
            of four arrays of a train/test split with features and
            targets: ``(X_train, y_train, X_test, y_test)``
        components (int): Number of principal components to
            calculate.
        indices (Sequence of int, optional): Sequence of feature
            indices to use in the PCA. Defaults to all features.
        classifier (FitScoreClassifier, optional): The classifier
            to use for the cumulative classification. A tuple of arrays
            has to be given as 'data' if the accuracy values should be
            plotted.
        targets (array, optional): Targets for the classification
            (y-values). Specify 'targets' and 'classifier' to get the
            accuracy values plotted.
        labels (sequence of str, optional): Sets descriptive labels for
            the single features on the corresponding bars.
        restrict (int | float, optional): Maximal number of
            features to be plotted. If a float is given, it is
            interpreted as the smallest allowed normalized feature
            score a feature can have to be included in the plot.
            Includes therefore all features with a score larger than
            or equal to ``restrict``. Defaults to 20.
        last (bool, optional): If set to True, plots the last
            `restrict` features instead of the first ones
            according to the ordered scores. Defaults to the first.
        axis (Axes, optional): Matplotlib axis to plot the data
            on. If None is given, a seperate figure and axis
            will be created and returned.

    Returns:
        Tuple of a matplotlib figure and axes holding the inserted
        plot or None if ``axis`` is provided.
    """
    fig, ax = (None, axis) if axis is not None else plt.subplots(1, 1)

    if classifier is not None:
        if not isinstance(data, tuple) or len(data) != 4:
            raise ValueError("If 'classifier' is supplied, 'data' has to be "
                             "a tuple of four arrays")
    X = data[0] if isinstance(data, tuple) else data

    scores = score_features(X, pca_components=pca_components)
    scores /= np.max(scores)
    scores_order = np.argsort(scores)[::-1]
    if isinstance(restrict, float):
        if last:
            raise ValueError(
                "If 'last' is set to True, 'restrict' has to be an integer"
            )
        restrict = np.sum(scores >= restrict)
    scores_order = scores_order[-restrict:] if last else (
        scores_order[:restrict]
    )
    scores_trunc = scores[scores_order]

    bar_color = bar_color if bar_color is not None else (.0, .39, .68, .75)
    ax.bar(
        range(len(scores_order)),
        scores_trunc,
        color=bar_color,
        label="Normalized\nFeature Score",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Feature")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([], minor=True)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    if labels is not None:
        for i, index in enumerate(scores_order):
            ax.annotate(
                labels[index],
                xy=(i, 0),
                xytext=(0, 5),
                textcoords="offset pixels",
                rotation=90,
                ha="center",
                va="bottom",
            )

    if classifier is not None:
        acc_color = acc_color if acc_color is not None else (.71, .09, .13, 1.)
        accs = []
        for i in range(scores_order.size):
            indices = (
                np.r_[scores_order[:-restrict], scores_order[:i+1]] if last
                else scores_order[:i+1]
            )
            classifier.fit(data[0][:, indices], data[1])
            accs.append(classifier.score(data[2][:, indices], data[3]))
        ax.plot(
            accs,
            marker="s",
            color=acc_color,
            label="Accuracy of\nCumulative\nFeature Set",
        )
        classifier.fit(data[0], data[1])
        acc = classifier.score(data[2], data[3])
        ax.hlines(
            [acc],
            xmin=ax.get_xlim()[0],
            xmax=ax.get_xlim()[1],
            color=acc_color,
        )
        ax.annotate(
            f"{acc:.2f}",
            xy=(ax.get_xlim()[0], acc),
            xytext=(10, 0),
            textcoords="offset pixels",
            va="center",
            ha="left",
            rotation=90,
            color=acc_color,
        )

    ax.legend(loc="best")

    if fig is not None:
        return fig, ax
    return None
