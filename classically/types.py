from typing import Any, Protocol, TypeVar

import numpy as np

_TFitScoreSelf = TypeVar("_TFitScoreSelf", bound="FitScoreClassifier")


class FitScoreClassifier(Protocol):

    def score(self, X: np.ndarray, y: np.ndarray) -> Any:
        ...

    def fit(
        self: _TFitScoreSelf,
        X: np.ndarray,
        y: np.ndarray,
    ) -> _TFitScoreSelf:
        ...
