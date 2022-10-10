import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split

from classic import plot_feature_score
from classic.score import score_features

np.random.seed(111)

dataset = datasets.load_iris()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = map(
    np.array, train_test_split(X, y, test_size=0.2)
)

print(score_features(X_train, pca_components=1))

classifier = RidgeClassifierCV()

fig, ax = plot_feature_score(
    (X_train, y_train, X_test, y_test),
    labels=dataset.feature_names,
    classifier=classifier,
)

plt.savefig("score_example.png", dpi=128, bbox_inches="tight")
plt.show()
