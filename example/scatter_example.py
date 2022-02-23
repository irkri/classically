import numpy as np
import matplotlib.pyplot as plt

from classic import scatter_comparison

accuracy = np.array([
    [60, 81, 62, 19, 93, 54, 53, 41, 21, 97, 32, 82, 38, 75],
    [33, 68, 43, 23, 90, 43, 32, 20, 22, 86, 21, 82, 41, 73],
    [25, 64, 40, 10, 85, 39, 31, 19, 18, 90, 23, 78, 43, 71],
    [64, 84, 60, 26, 95, 60, 36, 37, 19, 95, 44, 84, 41, 84],
    [37, 68, 47, 18, 88, 37, 27, 25, 24, 79, 25, 83, 36, 64],
]) / 100.0

fig, ax = scatter_comparison(
    accuracy[(0, 1, 3), :],
    labels=["A", "B", "D"],
    draw_hist=False,
    max_cols=3,
)

plt.savefig("scatter_example.png", dpi=512, bbox_inches='tight')
plt.show()
