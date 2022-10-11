import numpy as np
import matplotlib.pyplot as plt

from classically import critical_difference_diagram

accuracy = np.array([
    [60, 81, 62, 19, 93, 54, 53, 41, 21, 97, 32, 82, 38, 75],
    [33, 68, 43, 23, 90, 43, 32, 20, 22, 86, 21, 82, 41, 73],
    [25, 64, 40, 10, 85, 39, 31, 19, 18, 90, 23, 78, 43, 71],
    [64, 84, 60, 26, 95, 60, 36, 37, 19, 95, 44, 84, 41, 84],
    [37, 68, 47, 18, 88, 37, 27, 25, 24, 79, 25, 83, 36, 64],
]) / 100.0

fig, ax = critical_difference_diagram(
    accuracy,
    labels=["A", "B", "C", "D", "E"],
)

plt.savefig("cdd_example.png", dpi=256)
plt.show()
