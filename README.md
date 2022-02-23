# CLASSIC - Classification Comparison
A python module for the comparison of paired metric data that is categorized.
The main application is the comparison of accuracy results coming from different
classification methods in machine learning.

__Classic__ implements the *critical difference diagram* as well as a special type of scatter plot
that is designed to compare multiple categories of data in a new kind of scatter matrix.

## Example
The files for generating the plots in the following example can be found in the [example](/example) folder.

Imagine that we have five different classification methods tested on 14 different datasets.
Every classifiers returns an accuracy result on each test set in the corresponding dataset.
We collect the results in a table like this:

Classifier |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
A          | 0.60 | 0.81 | 0.62 | 0.19 | 0.93 | 0.54 | 0.53 | 0.41 | 0.21 | 0.97 | 0.32 | 0.82 | 0.38 | 0.75 |
B          | 0.33 | 0.68 | 0.43 | 0.23 | 0.90 | 0.43 | 0.32 | 0.20 | 0.22 | 0.86 | 0.21 | 0.82 | 0.41 | 0.73 |
C          | 0.25 | 0.64 | 0.40 | 0.10 | 0.85 | 0.39 | 0.31 | 0.19 | 0.18 | 0.90 | 0.23 | 0.78 | 0.43 | 0.71 |
D          | 0.64 | 0.84 | 0.60 | 0.26 | 0.95 | 0.60 | 0.36 | 0.37 | 0.19 | 0.95 | 0.44 | 0.84 | 0.41 | 0.84 |
E          | 0.37 | 0.68 | 0.47 | 0.18 | 0.88 | 0.37 | 0.27 | 0.25 | 0.24 | 0.79 | 0.25 | 0.83 | 0.36 | 0.64 |

We load this table in a `numpy array` of shape `(5, 14)` and call the function
`classic.critical_difference`. The resulting plot can be seen below.
![critical difference diagram](example/cd_example.png)
Markings on this number line represent the average ranks of one classifier based on his accuracy over all datasets. The lowest rank corresponds to the highest accuracy. Classifiers are connected by a horizontal line if they do not have a significant difference. This significance is based on post-hoc Wilcoxon signed rank tests for each pair of classifiers.

Therefore, it seems like classifier D is the best choice for the overall classification task.
It works best on the 14 chosen datasets, altough it's not the best classfier for every single dataset on its own.
But we can also see, that there is no significant (`alpha=0.05`) difference in the accuracy results
of classifier D and A. If D would be much more computationally expensive than A, then we should
consider choosing A as the better classifier.
For a more elaborate decision we could directly compare the best three classifiers A, B and D using
the function `classic.scatter_comparison.`
![scatter comparison](example/scatter_example.png)
Points above the diagonal line represent datasets that are better classified by the method in the upper left corner. A horizontal and vertical line indicates the mean accuracy of the corresponding classifier. A solid line marks the higher mean.

A choice can now be easily made for the comparison of classifier A and B as well as B and D. We also see that A is better than D in mean accuracy but that D has a big advantage on one dataset that is well beyond the diagonal line for five percent difference.

The datasets could now be further analyzed by, for example, looking at the number of training and test instances. An option for setting the opacity value of points in the scatterplots accordingly is available.
