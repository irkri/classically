# CLASSIC - Classification Comparison
A python script for the comparison of paired metric data that is categorized.
The main application is the comparison of accuracy results coming from different
classification methods in machine learning.

__Classic__ implements the *critical difference diagram* as well as a special type of scatter plot
that is designed to compare multiple categories of data in a new kind of scatter matrix.

## Example
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
`classic.critical_difference_diagram`. The resulting plot can be seen below.
![critical difference diagram](example/cd_example.png)
It seems like classifier D is the best choice for the overall classification task. It works best
on the 14 chosen datasets, altough it's not the best classfier for every single dataset on its own.
But we can also see, that there is no significant (`alpha=0.05`) difference in the accuracy results
of classifier D and A. If D would be much more computationally expensive than A, then we should
consider choosing A as the better classifier.
