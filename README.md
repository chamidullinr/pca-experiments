# Principal Component Analysis (PCA) experiments

Principal Component Analysis (PCA) is a very useful and one of the common algorithms used 
in statistics and machine learning. This tool is widely used in various applications such as 
dimensionality reduction for visualization and analysis, compression, outlier detection and image processing.

PCA is one of my favorite tools that I have employed in various tasks, often for visualization purposes.
However, I have come to a realization that all this time I have been using it as a black-box with only shallow understanding of its concepts.
Thus, it has motivated me to create this repository with custom implementations of PCA.

Please note, this repository does not intend to describe full details around PCA.
Merely show some python code to help better understand how it is computed.
For a better and comprehensive source, I found **A Tutorial on Principal Component Analysis** [1] very useful.
  

### About PCA
Simply put, the method diagonalizes covariance matrix of input data.
The property of diagonal matrix is that all values are zeros except for values on diagonal which must be non-zero.
The method assumes that there is a linear relationship between variables of the input data and it removes the relationship between them.

There are several approaches of computing PCA:
1. Via **covariance matrix** - this is useful when number of features is **lower** than number of records.
   And it is easier to interpret the approach.
2. Via **scalar product matrix** - this is useful when number of features is **higher** than number of records.
3. Via **Singular Value Decomposition (SVD)** - this approach is used in practice the most (Scikit-Learn library utilizes SVD for PCA).

In this repository, there are covered **covariance matrix** and **SVD** approaches for computing PCA.
I have inspired from MatLab code in **A Tutorial on Principal Component Analysis** - Appendix B [1]
and Scikit-Learn library [2].


## References

1.  J. Shlens: [A Tutorial on Principal Component Analysis](https://arxiv.org/abs/1404.1100v1). Google research, 2014.
2. [Scikit-Learn - PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)


## Authors
**Rail Chamidullin** - chamidullinr@gmail.com  - [Github account](https://github.com/chamidullinr)