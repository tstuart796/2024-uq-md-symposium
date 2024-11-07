import numpy as np
from scipy.spatial.distance import pdist, squareform

def pos_to_dmat(pos):
  pos_formatted = [pos[i] for i in pos]
  return squareform(pdist(pos_formatted, "euclidean"))



# This function computes the mean squared elementwise difference between matrices, normalising the matrices first.

# Why do we have to normalise? Otherwise the result depends on the scale of the matrices.
# For example, if we multiply one of the matrices by 10, the result will very different.
# Some algorithms might simply produce distance matrices with bigger numbers, throwing off the comparison.
# What we really care about is the distribution of distances within the matrix.

# That means we have to normalise the matrices. I've chosen to scale them such that the mean value across all 
# positions in the matrix is 1. 
# In other words, the sum of all the elements in the matrix should be equal to the number of elements in the matrix.
# Therefore our scaling factor for the matrices will be M.size() / M.sum().
# Once both matrices are normalised like this, we can make a fair comparison between them.

# Mean squared error = ((mat_2 * scaling_factor2 - mat_1 * scaling factor1)**2).sum() / size
def normalised_mse(mat_1, mat_2):
  sum_1 = mat_1.sum()
  sum_2 = mat_2.sum()

  # I'm doing this weird cross-multiplication to avoid making a division early, and hence avoiding taking the difference between small floating point numbers.
  # (For numerical stability -- probably doesn't matter)
  # This is equivalent to the Mean squared error formula above.
  scaled_squared_error = (mat_2 * sum_1 - mat_1 * sum_2) ** 2 
  return scaled_squared_error.size * scaled_squared_error.sum()/(sum_1**2 * sum_2**2)
