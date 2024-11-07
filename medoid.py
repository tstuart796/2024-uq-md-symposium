from polyleven import levenshtein
from scipy.spatial.distance import pdist, squareform
import numpy as np

def get_sequence_distance_matrix(
    sequences
):
  sequences_formatted = [[x] for x in sequences] # turn it into a 2-D structure so pdist will accept it
  return squareform(pdist(sequences_formatted, lambda x, y: levenshtein(x[0], y[0])))


def find_medoid(
    sequences: list
) -> str:
  
  distances = get_sequence_distance_matrix(sequences)

  medoid_index = np.argmin(distances.sum(axis = 1)) # find index of sequence with least sum of Levenshtein distance to the others
  medoid = sequences[medoid_index]

  return medoid



