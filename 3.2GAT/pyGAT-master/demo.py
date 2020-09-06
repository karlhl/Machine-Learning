import numpy as np
from scipy.sparse import csc_matrix
row = np.array([0, 2, 2, 0, 1, 2])
col = np.array([0, 0, 1, 2, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6])
a = csc_matrix((data, (row, col)), shape=(4, 4)).toarray()
print(a )