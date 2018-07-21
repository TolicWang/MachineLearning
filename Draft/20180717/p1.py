import numpy as np
from scipy import sparse as sp
import scipy as sp
t=np.array([[0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, 0],
            [0.1, 0, 0, 0, 0, 0, 0, 0.4, 0.5, 0, 0, 0, 0, 0, 0],
            [0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0],
            [0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
cm=sp.csr_matrix(t)
dm=sp.csr_matrix(t)

print(cm)
print()
print(dm)
