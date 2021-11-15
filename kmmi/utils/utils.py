import numpy as np

class Error(Exception):
    """Base class for other exceptions"""
    pass

class SeedNodeError(Error):
    """Exception raised if seed nodes are not fulfilling the desired selection criteria.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=""):
        self.message = message + 'Increase number of seed nodes in the immediate ' \
                       'neighborhood or remove v from force-selected nodes.'
        super().__init__(self.message)

def to_numpy_array(U: list) -> np.array:
    """
    Tansforms list U of lists u \in U, where each u represents a list of node 
    ids as integers, to numpy 2d array of size n x p where n is the length of 
    the list U and p is the length of the longest item u in the list. Note 
    that for shorter elements u s.t. len(u) < p, the rows are padded from the 
    right with -1.
    """
    n, k = len(U), len(sorted(U, key=len)[-1])
    Uc = np.zeros((n,k), dtype=np.int64) - 1 
    for i,u in enumerate(U):
        Uc[i,:len(u)] = u
        
    return Uc

def to_symmetric_matrix(E: np.array) -> np.array:
    E_hat = np.tril(E, k=-1) + np.tril(E.T, k=-1)
    return E_hat + E_hat.T