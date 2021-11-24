import numpy as np
from numba import njit

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

@njit
def mean_ndiag(B):
    n = B.shape[0]
    return (B.sum()-np.diag(B).sum())/(n*(n-1))

def to_symmetric_matrix(E: np.array) -> np.array:
    E_hat = np.tril(E, k=-1) + np.tril(E.T, k=-1)
    return E_hat + E_hat.T

flatten = lambda x : list(chain.from_iterable(x))
overlap_coefficient = lambda A,B: len(A & B) / np.min([len(A),len(B)])

@njit
def prune_by_edge_weight(E: np.array, threshold: float=1e-3):
    """
    Prune edges for which weight is below given threshold value. 
    """
    n = E.shape[0]
    Et = np.zeros((n,n)) 
    for i in range(n):
        for j in range(n):
            if E[i,j] < threshold:
                Et[i,j] = 0.0
            else:
                Et[i,j] = E[i,j]
    return Et

@njit
def unique_nodes(U: np.array, n: int) -> np.array:
    """Function to find unique nodes in U.
    
    Notes
    -----
    This implementation is ca. 40-50x faster than np.unique(U) 
    or set(U.ravel()), but requires knowledge about the number 
    of nodes in the network. 
    """
    set_u = np.array([False]*n)
    for u in U:
        for v in u:
            if v != -1:
                set_u[v] = True
    return np.where(set_u)[0]