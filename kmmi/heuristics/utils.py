from time import process_time
from datetime import timedelta as td
import numpy as np
from numba import *
from kmmi.heuristics.initialize import *

def __to_len_classes(ss):
    n_ss = {}
    for i,s in enumerate(ss):
        n_s = len(s)
        if n_s not in n_ss:
            n_ss[n_s] = []
        n_ss[n_s].append(i)
    return n_ss

@njit
def __svns_score(H_w, Ho_w, H, Ho, k):
    return (H_w / Ho_w) + (k - (H & Ho).sum()) / k

@njit
def __update_degree_vecs(A, alpha, beta, xi, xj, inplace=False):
    alpha_p = alpha if not inplace else alpha.copy()
    beta_p = beta if not inplace else beta.copy()
    for y in range(A.shape[0]):
        if y != xj:
            alpha_p[y] = alpha[y] - A[y,xi] + A[y,xj]
            beta_p[y] = beta[y] + A[y,xi] - A[y,xj]
    return alpha_p, beta_p

@njit
def __create_bvns_array(A):
    """Compute neighbor array for bvns such that ith row corresponds to node i and 
    indeces of nodes adjacent to i are the first elements in the row, while end of
    the rows are padded with -1.
    """
    n = A.shape[0]
    
    Ap = np.zeros((n,n), dtype=np.int64) - 1 
    for i in range(n):
        nz = np.where(A[i,:])[0]
        n_nz = nz.shape[0]
        Ap[i,:n_nz] = nz
        Ap[i,n_nz:] = -1
    return Ap

@njit
def __create_beam_array(A_input, A_as, w_thres):
    """Compute a beam array out of adjacency matrix A. In a beam array each row 
    i will contain the indexes of all connected nodes for node i in sorted order  
    based on the link weight."""
    
    n = A_input.shape[0]
    
    A_beam = np.zeros((n,n)) - 1 
    maxlen = n
    for i in range(n):
        j = 0
        for k in A_as[i,:]:
            if A_input[i,k] >= w_thres:
                A_beam[i,j] = k
                j+=1
            else:
                if j < maxlen:
                    maxlen = j
                break
                
    return A_beam[:,:maxlen]

@njit
def __create_beam_array_constant_width(A, A_as, w_thres):
    """Compute a beam array out of adjacency matrix A. In a beam array each row 
    i will contain the indexes of all connected nodes for node i in sorted order  
    based on the link weight."""
    
    #print('Beam width set')
    n_beam = 6
    n = A.shape[0]
    
    A_beam = np.zeros((n,n_beam)) - 1
    maxlen = n
    for i in range(n):
        for j in range(n_beam):
            k = A_as[i,j]
            if A[i,k] > 0.0:
                A_beam[i,j] = k
                j+=1
            else:
                if j < maxlen:
                    maxlen = j
                break
                
    if maxlen < n_beam:
        A_beam = A_beam[:,:maxlen]
    
    return A_beam
