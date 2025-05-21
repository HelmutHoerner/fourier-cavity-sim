# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:52:15 2024

@author: Helmut Hoerner
"""
import os    
import numpy as np
#import jax
#jax.config.update("jax_enable_x64", True)
#import jax.numpy as jnp
import math
from scipy.sparse import diags, spmatrix
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import time
import multiprocessing 
import pickle
import portalocker
from collections import deque
from enum import Enum
from matplotlib.ticker import AutoLocator, ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, \
    Normalize, LinearSegmentedColormap
from joblib import Parallel, delayed
import pandas as pd
#import tempfile
import random
import string
import joblib
import gc
from functools import partial
#from matplotlib.ticker import MaxNLocator, FormatStrFormatter
#import cmath
#import scipy
#from scipy.sparse.linalg import inv


__version__ = "1.0.0"
#############################################################################
# Enums
#############################################################################
class Dir(Enum):
    # Enumeration for propagation directions
    LTR = 1 # left-to-right
    RTL = 2 # right-to-left
    
class Dir2(Enum):
    # Enumeration for propagation directions
    BOTH = 0 # both directions
    LTR = 1 # left-to-right
    RTL = 2 # right-to-left

class Side(Enum):
    # Enumertion for left or right side
    LEFT = 1 # left side
    RIGHT = 2 # right side
    
class Res(Enum):
    # Enumeration for Resolutions
    FOV = 1 # field-of-view-resolution
    TOT = 2 # Total resolution

class Path(Enum):
    # Enumerattion for Arm A or B in 4 port cavity
    A = 1 # path A (e.g. horizonal)
    B = 2 # path B (e.g. vertical)

#############################################################################
# Experimental functions
#############################################################################
#from concurrent.futures import ProcessPoolExecutor, as_completed

#def fourier_basis_func(N: int, ax, nx: int, ny: int, tot: bool, k_space_out: bool):
#        """
#        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#        % Creates the normalized (nx, ny) Fast-Fourier basis function  
#        % either in position space or in k-space
#        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#        % Input parameters:
#        % nx   ... mode number in x-direction
#        % ny   ... mode number in y-direction
#        % tot  ... if True, then for the whole grid, else for field-of-view only 
#        % k_space_out .. if true, output in k-space, otherwise in position space 
#        %
#        % Output:
#        % normalized basis function
#        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#        """
#        c = N // 2
#        psi = np.zeros((N, N), dtype=complex)
#        if c-nx>=0 and c-nx < N and c-ny>=0 and c-ny < N:    
#            psi[c-ny, c-nx]=1 # create normalized basis function in f-space
#        
#        if not k_space_out:
#            psi = ifft2_phys_spatial(psi, ax)
#        
#        return psi

#def arr_to_vec(X, N, ax, mode_idx, k_space_in, column_vec = True):        
#        """
#        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#        % Converts position-space array or k-space-array 
#        % into a Fourier-coefficient-vector with the coefficients ordered 
#        % according to the mode_numbers vector
#        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#        % Input:
#        % X ............ k-space or position-space input array
#        % k_space_in ... if true:  X is a k-space-array,
#        %                if false: X is a position-space-array. 
#        % column_vec ... optional. If true, a column vector is returned
#        %
#        % Output:
#        % vector with Fourier coefficients, ordered according to 
#        % mode_numbers vector
#        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#        """
#        
#        if not k_space_in:
#            # position-space-array as input -> 
#            # convert to spatial-frequency-space-array first
#            X = fft2_phys_spatial(X, ax)
#        
#        # convert nx, ny values from mode_numbers vector into 
#        # linear index vector for the input array
#        if column_vec:
#            return X[mode_idx[:,0], mode_idx[:,1]].reshape(-1, 1)
#        else:
#            return X[mode_idx[:,0], mode_idx[:,1]]
    
#def test_worker(i, mode_numbers, lens_mask, N, ax):
#    nx = mode_numbers[i, 0]
#    ny = mode_numbers[i, 1]
#    psi = fourier_basis_func(N, ax, nx, ny, True, False)
#    psi *= lens_mask
#    # return i, np.random.rand(10000)
#    return i, arr_to_vec(psi,N, ax, mode_numbers,  False, False)
   


###############################################################################
# Common Functions
###############################################################################
def keep_clsBlockMatrix_alive(x):
    for instance in clsBlockMatrix.get_instances():
        instance.keep_alive = x
    gc.collect()


def closest_multiple(a: float, b: float) -> float:
    """
    Returns the value that is the integer multiple of b closest to a.

    Parameters:
        a (float): The target value.
        b (float): The base value whose multiple we seek.

    Returns:
        float: Closest integer multiple of b to a.
    """
    n = round(a / b)
    return n * b

def delta_f_to_delta_lambda(delta_f_hz, lambda_center_m, n=1.0):
    """
    Convert frequency difference Δf to wavelength difference Δλ,
    using the center wavelength λ instead of frequency.

    Parameters:
    - delta_f_hz: frequency spacing Δf [Hz]
    - lambda_center_m: center wavelength λ [meters]
    - n: refractive index of the medium (default is 1.0 for vacuum)

    Returns:
    - delta_lambda_m: wavelength spacing Δλ [meters]
    """
    c = 299_792_458  # exact speed of light in vacuum [m/s]
    f_center = (c / n) / lambda_center_m  # convert λ to f
    delta_lambda = - (c / n) / (f_center ** 2) * delta_f_hz
    return delta_lambda

def polar_interpolation(z1, z2, w1=0.5):
    """
    Performs weighted polar interpolation between two complex numbers z1 and z2.

    Parameters:
        z1 (complex): First complex number.
        z2 (complex): Second complex number.
        w1 (float): Weight for z1 (default is 0.5).

    Returns:
        complex: Interpolated complex value.
    """
    # Second weight
    w2 = 1 - w1
    
    # Convert to polar form
    A1, phi1 = np.abs(z1), np.angle(z1)
    A2, phi2 = np.abs(z2), np.angle(z2)
    
    # Interpolated amplitude (linear interpolation)
    A_avg = w1 * A1 + w2 * A2

    # Compute weighted circular mean for phase
    phase_avg = np.angle(w1 * np.exp(1j * phi1) + w2 * np.exp(1j * phi2))

    # Reconstruct interpolated complex value
    return A_avg * np.exp(1j * phase_avg)


def find_closest_value(v, x):
    """
    v ... sorted vector with float values
    x ... value to be searched
    returns the closest value and its index
    """
    idx = np.searchsorted(v, x)

    # Check the boundary cases
    if idx == 0:
        closest_value = v[0]
        closest_idx = 0
    elif idx == len(v):
        closest_value = v[-1]
        closest_idx = len(v) - 1
    else:
        # Find the closest value by comparing the two nearest values
        if abs(v[idx - 1] - x) <= abs(v[idx] - x):
            closest_value = v[idx - 1]
            closest_idx = idx - 1
        else:
            closest_value = v[idx]
            closest_idx = idx

    return closest_value, closest_idx

def get_sample_vec(NoOfPoints: int, LR_range, center, stacked:bool, p = 1):
    """
    returns a vector  with NoOfPoints entries around center, 
    with the  values denser around the center. 
    Smalles value: center-LR_range
    Largest value: center+LR_Range
    If stacked==True, then the values are stacked around the center postion
    p defines how dense the values are 
    """
    if NoOfPoints % 2 == 0:
        NoOfPoints += 1
    
    if NoOfPoints == 1:
        return [center] 
    
    p = int(p)
    
    x = (np.arange(1, NoOfPoints + 1) - (NoOfPoints + 1) / 2) / ((NoOfPoints - 1) / 2)
    
    if p % 2 == 0:
        # even p
        y = x**p * np.sign(x) * LR_range + center
    else:
        y = x ** p * LR_range + center
    
    if stacked:
        B = -np.arange(1, (NoOfPoints - 1) // 2 + 1)
        A = -B
        A += (NoOfPoints - 1) // 2 + 1
        B += (NoOfPoints - 1) // 2 + 1
        AB = np.vstack((A, B))
        idx_vec = AB.reshape(-1, order='F')
        idx_vec = np.insert(idx_vec, 0, (NoOfPoints-1)/2+1)-1    
        y = y[idx_vec.astype(int)]
        
    return y


def find_best_match(ax, x):
    """" Find best matching value in vector ax for value x """
    # Compute the absolute differences between dx and each element in ax
    differences = np.abs(ax - x)
    # Find the index of the minimum difference
    min_index = np.argmin(differences)
    # Return the element in ax at the index of the minimum difference
    return ax[min_index]

def is_matrix(X):
    """ returns True if X is a numpy array or a scipi sparse matrix"""
    if isinstance(X, np.ndarray):
        return True
    elif isinstance(X, spmatrix):
        return True
    else:
        return False

def mat_minus(X, Y):
    """ 
    calculates X-Y, with X and Y being matrices,
    but allowing X and/or Y being also scalars
    if only X is scalar, it calculates I X - Y
    if only Y is scalar, it calculates X - I Y
    if X and Y are scalars, it calculates X-Y
    """
    if is_matrix(X) and is_matrix(Y):
         # X and Y are matrices
         return X-Y
    elif is_matrix(X) and (not is_matrix(Y)):
         # X is a matrix, Y is a scalar
         if Y == 0:
            return X
         else:
            return X - np.eye(X.shape[1])*Y
    elif (not is_matrix(X)) and is_matrix(Y):
        # X is a scalar, Y is a matrix
        if X == 0:
            return Y
        else:
            return np.eye(Y.shape[1])*X - Y
        # X and Y are both scalars
    return X - Y

def bmat2_minus(X, Y):
    """ 
    calculates X-Y, with X and Y being 2x2 block matrices
    """    
    if isinstance(X, clsBlockMatrix) and isinstance(Y, clsBlockMatrix):
        Z = clsBlockMatrix(2, X.file_caching, X.tmp_dir)
        for i in range(2):
            for j in range(2):
                Z.set_block(i, j, mat_minus(X.get_block(i,j),Y.get_block(i,j)))
        return Z
    else:
        A = mat_minus(X[0][0],Y[0][0])
        B = mat_minus(X[0][1],Y[0][1])
        C = mat_minus(X[1][0],Y[1][0])
        D = mat_minus(X[1][1],Y[1][1])
        return [[A, B],[C, D]]

def mat_plus(X, Y):
    """ calculates X+Y, with X and Y being matrices,
    but allowing X and/or Y being also scalars
    if only X is scalar, it calculates I X + Y
    if only Y is scalar, it calculates X + I Y
    if X and Y are scalars, it calculates X+Y
    """
    if is_matrix(X) and is_matrix(Y):
         # X and Y are matrices
         return X+Y
    elif is_matrix(X) and (not is_matrix(Y)):
         # X is a matrix, Y is a scalar
         if Y == 0:
            return X
         else:
            return X + np.eye(X.shape[1])*Y
    elif (not is_matrix(X)) and is_matrix(Y):
        # X is a scalar, Y is a matrix
        if X == 0:
            return Y
        else:
            return np.eye(Y.shape[1])*X + Y
    else:
        # X and Y are both scalars
        return X + Y

def bmat2_plus(X, Y):
    """ 
    calculates X+Y, with X and Y being 2x2 block matrices
    """    
    if isinstance(X, clsBlockMatrix) and isinstance(Y, clsBlockMatrix):
        Z = clsBlockMatrix(2, X.file_caching, X.tmp_dir)
        for i in range(2):
            for j in range(2):
                Z.set_block(i, j, mat_plus(X.get_block(i,j),Y.get_block(i,j)))
        return Z
    else:
        A = mat_plus(X[0][0],Y[0][0])
        B = mat_plus(X[0][1],Y[0][1])
        C = mat_plus(X[1][0],Y[1][0])
        D = mat_plus(X[1][1],Y[1][1])
        return [[A, B],[C, D]]

def mat_plus3(X, Y, Z):
    """ matrix addition between matrices X, Y, and Z
    also allowing X or Y to be scalars (floats) """   
    return mat_plus(mat_plus(X,Y),Z)

def mat_mul(X, Y):
    """ matrix multiplication between matrices X and Y
    also allowing X or Y to be scalars """  
    if is_matrix(X) and is_matrix(Y):
        # X and Y are matrices
        return X@Y   
    
    elif is_matrix(X) and (not is_matrix(Y)):
        # X is a matrix, Y is a scalar
        if Y == 0:
            return 0
        elif Y == 1:
            return X
        else:
            return X*Y
    
    elif (not is_matrix(X)) and is_matrix(Y):
        # X is a scalar, Y is a matrix
        if X == 0:
            return 0
        elif X == 1:
            return Y
        else:
            return X*Y
    else:
        # X and Y are both scalars
        return X*Y

def mat_inv_X_mul_Y(X, Y):
    """ efficiently calculates inv(X) Y """
    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        return np.linalg.solve(X, Y)
    else:
        return mat_mul(mat_inv(X), Y)

def mat_div_bm(X, Y, p = None, msg = ""):
    """ 
    matrix division X/Y. X and Y must be callable functions
    to be used together with clsBlockMatrix as follows:
        
    M01 = partial(M.get_block, 0, 1)
    M11 = partial(M.get_block, 1, 1)
    tmp1 = mat_div_bm(M01, M11)
    
    this saves memory compared to  
    mat_div(M.get_block(0,1), M.get_block(1,1))
    """
    if not p is None:
        if not msg == "":
            p.tic_reset(2, False, msg)
    
    Y_inv = mat_inv(Y()) 
    gc.collect()
    if not p is None:
        p.tic()
    result = mat_mul(X(), Y_inv) 
    del Y_inv
    if not p is None:
        p.tic()
    return result

def mat_div(X, Y):
    """ matrix division X/Y, allowing matrices X and Y to be scalars """
    if isinstance(X, np.ndarray):
        # X is a numpy array
        if isinstance(Y, np.ndarray):
            # X is a numpy array, Y is a numpy array
            return np.linalg.solve(Y.T, X.T).T
        elif isinstance(Y, spmatrix):
            #  X is a numpy array, Y is a sparse (diagonal) matrix
            return mat_mul(X, mat_inv(Y))
        else:
            # X is a numpy array, Y is a scalar
            return X/Y
        
    elif isinstance(X, spmatrix):
        # X is a sparse (diagonal) matrix
        if isinstance(Y, np.ndarray):
            # X is a sparse (diagonal) matrix, Y is a numpy array
            return mat_mul(X, mat_inv(Y))
        elif isinstance(Y, spmatrix):
            #  X is a sparse (diagonal) matrix, Y is a sparse (diagonal) matrix
            return mat_mul(X, mat_inv(Y))
        else:
            # X is a sparse (diagonal) matrix, Y is a scalar
            return X/Y
    else:
        # X is a scalar
        return X * mat_inv(Y)
    
def mat_mul3(X, Y, Z):
    """ matrix multilication between matrices X, Y, and Z
    also allowing X or Y to be scalars (floats) """   
    return mat_mul(mat_mul(X,Y),Z)

def mat_is_zero(X):
    """ returns true, if X is zero, where X may be a numpy matrix,
    a sparse dagonal or a scalar"""
    if isinstance(X, np.ndarray):
        # X is a full numpy matrix
        return np.all(X == 0)
    elif isinstance(X, spmatrix):
        # X is a sparse diagonal matrix
        return np.all(X.data == 0)
    else:
        # X is a scalar
        return (X==0)

def bmat2_is_zero(X):
    """ 
    returns true, if the 2x2 block matrix X is zero
    """
    if isinstance(X, clsBlockMatrix):
        return X.is_zero()
    else:
        for row in X:
            for elem in row:
                if not mat_is_zero(elem):
                    return False
        return True
    
def mat_inv(X):
    """ matrix inversion, also allowing X to be scalar or sparse diagnonal"""    
    if isinstance(X, np.ndarray):
        # X is a full numpy matrix
        return np.linalg.inv(X) 
        #return jax.device_get(jnp.linalg.inv(X))
    elif isinstance(X, spmatrix):
        # X is a sparse diagonal matrix
        return diags(1.0/X.diagonal())
    else:
        # X is a scalar
        return 1.0/X

def mat_conj_transpose(X):
    """ 
    calculates the conjugate transpose of matrix X, 
    also allowing X to be scalar or sparse diagnonal
    """    
    if isinstance(X, np.ndarray):
        # X is a full numpy matrix
        return np.conj(X.T) 
    elif isinstance(X, spmatrix):
        # X is a sparse diagonal matrix
        return X.conj()
    else:
        # X is a scalar
        return np.conj(X)

def bmat_conj_transpose(X):
    """ 
    calculates the conjugate transpose of a block matrix X
    (2x2 or 4x4, using clsBlockMatrix)
    """   
    Z = clsBlockMatrix(X.dim, X.file_caching, X.tmp_dir)
    for i in range(X.dim):
        for j in range(X.dim):
            Z.set_block(j, i, mat_conj_transpose(X.get_block(i,j)))
            gc.collect()
    return Z

def bmat2_conj_transpose(X):
    """ 
    calculates the conjugate transpose of the 2x2 block matrix X, 
    with X being a list
    """       
    A = mat_conj_transpose(X[0][0])
    B = mat_conj_transpose(X[0][1])
    C = mat_conj_transpose(X[1][0])
    D = mat_conj_transpose(X[1][1])
    return [[A, C],[B, D]]

def get_bmat4_quadrants(X):
    """
    takes the 4x4 block matrix X  =
    [[X11, X12, X13, X14], ..., [X41, X42, X43, X44]]
    and returns the four quadrants (each a 2x2 block matrix)
    """
    A = [[X[0][0], X[0][1]], [X[1][0], X[1][1]]]
    B = [[X[0][2], X[0][3]], [X[1][2], X[1][3]]]
    C = [[X[2][0], X[2][1]], [X[3][0], X[3][1]]]
    D = [[X[2][2], X[2][3]], [X[3][2], X[3][3]]]
    return A, B, C, D

def bmat4_from_quadrants(A, B, C, D):
    """
    takes the four quadrand block matrices A, B, C, D 
    and returns a 4x4 block matrix
    """
    S1 = [A[0][0], A[0][1], B[0][0], B[0][1]]
    S2 = [A[1][0], A[1][1], B[1][0], B[1][1]]
    S3 = [C[0][0], C[0][1], D[0][0], D[0][1]]
    S4 = [C[1][0], C[1][1], D[1][0], D[1][1]]
    return [S1, S2, S3, S4]

def bmat4_conj_transpose(X):
    """ 
    calculates the conjugate transpose of the 4x4 block matrix X, 
    with X being a list
    """    
    X11, X12, X21, X22 = get_bmat4_quadrants(X)
    A = bmat2_conj_transpose(X11)
    B = bmat2_conj_transpose(X12)
    C = bmat2_conj_transpose(X21)
    D = bmat2_conj_transpose(X22)
    return bmat4_from_quadrants(A, B, C, D)

def bmat2_M_to_S(M, p = None, msg = ""):
    """ 
    converts the transfer matrix  into the scattering matrix S
    both M and S are block matrices
    if p is set to a clsProgressPrinter instance, then tics are sent
    if msg != "" the the progress printer is resetted
    """
    
    if not p is None:
        if not msg == "":
            p.tic_reset(4, False, msg)
    
    if isinstance(M, clsBlockMatrix): 
        S = clsBlockMatrix(2, M.file_caching, M.tmp_dir)
                
        # C = inv(M22)
        S.set_block(1, 0, mat_inv(M.get_block(1,1)))
        gc.collect()
        if not p is None:
            p.tic()
        
        # A = M12 * M22_inv
        S.set_block(0, 0, mat_mul(M.get_block(0,1),S.get_block(1,0)))
        gc.collect()
        if not p is None:
            p.tic()

        # B = mat_minus(M11, mat_mul(M12_M22_inv, M21)) 
        tmp = mat_mul(S.get_block(0,0), M.get_block(1,0))
        gc.collect()
        S.set_block(0,1, mat_minus(M.get_block(0,0), tmp))
        del tmp
        gc.collect()
        if not p is None:
            p.tic()

        #  D = - mat_mul(M22_inv, M21)
        tmp2 = mat_mul(S.get_block(1,0), M.get_block(1,0))
        gc.collect()
        S.set_block(1,1, -tmp2)
        del tmp2
        gc.collect()
        if not p is None:
            p.tic()
        
        return S
    else:
        M11, M12, M21, M22 = M[0][0], M[0][1], M[1][0], M[1][1]
        
        M22_inv = mat_inv(M22)
        if not p is None:
            p.tic()
            
        M12_M22_inv = mat_mul(M12, M22_inv)
        if not p is None:
            p.tic()
        
        A = M12_M22_inv
        B = mat_minus(M11, mat_mul(M12_M22_inv, M21))
        if not p is None:
            p.tic()
            
        C = M22_inv
        D = - mat_mul(M22_inv, M21)
        if not p is None:
            p.tic()
    
        return [[A, B], [C, D]]

def bmat4_M_to_S(M, p = None, msg = ""):
    """ 
    converts the 4x4 transfer block matrix  
    into the 4x4 scattering block matrix S
    """
    
    if isinstance(M, clsBlockMatrix):
        S = clsBlockMatrix(4, M.file_caching, M.tmp_dir)
        
        M11 = M.get_quadrant(1)
        M12 = M.get_quadrant(2)
        M21 = M.get_quadrant(3)
        M22 = M.get_quadrant(4)
        
        if not p is None:
            if msg == "":
                msg = "converting M-matrix to S-matrix"
        
            p.tic_reset(5+4, False, msg) 
            
        M22_inv = bmat2_inv(M22, p, msg)
        
        M12_M22_inv = bmat_mul(M12, M22_inv)
        if not p is None:
            p.tic()
        
        S.set_quadrant(1, M12_M22_inv)
        S.set_quadrant(2, bmat2_minus(M11, bmat_mul(M12_M22_inv, M21)))
        if not p is None:
            p.tic()
        
        S.set_quadrant(3, M22_inv)
        S.set_quadrant(4, bmat_flip_sign(bmat_mul(M22_inv, M21)))
        if not p is None:
            p.tic()
        
        return S
        
        
    else:
        M11, M12, M21, M22 = get_bmat4_quadrants(M)
        
        M22_inv = bmat2_inv(M22)
        M12_M22_inv = bmat_mul(M12, M22_inv)
        
        A = M12_M22_inv
        B = bmat2_minus(M11, bmat_mul(M12_M22_inv, M21))
        C = M22_inv
        D = bmat_flip_sign(bmat_mul(M22_inv, M21))
        
        return bmat4_from_quadrants(A, B, C, D)
    
def bmat_mul(X,Y, p = None, msg = ""):
    """ 
    multiplies two block matrices X, Y of arbitrary size 
    X, Y must be given as nested lists or clsBlockMatrix
    if p is set to a clsProgressPrinter instance, then tics are sent
    if msg != "" the the progress printer is resetted
    """
        
    if isinstance(X, clsBlockMatrix) and isinstance(Y, clsBlockMatrix):
        if X.dim != Y.dim:
            raise ValueError("X and Y must have the same dimensions")
        
        
        if not p is None:
            if not msg == "":
                p.tic_reset(X.dim * Y.dim * X.dim , False, msg)
        
        Z = clsBlockMatrix(X.dim, X.file_caching, X.tmp_dir)
        # Perform matrix multiplication
        for i in range(X.dim):
            for j in range(Y.dim):
                for k in range(X.dim):                         
                    Z.set_block(i,j, 
                        mat_plus(Z.get_block(i,j), 
                        mat_mul(X.get_block(i,k), Y.get_block(k,j))))
                    gc.collect()
                    if not p is None:
                        p.tic()
        
        return Z
    
    else:
        # Number of rows in X
        X_rows = len(X)
        # Number of cols in X / rows in Y
        X_cols = len(X[0])
        # Number of cols in Y
        Y_cols = len(Y[0])
        
        # Result matrix Z with dimensions (X_rows x Y_cols)
        Z = [[0]*Y_cols for _ in range(X_rows)]
        
        # Check if multiplication is possible
        if len(X[0]) != len(Y):
            raise ValueError("X's columns must match Y's rows for multiplication.")
        
        if not p is None:
            if not msg == "":
                p.tic_reset(X_rows * X_cols * Y_cols , False, msg)
        
        # Perform matrix multiplication
        for i in range(X_rows):
            for j in range(Y_cols):
                for k in range(X_cols):  # or len(Y)
                    Z[i][j] = mat_plus(Z[i][j], mat_mul(X[i][k], Y[k][j]))
                    if not p is None:
                        p.tic()
        
        return Z

def bmat_flip_sign(X):
    """ flips the sign in every entry of the block matrix Y"""
    if isinstance(X, clsBlockMatrix):
        Z = clsBlockMatrix(X.dim, X.file_caching, X.tmp_dir)        
        for i in range(X.dim):
            for j in range(X.dim):
                Z.set_block(i, j, - X.get_block(i, j))
                gc.collect()
        return Z
    
    else:
        return [[-x for x in sublist] for sublist in X]


def mat_inv_X_mul_Y_bm(X, Y):
    """ 
    efficiently calculates inv(X) Y 
    X and Y must be callable functions
    to be used together with clsBlockMatrix as follows:
        
    M01 = partial(M.get_block, 0, 1)
    M11 = partial(M.get_block, 1, 1)
    tmp1 = mat_inv_X_mul_Y_bm(M01, M11)    
    """
    X_inv = mat_inv(X()) 
    gc.collect()
    result = mat_mul(X_inv, Y()) 
    del X_inv
    return result    

def bmat2_inv(X, p=None, msg=""):
    """
    calcuates the inverse of the 2x2 block matrix X   
    given as X = [[A, B],[C, D]]
    if p is set to a clsProgressPrinter instance, then tics are sent
    if msg != "", then the progess printer is resetted
    """
    
    if not p is None:
        if not msg == "":
            p.tic_reset(5, False, msg)
    
    if isinstance(X, clsBlockMatrix):
        Z = clsBlockMatrix(X.dim, X.file_caching, X.tmp_dir)
        
        if X.block_is_zero(0, 0) and X.block_is_zero(1, 1):
            Z.set_block(0, 1, mat_inv(X.get_block(1,0)))
            gc.collect()
            if not p is None:
                p.tic()
                p.tic()
                
            Z.set_block(1, 0, mat_inv(X.get_block(0,1)))
            gc.collect()
            if not p is None:
                p.tic()
                p.tic()
                p.tic()
                
            return Z
        
        elif X.block_is_zero(0, 1) and X.block_is_zero(1, 0):
            Z.set_block(0, 0, mat_inv(X.get_block(0,0)))
            gc.collect()
            if not p is None:
                p.tic()
                p.tic()
                
            Z.set_block(1, 1, mat_inv(X.get_block(1,1)))
            gc.collect()
            if not p is None:
                p.tic()
                p.tic()
                p.tic()
                
            return Z
        
        else:
            tmp = clsBlockMatrix(2, X.file_caching, X.tmp_dir)
            
            # invA_B = tmp[0,0]
            X00 = partial(X.get_block, 0, 0)
            X01 = partial(X.get_block, 0, 1)
            tmp.set_block(0, 0, mat_inv_X_mul_Y_bm(X00, X01)) 
            
            gc.collect()
            if not p is None:
                p.tic()
                
            #inv_D_minus_C_invA_B = Z[1, 1]
            a = mat_mul(X.get_block(1,0), tmp.get_block(0,0))
            gc.collect()
            b = mat_minus(X.get_block(1,1), a)
            del a
            gc.collect()
            Z.set_block(1, 1, mat_inv(b))
            del b
            gc.collect()
            if not p is None:
                p.tic()
                
            #invD_C = tmp[0, 1]
            X11 = partial(X.get_block, 1, 1)
            X10 = partial(X.get_block, 1, 0)
            tmp.set_block(0, 1, mat_inv_X_mul_Y_bm(X11, X10))
            gc.collect()
            if not p is None:
                p.tic()                
            
            a = mat_mul(X.get_block(0,1), tmp.get_block(0,1))
            gc.collect
            b = mat_minus(X.get_block(0,0), a)
            del a
            gc.collect
            Z.set_block(0, 0, mat_inv(b))
            del b
            gc.collect
                    
            if not p is None:
                p.tic()

            a = mat_mul(tmp.get_block(0,0), Z.get_block(1,1))        
            Z.set_block(0, 1, -a)
            del a
            gc.collect()
            
            a = mat_mul(tmp.get_block(0,1), Z.get_block(0,0))                
            Z.set_block(1, 0, -a)
            del a
            gc.collect()
            
            del tmp
            if not p is None:
                p.tic()
            
            return Z
        
        
    else:
        A = X[0][0]
        B = X[0][1]
        C = X[1][0]
        D = X[1][1]
        
                
        if mat_is_zero(A) and mat_is_zero(D):
            BB = mat_inv(C)
            if not p is None:
                p.tic()
                p.tic()
                
            CC = mat_inv(B)
            if not p is None:
                p.tic()
                p.tic()
                p.tic()
                
            return [[0, BB],[CC,0]]
        
        elif mat_is_zero(B) and mat_is_zero(C):
            AA = mat_inv(A)
            if not p is None:
                p.tic()
                p.tic()
                
            DD = mat_inv(D)
            if not p is None:
                p.tic()
                p.tic()
                p.tic()
                
            return [[AA, 0],[0, DD]]
        
        else:
            
            invA_B = mat_inv_X_mul_Y(A, B)
            if not p is None:
                p.tic()
                
            inv_D_minus_C_invA_B = mat_inv(mat_minus(D, mat_mul(C, invA_B)))
            if not p is None:
                p.tic()
                
            invD_C = mat_inv_X_mul_Y(D, C)
            if not p is None:
                p.tic()
                
            inv_A_minus_B_invD_C=  mat_inv(mat_minus(A, mat_mul(B, invD_C)))
            if not p is None:
                p.tic()
                
            AA = inv_A_minus_B_invD_C
            BB = -mat_mul(invA_B, inv_D_minus_C_invA_B)
            CC = -mat_mul(invD_C, inv_A_minus_B_invD_C)
            DD = inv_D_minus_C_invA_B
            if not p is None:
                p.tic()
            
            return [[AA, BB], [CC, DD]]
        
def bmat4_inv(X, p=None, msg=""):
    """
    caluates the inverse of the 4x4 block matrix X   
    given as X = [[X11, X12, X13, X14],, ..., [X41, X42, X43, X44]]
    if p is set to a clsProgressPrinter instance, then tics are sent
    """
    if not p is None:
        if msg == "":
            msg = "inverting 4x4 block matrix"
    
    if isinstance(X, clsBlockMatrix):
        A = X.get_quadrant(1, False)
        B = X.get_quadrant(2, False)
        C = X.get_quadrant(3, False)
        D = X.get_quadrant(4, False)
        
        Z = clsBlockMatrix(4, X.file_caching, X.tmp_dir)
        
        if bmat2_is_zero(A) and bmat2_is_zero(D):
            if not p is None:
                p.tic_reset(2*5, False, msg)      
            Z.set_quadrant(2, bmat2_inv(C, p), False)            
            Z.set_quadrant(3, bmat2_inv(B, p), False)
            return Z
        
        elif bmat2_is_zero(B) and bmat2_is_zero(C):
            if not p is None:
                p.tic_reset(2*5, False, msg)
            Z.set_quadrant(1, bmat2_inv(A, p))
            Z.set_quadrant(4, bmat2_inv(D, p))
            return Z
        
        else:
            if not p is None:
                p.tic_reset(4*5+1, False, msg)
                
            invA_B = bmat_mul(bmat2_inv(A, p), B)
            inv_D_minus_C_invA_B = bmat2_inv(bmat2_minus(D, bmat_mul(C, invA_B)), p)
            
            invD_C = bmat_mul(bmat2_inv(D, p), C)
            inv_A_minus_B_invD_C=  bmat2_inv(bmat2_minus(A, bmat_mul(B, invD_C)), p)
            
            Z.set_quadrant(1, inv_A_minus_B_invD_C)
            Z.set_quadrant(2, bmat_flip_sign(bmat_mul(invA_B, inv_D_minus_C_invA_B)))
            Z.set_quadrant(3, bmat_flip_sign(bmat_mul(invD_C, inv_A_minus_B_invD_C)))
            Z.set_quadrant(4, inv_D_minus_C_invA_B)
            if not p is None:
                p.tic()
                
            return Z
    
    else:
        A, B, C, D = get_bmat4_quadrants(X)
        Z = [[0,0],[0, 0]]
       
        
        if bmat2_is_zero(A) and bmat2_is_zero(D):
            if not p is None:
                p.tic_reset(2*5, False, msg)
            BB = bmat2_inv(C, p)
            CC = bmat2_inv(B, p)
            return bmat4_from_quadrants(Z, BB, CC, Z)
        
        elif bmat2_is_zero(B) and bmat2_is_zero(C):
            if not p is None:
                p.tic_reset(2*5, False, msg)
            AA = bmat2_inv(A, p)
            DD = bmat2_inv(D, p)
            return bmat4_from_quadrants(AA, Z, Z, DD)
        
        else:
            if not p is None:
                p.tic_reset(4*5+1, False, msg)
            invA_B = bmat_mul(bmat2_inv(A, p), B)
            inv_D_minus_C_invA_B = bmat2_inv(bmat2_minus(D, bmat_mul(C, invA_B)), p)
            
            invD_C = bmat_mul(bmat2_inv(D, p), C)
            inv_A_minus_B_invD_C=  bmat2_inv(bmat2_minus(A, bmat_mul(B, invD_C)), p)
            
            AA = inv_A_minus_B_invD_C
            BB = bmat_flip_sign(bmat_mul(invA_B, inv_D_minus_C_invA_B))
            CC = bmat_flip_sign(bmat_mul(invD_C, inv_A_minus_B_invD_C))
            DD = inv_D_minus_C_invA_B
            if not p is None:
                p.tic()
                
            return bmat4_from_quadrants(AA, BB, CC, DD)
        
def bmat_mul3(X, Y, Z, p = None, msg = ""):
    """ Multiplies the block matrices X, Y, and Z 
    given as nested lists  """
    if not p is None:
        if not msg == "":
            if isinstance(X, clsBlockMatrix):
                steps = 2 * X.dim * Y.dim * X.dim
            else:
                X_rows = len(X)        
                X_cols = len(X[0])        
                Y_cols = len(Y[0])
                steps = 2 * X_rows * X_cols * Y_cols
            p.tic_reset(steps , False, msg)
    return bmat_mul(bmat_mul(X,Y, p, ""), Z, p, "")    

def fft2_phys_spatial(X, ax):
    """ 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2D spatial Fast Fourier Transform according to 
    % Physics Spatial Fourier Transform Convention 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Input parameters:
    % X  ... position-space array array holding spatial function values
    % ax ... vector with axis coordinates in m
    %
    % Output:
    % fourier_coeff ... array with spatial-frequency Fourier coefficients
    %
    % if the input is normalied, so that 
    % trapz(ax, trapz(ax, X .* conj(X), 2)) == 1,
    % then the output is also normalized so that norm(fourier_coeff)==1 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    ax_len = max(ax)-min(ax)
    return np.fft.fftshift(np.fft.ifft2(X * ax_len))    
    
def ifft2_phys_spatial(X, ax):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2D spatial Inverse Fast Fourier Transform according to 
    % Physics Spatial Fourier Transform Convention 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Input parameters:
    % X  ... array with spatial-frequency Fourier coefficients
    % ax ... vector with axis coordinates in m
    % 
    % Output:  
    % Y ... position-space-array holding spatial function values
    %
    % if the input is normalied, so that norm(fourier_coeff)==1 
    % then the output is also normalized so that 
    % trapz(ax, trapz(ax, X .* conj(X), 2)) == 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """            
    ax_len = max(ax)-min(ax)
    return np.fft.fft2(np.fft.ifftshift(X)) / ax_len





###############################################################################
# clsBlockMatrix
###############################################################################
class clsBlockMatrix:
    #instance_count = 0   
    instances = []
    
    def __init__(self, dimension: int, file_caching: bool, tmp_dir="", name = ""):        
        #clsBlockMatrix.instance_count += 1
        clsBlockMatrix.instances.append(self)
        self.name = name
        if dimension==4:
            self.__dim = 4
        else:
            self.__dim = 2
        self.__file_caching = file_caching
        self.__tmp_dir = tmp_dir
        self.keep_tmp_files = False
        self._by_ref = False
        self.keep_alive = False
        
        # ram memory for holding information which is not stored in a file
        self.__ram = [[0 for _ in range(dimension)] for _ in range(dimension)]
        # this holds information if the respective block is zero
        self.__file_block_is_zero = [[True for _ in range(dimension)] for _ in range(dimension)]
        
        if file_caching:
            # create random file names for each block
            self.__file_names = \
                [["tmp_"+''.join(random.choices(string.ascii_letters + string.digits, k=28))+".pkl"
                 for _ in range(dimension)] for _ in range(dimension)]
            
            # add tmp folder
            if tmp_dir != "":
                if tmp_dir[-1] != os.sep:
                    tmp_dir += os.sep
                for i in range(self.__dim):
                    for j in range(self.__dim):
                        self.__file_names[i][j] = tmp_dir + self.__file_names[i][j]         
        
        q1 = [[0,0],[0,1],[1,0],[1,1]]
        q2 = [[0,2],[0,3],[1,2],[1,3]]
        q3 = [[2,0],[2,1],[3,0],[3,1]]
        q4 = [[2,2],[2,3],[3,2],[3,3]]
        self.__quadrants = [q1, q2, q3, q4]

    #def __copy__(self):
    #    clsBlockMatrix.copy_count += 1
    #    return clsBlockMatrix(self.__dim, self.__file_caching, self.__tmp_dir, self.name)

    #def __deepcopy__(self, memo):
    #    clsBlockMatrix.deep_copy_count += 1
    #    return clsBlockMatrix(self.__dim, self.__file_caching, self.__tmp_dir, self.name)    
    
    def set_file_block_is_zero(self, i, j, x):
        self.__file_block_is_zero[i][j] = x    
        
    def __del__(self):
        #print("*************************")
        #print(f'Instance is destroyed')
        #print(self.__file_names)
        #print("*************************")
        self.clear()
        if self in clsBlockMatrix.instances:
            clsBlockMatrix.instances.remove(self)
        
    @classmethod
    def get_instances(cls):
        # Class method to return the list of current instances
        return cls.instances
    
    @property 
    def file_caching(self):
        return self.__file_caching
    
    @property
    def tmp_dir(self):
        return self.__tmp_dir
    
    @property
    def dim(self):
        return self.__dim
    
    @property
    def file_names(self):
        if self.__file_caching:
            return self.__file_names
        else:
            return []
    
    def get_quadrant(self, quadrant:int, clone = False):
        """
        In a 4x4 Block matrix: returns the quadrant 1, 2, 3 or 4
        as a 2x2 Block Matrix.
        (1... top left, 2 ... top right, 3 ... bottom left, 4 ..bottom right)
        """
        if self.__dim !=4:
            raise ValueError("matrix is not a 4x4 block matrix")
        if quadrant<1 or quadrant>4:
            raise ValueError("qudrant must be 1, 2,3 or 4")
        
        X = clsBlockMatrix(2, self.__file_caching, tmp_dir = self.__tmp_dir)
        X._by_ref = True
        
        quadrant = quadrant - 1
        if clone:
            # clone the blocks
            for i, j in self.__quadrants[quadrant]:
                X.set_block(i%2, j%2, self.get_block(i, j), True)
        else:
            # not cloning the blocks
            for i, j in self.__quadrants[quadrant]:
                if self.__file_caching:
                    # copy temp file name and location
                    X.set_tmp_file_name(i%2, j%2, self.__file_names[i][j])
                if not self.__ram[i][j] is None:
                    # if scalar valie: copy valiue
                    X.set_block(i%2, j%2, self.__ram[i][j])
                else:
                    # not a sclar value: copy "is zero" information
                    X.set_file_block_is_zero(i%2, j%2,  self.__file_block_is_zero[i][j])
                
        return X
    
    def is_saved_in_tmp_file(self, i, j):
        return (self.__ram[i][j] is None)
    
    
    def set_quadrant(self, quadrant, X, clone = False):
        """ sets values of Quadrant from X """
        
        
        if quadrant<1 or quadrant>4:
            raise ValueError("qudrant must be 1, 2,3 or 4")
        
        quadrant = quadrant - 1
        if  self.__dim == 2:            
            i, j = self.__quadrants[0][quadrant]
            self.set_block(i, j, X, clone)
        
        elif self.__dim == 4:
            if clone:
                for i, j in self.__quadrants[quadrant]:
                    self.set_block(i, j, X.get_block(i%2, j%2), True)
               
            
            else:
                X._by_ref = True
                               
                for i, j in self.__quadrants[quadrant]:
                    if self.__file_caching:
                        self.__file_names[i][j] = X.get_tmp_file_name(i%2, j%2)
                        self.__ram[i][j] = None
                        
                    if X.is_saved_in_tmp_file(i%2, j%2):
                        self.__file_block_is_zero[i][j] = X.block_is_zero(i%2, j%2)
                    else:
                        self.set_block(i, j, X.get_block(i%2, j%2))                        
                        
    
    def set_from_quadrants(self, A, B, C, D, clone = False):
        """ Sets values from Quadrants A, B, C, D """
        if self.__dim == 2:
            self.set_block(0, 0, A, clone)
            self.set_block(0, 1, B, clone)
            self.set_block(1, 0, C, clone)
            self.set_block(1, 1, D, clone)
        
        elif self.__dim == 4:
            self.set_quadrant(1, A, clone)
            self.set_quadrant(2, B, clone)
            self.set_quadrant(3, C, clone)
            self.set_quadrant(4, D, clone)
                
            
    def set_tmp_file_name(self, i, j, name):
        self.__file_names[i][j] = name
        self.__ram[i][j] = None
        
    def get_tmp_file_name(self, i, j):
        return self.__file_names[i][j]
    
    
    def clear(self):
        """ Clears RAM and deletes all temporary files """        
        self.__ram = [[0 for _ in range(self.__dim)] for _ in range(self.__dim)]
        if self.__file_caching:            
            if not self.keep_tmp_files:
                if not self.keep_alive:
                    if not self._by_ref:
                        # delete all temporary files (if not protected)
                        for line in self.__file_names:
                            for file_name in line:                        
                                if os.path.exists(file_name):
                                    os.remove(file_name)
        
    def get_block(self, line, col):
        """
        getting block
        """
        if not self.__file_caching:
            # no file caching: Get Block from ram
            return self.__ram[line][col]
        
        elif not self.__ram[line][col] is None:
            # file caching is active, but there is a scalar value
            # stored in ram nevertheless
            return self.__ram[line][col]
            
        else:
            # file caching: load from tmp pfile
            block = joblib.load(self.__file_names[line][col])
            #with open(self.__file_names[line][col], 'rb') as file:
            #    block = pickle.load(file)
            gc.collect()
            return block
    
    def clone(self, X):
        """
        Cloning from the other block matrix
        """
        if self.__dim != X.dim:
            raise ValueError("block matrices must have the same dimensions")
        
        for i in range(self.__dim):
            for j in range(self.__dim):
                self.set_block(i, j, X.get_block(i, j), True)
        
    def set_block(self, line: int, col: int, X, clone = False):
        """ 
        assigning block X to block matrix 
        line and col ... from 0 to dim-1
        """
        copy = False
        if clone:
            if is_matrix(X):
                if not self.__file_caching:
                    copy = True
        
        if not self.__file_caching:
            # no file caching: store in ram
            if copy:
                self.__ram[line][col] = X.copy()
            else:
                self.__ram[line][col] = X
        
        elif not is_matrix(X):
            # file caching active, scalar value: store in RAM           
            self.__ram[line][col] = X
            # delete tmp file (if it exists)
            file_name = self.__file_names[line][col]
            if os.path.exists(file_name):
                os.remove(file_name)
        
        else:
            # file caching active, matrix value: store in tmp file  
            self.__file_block_is_zero[line][col] = mat_is_zero(X)
            self.__ram[line][col] = None
            joblib.dump(X, self.__file_names[line][col])  
            gc.collect()
        X = None
    
    def block_is_zero(self, line: int, col: int):       
        """ 
        returns True, if the according block is zero
        """
        if self.__ram[line][col] is None:
            return self.__file_block_is_zero[line][col]
        else:
            return mat_is_zero(self.__ram[line][col])
    
    def is_zero(self):       
        """ 
        returns True, if whole matrix is zero
        """
        for i in range(self.__dim):
            for j in range(self.__dim):
                if not mat_is_zero(self.get_block(i, j)):
                    return False
            
        return True
        

###############################################################################
# clsProgressPrinter
# for progress outputs
###############################################################################
class clsProgressPrinter:
    def __init__(self):
        self.__tic_goal_count = 1
        self.__tic_cur_count = 0
        self.__tic_start_time = 0
        self.__max_time_betw_tic_outputs = 10
        self.__max_time_without_tic_duration_output = 60
        self.__last_tic_time = 0
        self.__last_tic_output_time = 0
        self.__indent_spaces = 3
        self.__start_times = deque()
        self.__min_time_to_show = 3
        self.__always_print_first_and_last = True
        self.__first_message = ""
        self.__print_count = 0
        self.__step_duration_printed = False
        self.__silent = False
     
    @property
    def silent(self):
        return self.__silent
    
    @silent.setter
    def silent(self, new_val):
        self.__silent = new_val
       
    @property
    def min_time_to_show(self):
        return self.__min_time_to_show
    
    @min_time_to_show.setter
    def min_time_to_show(self, new_val):
        self.__min_time_to_show = new_val
    
    @property
    def indent(self):
        if len(self.__start_times)==0:
            return ""
        else:
            return " " * self.__indent_spaces * (len(self.__start_times)-1)
     
    def push_print(self, msg, start_time=0):
        """ pushes a new sub-level and prints the message """
        if start_time == 0:
            self.__start_times.append(time.perf_counter())       
        else:
            self.__start_times.append(start_time)   
            
        if not self.__silent:
            print(self.indent + msg)
    
    def print(self, msg):
        """ prints the message """  
        if not self.__silent:
            print(self.indent + msg)
        
    def pop(self):
        """ 
        ends the sub-task by printing 'done' and the duration 
        returns elapsed time in seconds
        """
        if len(self.__start_times)==0:
            return
        
        indent = self.indent
        start_time = self.__start_times.pop()
        elapsed_time =  time.perf_counter()- start_time  
        if not self.__silent:
            if elapsed_time>=self.__min_time_to_show:
                print(f"{indent}done ({elapsed_time:.1f} seconds)")
        
        return elapsed_time
    
    @property
    def max_time_betw_tic_outputs(self):
        return self.__max_time_betw_tic_outputs
    
    @max_time_betw_tic_outputs.setter
    def max_time_betw_tic_outputs(self, t):
        
        if t<0:
            t = 0        
        self.__max_time_betw_tic_outputs = t
        
    @property
    def max_time_without_tic_duration_output(self):
        return self.__max_time_without_tic_duration_output
    
    @max_time_without_tic_duration_output.setter
    def max_time_without_tic_duration_output(self, t):
        if t<0:
            t = 0
        self.__max_time_without_tic_duration_output = t
    
    def tic_reset(self, goal_count: int, always_print_first_and_last=True, msg = ""):
        self.__first_message = msg
        self.__tic_goal_count = goal_count
        self.__tic_cur_count = 0
        self.__tic_start_time = time.perf_counter()
        self.__last_tic_time = self.__tic_start_time
        if always_print_first_and_last:
            self.__last_tic_output_time = 0
        else:
            self.__last_tic_output_time = self.__tic_start_time
        self.__always_print_first_and_last = always_print_first_and_last
        self.__print_count = 0
        self.__step_duration_printed = False
            
    def tic(self):        
        self.__tic_cur_count += 1
        cur_time = time.perf_counter()
        diff_time = cur_time - self.__last_tic_output_time        
        print_flag = False
        
        if diff_time>self.__max_time_betw_tic_outputs:            
            print_flag = True
        elif self.__tic_cur_count >= self.__tic_goal_count:
            if self.__print_count>0:
                print_flag = True
            else:
                print_flag = self.__always_print_first_and_last
            
        if print_flag:
            self.__print_count += 1
            if self.__print_count == 1 and not self.__first_message=="":
                self.push_print(self.__first_message, self.__tic_start_time)
            self.__last_tic_output_time = cur_time
            percent = self.__tic_cur_count / self.__tic_goal_count * 100
            indent = self.indent + " " * self.__indent_spaces
            tic_duration_time = cur_time - self.__last_tic_time 
            if tic_duration_time >=  self.__max_time_without_tic_duration_output or self.__step_duration_printed:
                self.__step_duration_printed = True
                msg = f"{indent}{percent:.1f}% done (step {self.__tic_cur_count} of {self.__tic_goal_count}, {tic_duration_time:.1f} seconds)"
            else:
                msg = f"{indent}{percent:.1f}% done (step {self.__tic_cur_count} of {self.__tic_goal_count})"
            print(msg)
            if self.__tic_cur_count == self.__tic_goal_count and self.__print_count>0:
                self.pop()
        self.__last_tic_time =cur_time
        

###############################################################################
# clsBmat2_parmul
# parallel multiplication of 2x2 block matrices
###############################################################################
class clsBmat2_parmul:
    def __init__(self, X, Y, par_processes: int):
        self.__X = X
        self.__Y = Y
        self.__par_processes = par_processes
        self.__Z11 = 0
        self.__Z12 = 0
        self.__Z21 = 0
        self.__Z22 = 0
        self._result = None
        
    def _helper1(self, idx: int):
        i = ((idx & 4) >> 2)
        j = ((idx & 2) >> 1)
        k = idx & 1
        return mat_mul(self.__X[i][k], self.__Y[k][j])
    
    def _helper2(self, idx: int):
        i = (idx << 1)
        return mat_plus(self._result[i], self._result[i + 1])
          
 
    
    def mult(self):     
        keep_clsBlockMatrix_alive(True)
        self._result = Parallel(n_jobs=self.__par_processes)(delayed(self._helper1)(idx) for idx in range(8))
        Z = Parallel(n_jobs=self.__par_processes)(delayed(self._helper2)(idx) for idx in range(4))
        keep_clsBlockMatrix_alive(False)
        return [[Z[0], Z[1]],[Z[2], Z[3]]]
        
    
###############################################################################
# clsPoolSingleton
# Multi-Processing Pool Singleton
###############################################################################
class clsPoolSingleton:
    _instance = None
    _init_args = {}

    def __new__(cls, processes=None):
        if cls._instance is None:
            cls._instance = super(clsPoolSingleton, cls).__new__(cls)
            cls._init_args['processes'] = processes
            # Create the pool with the specified number of processes
            cls._instance.pool = multiprocessing.Pool(processes=processes)           
        
        return cls._instance.pool
    
    @classmethod
    def reset_pool(cls):        
        if cls._instance is not None:
            cls._instance.pool.close()
            cls._instance.pool.join()
            cls._instance = None
            cls._init_args = {}
        
    
###############################################################################
# clsGrid
# represents a grid in xy direction
###############################################################################
class clsGrid:
    """ Represents the xy-Grid """
    def __init__(self, cavity):
        self.__res_fov = 10
        self.__res_tot = 10
        self.__length_fov = 0.001
        self.__length_tot = 0.001
        self.__cavity = cavity
        self.__axis_fov = None
        self.__axis_tot = None
        self.__k_axis_fov = None
        self.__k_axis_tot = None
        self.__n_axis_fov = None
        self.__n_axis_tot = None
        self.__mode_numbers_fov = None
        self.__mode_numbers_tot = None
        self.__mode_indices_fov = None
        self.__mode_indices_tot = None
        self.__fov_offset_x = 0
        self.__fov_offset_y = 0
        self.__pos_offset_x = 0
        self.__pos_offset_y = 0
        # fov_pos: 0 .. center, 1... top, 2 .. bottom, 3 .. left , 4 ... right
        self.__fov_pos = 0 
        self.set_res(10, 10, 0.001)
        
    
    def stretch_y(self, old_array, y0, c):
        """
        Compress (or stretch, if c>1) the rows of `old_array` around the 
        vertical position `y0` by a factor `c`.  
        The result is the same shape as `old_array`, but rows
        are resampled (interpolated) from the original.
        Integrated intensity stays constant

        :param old_array: 2D numpy array (NxM or NxN)
        :param y0:        The reference position about which we stretch
        :param c:         Stretch factor (<1 compresses, >1 expands)
        :return:          New array with the same shape, but y-resampled
        """
        # center coordinate index (can be float)
        y0_idx = self.get_center_index(False) + self.dist_to_pixels(y0)
        N = old_array.shape[0]
        new_array = self.empty_grid(False)
        
        amp_factor = 1 / math.sqrt(c)
        for y_new in range(N):
            # Distance from center line in the "new" space
            dy = (y_new - y0_idx)

            # Map that to old_array coordinates
            y_old_float = y0_idx + dy / c   # continuous float coordinate in old array

            # Get integer indices around y_old_float
            i1 = int(np.floor(y_old_float))
            i2 = i1 + 1

            w2 = y_old_float - i1  # fractional part (weight 2)
            i1_outside = (i1<0 or i1 > N-1)
            i1_valid = (not i1_outside)
            i2_outside = (i2<0 or i2 > N-1)
            i2_valid = (not i2_outside)
            w1 = 1.0 - w2 # weight 1
            
            
            # Linear interpolation
            if i1_valid and i2_valid:
                #new_array[y_new, :] = (1.0 - alpha)*old_array[i1, :] + alpha*old_array[i2, :]
                new_array[y_new, :] = polar_interpolation(
                    old_array[i1, :], old_array[i2, :], w1) * amp_factor
            elif i1_valid and i2_outside:
                new_array[y_new, :] = w1*old_array[i1, :] * amp_factor
            elif i1_outside and i2_valid:
                new_array[y_new, :] = w2*old_array[i2, :] * amp_factor

        return new_array
    
    def stretch_x(self, old_array, x0, c):
        """
        Compress (or stretch, if c>1) the rows of `old_array` around the 
        horizontal position `x0` by a factor `c`.  
        The result is the same shape as `old_array`, but rows
        are resampled (interpolated) from the original.
        Integrated intensity stays constant

        :param old_array: 2D numpy array (NxM or NxN)
        :param x0:        The reference position about which we stretch
        :param c:         Stretch factor (<1 compresses, >1 expands)
        :return:          New array with the same shape, but y-resampled
        """
        # center coordinate index (can be float)
        x0_idx = self.get_center_index(False) + self.dist_to_pixels(x0)
        N = old_array.shape[0]
        new_array = self.empty_grid(False)
        amp_factor = 1 / math.sqrt(c)
        
        for x_new in range(N):
            # Distance from center column in the "new" space
            dx = (x_new - x0_idx)

            # Map that to old_array coordinates
            x_old_float = x0_idx + dx / c   # continuous float coordinate in old array

            # Get integer indices around y_old_float
            i1 = int(np.floor(x_old_float))
            i2 = i1 + 1

            w2 = x_old_float - i1  # fractional part
            i1_outside = (i1<0 or i1 > N-1)
            i1_valid = (not i1_outside)
            i2_outside = (i2<0 or i2 > N-1)
            i2_valid = (not i2_outside)
            w1 = 1.0 - w2 # weight 1
            
            # Linear interpolation
            if i1_valid and i2_valid:
                #new_array[:, x_new] = (1.0 - alpha)*old_array[:, i1] + alpha*old_array[:, i2]
                new_array[:, x_new] = polar_interpolation(
                    old_array[:, i1], old_array[:, i2], w1) * amp_factor
            elif i1_valid and i2_outside:
                new_array[:, x_new] = w1 * old_array[:, i1] * amp_factor
            elif i1_outside and i2_valid:
                new_array[:, x_new] = w2 * old_array[:, i2] * amp_factor

        return new_array
    
    def dist_to_pixels(self, dist):
        """ converts the distace dist to picels (integer value)"""
        pix_per_m = self.res_tot/self.length_tot # pixels per m
        return int(round(dist * pix_per_m))
    
    @property
    def pos_offset_x(self):
        """ Shift in x center position because of fov_pos setting """
        return self.__pos_offset_x
    
    @property
    def pos_offset_y(self):
        """ Shift in y center position because of fov_pos setting """
        return self.__pos_offset_y
    
    @property
    def fov_pos(self):
        """ 
        FOV position in the larger grid 
        0 .. center, 1... top, 2 .. bottom, 3 .. left , 4 ... right
        """
        return self.__fov_pos
    
    @fov_pos.setter
    def fov_pos(self, p):
        """ 
        FOV position in the larger grid 
        0 .. center, 1... top, 2 .. bottom, 3 .. left , 4 ... right
        """
        if p < 0:
            p = 0
        if p > 4:
            p = 4
        self.__fov_pos = p
        self.__set_pov_offset()
    
    @property
    def cavity(self):
        return self.__cavity
    
    @property
    def factor(self):
        """ Factor by which side-length in total res. is larger than fov res."""
        return(self.__res_tot/self.__res_fov)
    
    def __smoothstep(self, t):
            return 6 * t**5 - 15 * t**4 + 10 * t**3

    def _bump_function(self, r, a, epsilon, black_value):
        if r <= a - epsilon:
            return 1
        elif r >= a + epsilon:
            return black_value
        else:
            t = (r - (a - epsilon)) / (2 * epsilon)
            return (1-self.__smoothstep(t)) * (1 - black_value) + black_value
    
    def get_soft_aperture_mask(self, aperture, epsilon, black_value):
        """ 
        Creates an full resolution aperture mask with soft border
        - for any radius < (aperture/2 - epsilon) the value is 1
        - for any radius > (aperture/2 + epsilon) the value is 0
        - for any radius between aperture/2 - epsilon and aperture/2 + epsilon
          the value is between 0 and 1
        """
        r = aperture/2
        ax = self.axis_tot
        x, y = np.meshgrid(ax, ax)
        R = np.sqrt(x**2 + y**2)
        mask = np.vectorize(self._bump_function, otypes=[float])(
            R, r, epsilon, black_value)
        return mask
    
    def get_aperture_mask(self, aperture, anti_alias_factor, black_value = 0,
                          consider_pos_offset = False):
        lens_R = aperture/2
        ax = self.axis_tot
        
        if consider_pos_offset:
            pix_width = ax[1]-ax[0]
            x_off = self.pos_offset_x * pix_width
            y_off = self.pos_offset_y * pix_width
        
        # Create lens pupil with 1s inside radius and 0s outside
        if anti_alias_factor==4:
            # create anti-aliased aperture mask
            # quadruple resolution axis
            N = self.res_tot
            delta = (ax[1]-ax[0])/4
            ax2 = np.empty(N * 2)
            ax2[0::2] = ax - delta
            ax2[1::2] = ax + delta   
            delta2 = (ax2[1]-ax2[0])/4
            ax4 = np.empty(N * 4)
            ax4[0::2] = ax2 - delta2
            ax4[1::2] = ax2 + delta2  
                
            # quadruple resolution coordinate grid
            x4, y4 = np.meshgrid(ax4, ax4)
            if consider_pos_offset:
                x4 -= x_off
                y4 -= y_off
                
            # quadruple resolution radial coordinate grid
            R4 = np.sqrt(x4**2 + y4**2)  
            lens_pupil4 = (R4 <= lens_R).astype(float)    
            if black_value != 0:
                lens_pupil4 = np.where(lens_pupil4 == 0, black_value, lens_pupil4)
            N = self.res_tot
            # now downscale for anti-aliasing
            lens_pupil = lens_pupil4.reshape(N, 4, N, 4).mean(axis=(1, 3))
            
        elif anti_alias_factor==2:
            # create anti-aliased aperture mask
            # double resolution axis
            N = self.res_tot
            delta = (ax[1]-ax[0])/4
            ax2 = np.empty(N * 2)
            ax2[0::2] = ax - delta
            ax2[1::2] = ax + delta           
            # Double resolution coordinate grid
            x2, y2 = np.meshgrid(ax2, ax2)
            if consider_pos_offset:
                x2 -= x_off
                y2 -= y_off
            
            # Double resolution radial coordinate grid
            R2 = np.sqrt(x2**2 + y2**2)  
            lens_pupil2 = (R2 <= lens_R).astype(float)     
            if black_value != 0:
                lens_pupil2 = np.where(lens_pupil2 == 0, black_value, lens_pupil2)
            N = self.res_tot
            # now downscale for anti-aliasing
            lens_pupil = lens_pupil2.reshape(N, 2, N, 2).mean(axis=(1, 3)) 
               
        else:               
            # create simple aperture mask
            x, y = np.meshgrid(ax, ax)
            if consider_pos_offset:
                x -= x_off
                y -= y_off
            
            R = np.sqrt(x**2 + y**2)  # Radial coordinate grid
            lens_pupil = (R <= lens_R).astype(float) 
            if black_value != 0:
                lens_pupil = np.where(lens_pupil == 0, black_value, lens_pupil)
            
        return lens_pupil
    
    def get_angle_from_nxy(self, tot:bool, nx, ny, Lambda, nr):
        """
        nx, ny... mode numbers
        Lambda .. wavelength
        nr ... real part of refractive index
        returns angle from nx, ny
        """
        if tot:
            N = self.res_tot
            L = self.axis_tot[-1] - self.axis_tot[0]  # Reduced length of axis in position-space
            L = L * N / (N - 1)  # Nominal length of axis in position-space
        else:
            N = self.res_fov
            L = self.axis_fov[-1] - self.axis_fov[0]  # Reduced length of axis in position-space
            L = L * N / (N - 1)  # Nominal length of axis in position-space
        kx = nx * 2 * np.pi / L
        ky = ny * 2 * np.pi / L
        k_tot = 2 * np.pi / Lambda
        kz = (nr * k_tot)**2 - kx**2 - ky**2
        if kz < 0:
            kz = 0
        else:
            kz = np.sqrt(kz)
        alpha = np.arccos(kz/k_tot)
        
        return(alpha)
    
    def empty_grid(self, fov_only: bool):
        """ 
        returns an empty grid (filled with zeros)
        fov_only  ... if True, then field-of-view dimensions only
        """
        if fov_only:
            return np.zeros((self.__res_fov, self.__res_fov), dtype=self.__cavity.dtype_c)
        else:
            return np.zeros((self.__res_tot, self.__res_tot), dtype=self.__cavity.dtype_c)
                
    def fourier_basis_func(self, nx: int, ny: int, tot: bool, k_space_out: bool):
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Creates the normalized (nx, ny) Fast-Fourier basis function  
        % either in position space or in k-space
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Input parameters:
        % nx   ... mode number in x-direction
        % ny   ... mode number in y-direction
        % tot  ... if True, then for the whole grid, else for field-of-view only 
        % k_space_out .. if true, output in k-space, otherwise in position space 
        %
        % Output:
        % normalized basis function
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        if tot:
            N = self.__res_tot
            ax = self.__axis_tot
        else:
            N = self.__res_fov
            ax = self.__axis_fov
            
        c = N // 2
        psi = np.zeros((N, N), dtype=self.__cavity.dtype_c)
        if c-nx>=0 and c-nx < N and c-ny>=0 and c-ny < N:    
            psi[c-ny, c-nx]=1 # create normalized basis function in f-space
        
        if not k_space_out:
            psi = ifft2_phys_spatial(psi, ax)
        
        return psi
    
    
    def get_sorted_mode_numbers(self, fov_only: bool, n_max:int, return_all_col):
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        returns a vector with all combinations of nx and ny up to n_max
        ordered by increasing angle of corresponding k-vectors with respect 
        to the z-axis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Input parameter:
        tot: if True, then for the whole grid, else for field-of-view only
        n_max: if > 0, then output is limited to sqrt(nx**2+ny**2)<=n_max
        return_all_col: if true, the the third nx**2+ny**2 is also returned
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Ouptupt:
        vector with all combinations of nx and ny up to n_max
        first column: nx-values, second column: ny-values, thirrd column: nx**2+ny**2
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        return self.__sorted_mode_numbers(not fov_only, n_max, return_all_col) 
        
    def __sorted_mode_numbers(self, tot: bool, n_max=-1, return_all_col = False):
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        returns a vector with all combinations of nx and ny up to n_max
        ordered by increasing angle of corresponding k-vectors with respect 
        to the z-axis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Input parameter:
        tot: if True, then for the whole grid, else for field-of-view only
        n_max: if > 0, then output is limited to sqrt(nx**2+ny**2)<=n_max
        return_all_col: if true, the the third nx**2+ny**2 is also returned
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Ouptupt:
        vector with all combinations of nx and ny up to n_max
        first column: nx-values, second column: ny-values)
        """
        if tot:
            N = self.__res_tot
        else:
            N = self.__res_fov
            
        nxy_max = N // 2  # maximum value for nx, ny
        odd = (N % 2 != 0)  # if True, then N is odd
        
        # Create result matrix and temp array based on the value of odd
        if odd:
            result = np.zeros(((1 + int(nxy_max) * 2)**2, 3), dtype=int)
            temp = np.arange(-int(nxy_max), int(nxy_max) + 1)
        else:
            result = np.zeros(((int(nxy_max) * 2)**2, 3), dtype=int)
            temp = np.arange(-int(nxy_max) + 1, int(nxy_max) + 1)  # Adjust for Python range
        
        # Fill first column with increasing nx-values
        result[:, 0] = np.repeat(temp, len(temp))
        # Fill second column with ny-values
        result[:, 1] = np.tile(temp, len(temp))
        # Fill third column with nx^2+ny^2
        result[:, 2] = result[:, 0]**2 + result[:, 1]**2
        # Sort by the third column, and then by the first and second column if needed        
        indices = np.lexsort((result[:, 1], result[:, 0], result[:, 2]))
        result = result[indices]
        
        if n_max>0:
            # filter nx**2+ny**2 <= n_max**2
            n_max_squared = n_max**2
            result = result[result[:, 2] <= n_max_squared]
            
        if return_all_col:
            # return all columns
            return result
        else:
            # return first and second column as result        
            return result[:, :2]
        
    
    def get_ax_plot_info(self):
        """ 
        Returns data required for plotting the axis:
        min value FOV, max value FOV,  min value tot, max value tot, unit
        """
        cm = 0.01
        mm = 0.001
        um = 0.000001
        nm = 0.000000001
        dx_fov = self.axis_fov[1]-self.axis_fov[0]
        dx_tot = self.axis_tot[1]-self.axis_tot[0]
        min_FOV = self.axis_fov[0] - dx_fov/2 
        max_FOV = self.axis_fov[-1] + dx_fov/2 
        #max_FOV = min_FOV + self.length_fov  
        min_tot = self.axis_tot[0]- dx_tot/2
        max_tot = self.axis_tot[-1] + dx_tot/2
        #max_tot = min_tot + self.length_tot
        if max_FOV < 100*nm:
            unit = "nm"
            f = 1000000000
        elif max_FOV < 100*um:
            unit = "um"
            f = 1000000
        elif max_FOV < 50*mm:
            unit = "mm"
            f = 1000
        elif max_FOV < 50*cm:
            unit = "cm"
            f = 100
        else:
            unit = "m"
            f = 1
    
        return min_FOV*f, max_FOV*f, min_tot*f, max_tot*f, unit
    
    @property
    def even_res(self):
        """" Returns True if resolution is even """
        return (self.__res_fov % 2 == 0)
    
    @property
    def odd_res(self):
        """" Returns True if resolution is even """
        return (self.__res_fov % 2 != 0)
    
    @property
    def mode_numbers_fov(self):
        """ 
        vector with all combinations of nx and ny up to n_max 
        for fov resolution 
        [[  0,   0],
         [ -1,   0],
         [  0,  -1],
         [  0,   1], ... ]
        """
        if self.__mode_numbers_fov is None:
            self.__mode_numbers_fov = self.__sorted_mode_numbers(False)
        return self.__mode_numbers_fov
    
    @property
    def mode_indices_fov(self):
        """ 
        vector with row and column indices for getting the mode values 
        out of an array in k-space 
        """
        if self.__mode_indices_fov is None:
            if self.even_res:
                C = self.__res_fov // 2
            else:
                C = (self.__res_fov - 1) // 2
            self.__mode_indices_fov = np.zeros_like(self.mode_numbers_fov)
            self.__mode_indices_fov[:,0] = C - self.mode_numbers_fov[:,1]
            self.__mode_indices_fov[:,1] = C - self.mode_numbers_fov[:,0]
        return self.__mode_indices_fov    
    
    @property
    def mode_numbers_tot(self):
        """ 
        vector with all combinations of nx and ny up to n_max 
        for total resolution 
        [[  0,   0],
         [ -1,   0],
         [  0,  -1],
         [  0,   1], ... ]
        """
        if self.__mode_numbers_tot is None:
            self.__mode_numbers_tot = self.__sorted_mode_numbers(True)
        return self.__mode_numbers_tot
    
    @property
    def mode_indices_tot(self):
        """ 
        vector row and column indices for getting the mode values 
        out of an array in k-space 
        """
        if self.__mode_indices_tot is None:
            if self.even_res:
                C = self.__res_tot // 2
            else:
                C = (self.__res_tot - 1) // 2
            self.__mode_indices_tot = np.zeros_like(self.mode_numbers_tot)
            self.__mode_indices_tot[:,0] = C - self.mode_numbers_tot[:,1]
            self.__mode_indices_tot[:,1] = C - self.mode_numbers_tot[:,0]
        return self.__mode_indices_tot    
    
    def get_center_index(self, fov_only: bool):
        """ returns the index corresponding to the zero position"""
        if fov_only:
            N = self.__res_fov
        else:
            N = self.__res_tot
            
        return N//2
    
    def get_row_col_idx_from_nx_ny(self, tot: bool, nx:int, ny:int):
        """ returns row and col index for total resolution k-space arry """
        if tot:
            N = self.__res_tot
        else:
            N = self.__res_fov
            
        c = N // 2        
        #if c-nx>=0 and c-nx < N and c-ny>=0 and c-ny < N:    
        return c-ny, c-nx
    
    @property
    def res_fov(self):
        """ returns field-of-view resolution per side in pixels """
        return self.__res_fov
    
    @property
    def res_tot(self):
        """ returns total resolution (including padding) in pixels """
        return self.__res_tot
    
    @property
    def length_fov(self):
        """ returns fov sidelength in m """
        return self.__length_fov
    
    @property
    def length_fov_mm(self):
        """ returns fov sidelength in mm """
        return self.__length_fov*1000
    
    @property
    def length_tot(self):
        """ returns total sidelength (including padding) in m """
        return self.__length_tot
    
    @property
    def length_tot_mm(self):
        """ returns total sidelength (including padding) in m """
        return self.__length_tot*1000
    
    @property
    def axis_fov(self):
        """ returns a numpy vector with the fov coordinates for each side """
        return self.__axis_fov
    
    @property
    def axis_tot(self):
        """ returns a numpy vector with the total coordinates for each side """
        return self.__axis_tot
    
    @property
    def k_axis_fov(self):
        """ returns a numpy vector with wave numbers matching
         the axis of the FFT-transformed FOV array """
        return self.__k_axis_fov
    
    @property
    def k_axis_tot(self):
        """ returns a numpy vector with wave numbers matching
         the axis of the FFT-transformed total array """
        return self.__k_axis_tot
    
    
    @property
    def n_axis_fov(self):
        """ Creates a vector with the same number of entries as the
        ax vector, but containing nx, ny mode numbers matching the 
        axis of the FFT-transformed position-space-array """
        return self.__n_axis_fov
    
    @property
    def n_axis_tot(self):
        """ Creates a vector with the same number of entries as the
        ax vector, but containing nx, ny mode numbers matching the 
        axis of the FFT-transformed position-space-array """
        return self.__n_axis_tot
    
    def set_res(self, res_fov: int, res_tot: int, length_fov: float):
        """ set field-of-view and total xy resolution in pixels """
        self.__mode_numbers_fov = None
        self.__mode_numbers_tot = None
        self.__mode_indices_fov = None
        self.__mode_indices_tot = None
        
        if res_fov<10:
            res_fov = 10    
            print ("Warning: minimum resolution is 10x10 pixels.")
            print ("Seting res_fov = 10.")                            
        
        if res_tot < res_fov:
            res_tot = res_fov
            print ("Warning: res_tot must not be smaller than res_fov.")            
            print ("Seting res_tot =", res_tot)            
        
        fov_even = ((res_fov % 2 )==0)
        tot_even = ((res_tot % 2 )==0)
        if fov_even and not tot_even:
            res_tot = res_tot + 1
            print ("Warning: if res_fov is even, res_tot must also be even.") 
            print ("Seting res_tot =", res_tot)
        if not fov_even and tot_even:
            res_tot = res_tot + 1
            print ("Warning: if res_fov is odd, res_tot must also be odd.") 
            print ("Seting res_tot =", res_tot)

        if length_fov < 0.000001:
            length_fov = 0.000001
            print ("Warning: length_fov must be >= 1um. Seting length_fov = 1um")
        
        self.__res_fov = res_fov
        self.__res_tot = res_tot
        self.__length_fov = length_fov
        self.__length_tot = length_fov * self.__res_tot / self.__res_fov   
        self.__create_axes()
        
        self.__set_pov_offset()
        
    def set_opt_res_based_on_sidelength(self, length_fov: float, factor: float, prop_dist: float, even: bool):
        """
        Calculates the optimal grid parameters for propagation distance z 
        length_fov is the required side length of the FOV input grid
        To allow for diffraction waves to be detected after propagation,
        the input grid is embedded in a larger grid, which sides are larger
        by "factor".  With this function, the desired side length of the 
        embedding grid (given by length_fov * factor) determines the resolution
        
        Input parameters:
        length_fov ... desired side-length of (smaller) "guarded" grid
        factor ....... factor by which the embedding grid's sides are larger
        prop_dist ... propagation distance (single propagation)
        even ......... If true: even number of steps, if false: odd number
        """
        self.__mode_numbers_fov = None
        self.__mode_numbers_tot = None
        self.__mode_indices_fov = None
        self.__mode_indices_tot = None
        
        if factor < 1:
            factor = 1
            print ("Warning: factor must be >= 1. Seting factor = 1") 
        
        if length_fov < 0.000001:
            length_fov = 0.000001
            print ("Warning: length_fov must be >= 1um. Seting length_fov = 1um") 
        
        if prop_dist < 0.000001:
            prop_dist = 0.000001
            print ("Warning: prop_dist must be >= 1um. Seting prop_dist = 1um") 
               
        lambda_times_dist = self.__cavity.Lambda*prop_dist 
        # estimation for the side-length of the larger grid
        length_tot = length_fov * factor
        # best-matching integer optimal resolution for length_tot
        self.__res_tot = round((length_tot*length_tot)/lambda_times_dist)
        
        if even:
            # even resolution demanded
            if not ((self.__res_tot % 2 )==0):
                # self.__res_tot is not even, make it even
                self.__res_tot = self.__res_tot + 1
        else:
            # odd resolution demanded
            if ((self.__res_tot % 2 )==0):
                # self.__res_tot is even, make it odd
                self.__res_tot = self.__res_tot + 1
                        
        # back-calculate exact side-length of the larger grid matching its resolution
        self.__length_tot = math.sqrt(self.__res_tot*lambda_times_dist);
                
        self.__res_fov = round(self.__res_tot/factor) # resolution fov grid
        if even:
            # even resolution demanded
            if not ((self.__res_fov % 2 )==0):
                # self.__res_fov is not even, make it even
                self.__res_fov = self.__res_fov + 1
        else:
            # odd resolution demanded
            if ((self.__res_fov % 2 )==0):
                # self.__res_fov is even, make it odd
                self.__res_fov = self.__res_fov + 1
    
        # calculate exact side-length of smaller FOV grid
        self.__length_fov = self.__length_tot * self.__res_fov / self.__res_tot
        self.__create_axes()
        self.__set_pov_offset()
    

    def set_opt_res_tot_based_on_res_fov(self, length_fov: float, res_fov: int, prop_dist: float):
        """
        Calculates the optimal total resolution of embedding grid
        based on the desired FOV side-length length_fov and FOV resolution
        fov_res and the propagation distance prop_dist
        """
        self.__mode_numbers_fov = None
        self.__mode_numbers_tot = None
        self.__mode_indices_fov = None
        self.__mode_indices_tot = None
        
        lambda_times_dist = self.__cavity.Lambda*prop_dist
        
        if length_fov < 0.000001:
            length_fov = 0.000001
            print ("Warning: sidelength_fov must be >= 1um. Seting sidelength_fov = 1um") 
            
        if res_fov<10:
            res_fov = 10    
            print ("Warning: minimum resolution is 10x10 pixels.")
            print ("Seting res_fov = 10.")      
                    
        self.__res_fov = res_fov        
        self.__res_tot = round(pow(res_fov/length_fov,2)*lambda_times_dist)
        res_fov_even = ((self.__res_fov % 2 )==0)
        res_tot_even = ((self.__res_tot % 2 )==0)  
        
        if res_fov_even != res_tot_even:
            self.__res_tot = self.__res_tot + 1
        
        self.__length_fov = length_fov
        self.__length_tot = length_fov * self.__res_tot / self.__res_fov        
        self.__create_axes()
        self.__set_pov_offset()
        
    def arr_to_vec(self, X, k_space_in, column_vec = True):        
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Converts position-space array or k-space-array 
        % into a Fourier-coefficient-vector with the coefficients ordered 
        % according to the mode_numbers vector
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Input:
        % X ............ k-space or position-space input array
        % k_space_in ... if true:  X is a k-space-array,
        %                if false: X is a position-space-array. 
        % column_vec ... optional. If true, a column vector is returned
        %
        % Output:
        % vector with Fourier coefficients, ordered according to 
        % mode_numbers vector
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        N = X.shape[0]
       
        if N == self.__res_fov:
            ax = self.axis_fov
            mode_idx = self.mode_indices_fov
        else:
            ax = self.axis_tot
            mode_idx = self.mode_indices_tot
        
        if not k_space_in:
            # position-space-array as input -> 
            # convert to spatial-frequency-space-array first
            X = fft2_phys_spatial(X, ax)
        
        # convert nx, ny values from mode_numbers vector into 
        # linear index vector for the input array
        if column_vec:
            return X[mode_idx[:,0], mode_idx[:,1]].reshape(-1, 1)
        else:
            return X[mode_idx[:,0], mode_idx[:,1]]
 
    def vec_to_arr(self, X, k_space_out):        
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Converts the Fourier-coefficient vector X into a
        % position-space-array or a spatial-frequency-space-array. 
        % The entries in the input vector FFT_vec are associated to the 
        % according nx/ny-modes by means of the mode_numbers vector
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Input:
        % X ............ Fourier-coefficient vector
        % k_space_out .. if true:  output is a k-space-array,
        %                if false: X is a position-space-array. 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        NN = X.shape[0]
        if NN == self.__res_fov**2:
            N = self.__res_fov
            ax = self.axis_fov
            mode_idx = self.mode_indices_fov
        else:
            N = self.__res_tot
            ax = self.axis_tot
            mode_idx = self.mode_indices_tot
       
        Y = np.zeros((N, N), dtype=self.__cavity.dtype_c) 
        Y[mode_idx[:, 0], mode_idx[:, 1]] = X.flatten()
       
        if not k_space_out:
            # output in position-space requested -> inverse FFT
            Y = ifft2_phys_spatial(Y, ax)
       
        return Y
       
    def limit_mode_numbers(self, X, mode_limit, k_space_in, k_space_out):
        """ 
        limits the number of FFT modes in grid X to 
        sqrt(nx**2 + ny**2)<=mode_limit
        """
        N = X.shape[0]
        if N == self.__res_fov:
            ax = self.n_axis_fov
        else:
            ax = self.n_axis_tot
            
        if not k_space_in:
            # if input in position-space, convert to k-space
            X = self.fft2_phys_spatial(X)
        
        # Limit mode numbers to a cirular area on the FFT grid
        nx, ny = np.meshgrid(ax, ax)
        R = np.sqrt(nx**2 + ny**2)
        aperture = (R <= mode_limit)
        X = X * aperture
        
        if not k_space_out:
            # output required in position-space, convert to position-space
            X = self.ifft2_phys_spatial(X)
        
        return X

    def fft2_phys_spatial(self, X):
        """ FFT transformation from pos-space to k-space """
        N = X.shape[0]
        if N == self.__res_fov:
            ax = self.n_axis_fov
        else:
            ax = self.n_axis_tot        
        
        return fft2_phys_spatial(X, ax)
    
    def ifft2_phys_spatial(self, X):
        """ inverse FFT transformation from k-space to pos-space """
        N = X.shape[0]
        if N == self.__res_fov:
            ax = self.n_axis_fov
        else:
            ax = self.n_axis_tot        
        
        return ifft2_phys_spatial(X, ax)
    
    def _convert_TR_mat_tot_to_fov_helper(self, i):
        nx = self.mode_numbers_fov[i,0]
        ny = self.mode_numbers_fov[i,1]
        X = self.fourier_basis_func(nx, ny, False, False) # small base function
        X = self.embed_image(X, False, False) # embedded in tot resolution
        X = self.arr_to_vec(X, False) # convert to vector
        X = self._TR_temp @ X # apply T or R matrix
        X = self.vec_to_arr(X, False) # back-convert to array (pos-space)
        X = self.extract_FOV(X, False, False) # extract FOV
        self.__cavity.progress.tic()
        return self.arr_to_vec(X, False, False)
        
        
    
    def convert_TR_mat_tot_to_fov(self, X, tics = 0):
        """ 
        input: X ... transmission or reflection matrix matching tot resolution 
        output: Transmission or reflection matrix matching fov resolution
        ("downscaling TR matrix")
        """
        if tics > 0:
            self.__cavity.progress.tic_reset(tics)
            
        if not is_matrix(X):
            return X
        
        NN = self.res_fov**2
        if tics == 0:
            self.__cavity.progress.tic_reset(NN)
    
            
        result =  np.empty((NN,NN), dtype=self.__cavity.dtype_c)
        for i in range(NN):
            nx = self.mode_numbers_fov[i,0]
            ny = self.mode_numbers_fov[i,1]
            Y = self.fourier_basis_func(nx, ny, False, False) # small base function
            Y = self.embed_image(Y, False, False) # embedded in tot resolution
            Y = self.arr_to_vec(Y, False) # convert to vector
            Y = X @ Y # apply T or R matrix
            Y = self.vec_to_arr(Y, False) # back-convert to array (pos-space)
            Y = self.extract_FOV(Y, False, False) # extract FOV
            result[:,i] = self.arr_to_vec(Y, False, False)
            self.__cavity.progress.tic()
        
        return result
    
    def convert_TR_bmat2_tot_to_fov(self, X):
        """ 
        input: X ... transmission or reflection 2x2 block matrix in tot resolution 
        output: Transmission or reflection 2x2 block matrix matching fov resolution
        ("downscaling TR matrix")
        """
        tics = 0
        if is_matrix(X[0][0]):
            tics += self.res_fov**2
        if is_matrix(X[0][1]):
            tics += self.res_fov**2
        if is_matrix(X[1][0]):
            tics += self.res_fov**2
        if is_matrix(X[1][1]):
            tics += self.res_fov**2
        A = self.convert_TR_mat_tot_to_fov(X[0][0], tics)
        B = self.convert_TR_mat_tot_to_fov(X[0][1], -1)
        C = self.convert_TR_mat_tot_to_fov(X[1][0], -1)
        D = self.convert_TR_mat_tot_to_fov(X[1][1], -1)
        return [[A, B], [C, D]]
    
        
    def __create_axes(self):
        """ Create the axes for the small (FOV) and large (total) grid """
        
        # FOV array physical axis
        N = self.__res_fov
        dx = self.__length_fov/N         
        self.__axis_fov = ((np.arange(1.0, N+1.0) - N/2.0) * dx - dx
                           ).astype(self.__cavity.dtype_r)
        
        # FOV FFT array, n-vector and k-vector axis
        if (N%2 == 0):
            n = np.arange(N//2, -N//2, -1)
        else:
            n = np.arange((N-1)//2, -(N+1)//2, -1)
        self.__n_axis_fov = n
        
        L = self.__axis_fov[-1] - self.__axis_fov[0]  # Reduced length of axis in position-space
        L = L * N / (N - 1)  # Nominal length of axis in position-space
        self.__k_axis_fov = (n * 2 * np.pi / L).astype(self.__cavity.dtype_r)
        
        
        # Total array physical axis
        N = self.__res_tot
        dx = self.__length_tot/N         
        self.__axis_tot = ((np.arange(1.0, N+1.0) - N/2.0) * dx - dx
                           ).astype(self.__cavity.dtype_r)
        
        # FOV Total array, n-vector and k-vector axis
        if (N%2 == 0):
            n = np.arange(N//2, -N//2, -1)
        else:
            n = np.arange((N-1)//2, -(N+1)//2, -1)
        self.__n_axis_tot = n    
            
        L = self.__axis_tot[-1] - self.__axis_tot[0]  # Reduced length of axis in position-space        
        L = L * N / (N - 1)  # Nominal length of axis in position-space
        self.__k_axis_tot = (n * 2 * np.pi / L).astype(self.__cavity.dtype_r)
        
        
    def __set_pov_offset(self):
        # default: center position
        self.__fov_offset_x = (self.res_tot - self.res_fov) // 2
        self.__fov_offset_y = (self.res_tot - self.res_fov) // 2
        
        x_ref = self.__fov_offset_x
        y_ref = self.__fov_offset_y
        
        if self.__fov_pos == 1: 
            # top position
            if self.res_fov > self.res_tot //2:
                self.__fov_offset_y = self.res_tot - self.res_fov
            else:
                self.__fov_offset_y = \
                    (self.res_tot+1)//2 + (self.res_tot//2-self.res_fov)//2
            
        elif self.__fov_pos == 2: 
            #bottom position
            if self.res_fov > self.res_tot //2:
                self.__fov_offset_y = 0
            else:
                self.__fov_offset_y = \
                    (self.res_tot//2-self.res_fov)//self.res_fov
                if self.res_tot % 2 == 0:
                    self.__fov_offset_y += 1
                     
        elif self.__fov_pos == 3:
            #left position
            if self.res_fov > self.res_tot //2:
                self.__fov_offset_x = 0
            else:
                self.__fov_offset_x = \
                    (self.res_tot//2-self.res_fov)//self.res_fov
                if self.res_tot % 2 == 0:
                    self.__fov_offset_x += 1
        
        elif self.__fov_pos == 4:
            # right position
            if self.res_fov > self.res_tot //2:
                self.__fov_offset_x = self.res_tot - self.res_fov
            else:
                self.__fov_offset_x = \
                    (self.res_tot+1)//2 + (self.res_tot//2-self.res_fov)//2
    
    
        self.__pos_offset_x = self.__fov_offset_x - x_ref
        self.__pos_offset_y = self.__fov_offset_y - y_ref
        
        
    def extract_FOV(self, E_in, k_space_in: bool, k_space_out: bool):
        """ 
        extracts the field-of-view from the larger E_in image        
        """
        if k_space_in:
            # input in k-space: convert to position space
             E_in = ifft2_phys_spatial(E_in, self.axis_tot)
         
        #offset = (self.res_tot - self.res_fov) // 2
        E_out = E_in[self.__fov_offset_y:self.__fov_offset_y+self.res_fov,
                     self.__fov_offset_x:self.__fov_offset_x+self.res_fov]
        
        if k_space_out:
            # output in k-space: convert to k-space
             E_out = fft2_phys_spatial(E_out, self.axis_fov)

        return E_out
    
    def embed_image(self, E_in, k_space_in: bool, k_space_out: bool):
        """ 
        embeds the field-of-view image into the larger image with res_tot
        """
        if k_space_in:
            # input in k-space: convert to position space
             E_in = ifft2_phys_spatial(E_in, self.axis_fov)
             
        #offset = (self.res_tot - self.res_fov) // 2
        E_out = np.zeros((self.res_tot, self.res_tot), dtype=self.__cavity.dtype_c)
        E_out[self.__fov_offset_y:self.__fov_offset_y+self.res_fov,
              self.__fov_offset_x:self.__fov_offset_x+self.res_fov] = E_in
        
        if k_space_out:
            # output in k-space: convert to k-space
             E_out = fft2_phys_spatial(E_out, self.axis_tot)

        return E_out
    
    def get_res_arr(self, X):
        """ 
        Determines the resolution matching the light field array X
        output: 0... unkonw res, 1... FOV res, 2 ... tot res
        """
        if isinstance(X, np.ndarray):
            N = X.shape[0]
            if self.res_tot == N:
                return 2
            elif self.res_fov == N:
                return 1
            else:
                return 0
        else:
            #scalar: assume full resolution
            return 2
    
    def is_fov_res(self, X):
        """ Returns True, if X is FOV resoultion """
        return (self.get_res_arr(X) == 1)
    
    def get_res_TR(self, TR):
        """ 
        Determines the resolution matching the transm. or refl. Matrix TR
        output: 0... unkonw res, 1... FOV res, 2 ... tot res
        """
        if not is_matrix(TR):
            return 2
        else:
            N = TR.shape[0]
            if self.res_tot**2 == N:
                return 2
            elif self.res_fov**2 == N:
                return 1
            else:
                return 0       
    
    def get_res_vec(self, vec):
        """ 
        Determines the resolution matching the mode vector vec
        output: 0... unkonw res, 1... FOV res, 2 ... tot res
        """
        return self.get_res_TR(vec)
    
    def convert(self, E_in, k_space_in: bool, k_space_out: bool, fov_out: bool):
        """ Converts E_in from and to k_space and converts size if neccessary"""
                
        if E_in is None:
            if fov_out:
                return np.zeros((self.res_fov, self.res_fov), dtype=self.__cavity.dtype_c)
            else:
                return np.zeros((self.res_tot, self.res_tot), dtype=self.__cavity.dtype_c)    
        
        if not isinstance(E_in, np.ndarray):            
            fov_in = False
            E_in = E_in * np.ones((self.res_tot, self.res_tot), dtype=self.__cavity.dtype_c)
        else:            
            fov_in = (E_in.shape[0]==self.res_fov)
            
        if self.res_fov == self.res_tot:
            fov_in = fov_out
            
        if fov_in:
            # small in
            if fov_out:
                # small in, small out
                if k_space_in:
                     # small in, small out, k_space_in
                     if k_space_out:
                         # small in, small out, k_space_in, k_space_out
                         return E_in
                     else:
                         # small in, small out, k_space_in, pos_space_out
                         return ifft2_phys_spatial(E_in, self.axis_fov)
                else:
                     # small in, small out, pos_space_in
                     if k_space_out:
                         # small in, small out, pos_space_in, k_space_out
                         return fft2_phys_spatial(E_in, self.axis_fov)
                     else:
                         # small in, small out, pos_space_in, pos_space_out
                         return E_in
            else:
                # small in, large out
                E_out = self.embed_image(E_in, k_space_in, k_space_out)
                return E_out
            
        else:
            # large in
            if fov_out:
                #large in, small out
                return self.extract_FOV(E_in, k_space_in, k_space_out)
            else:
                #large in, large out                
                if k_space_in:
                     # large in, large out, k_space_in
                     if k_space_out:
                         # large in, large out, k_space_in, k_space_out
                         return E_in
                     else:
                         # large in, large out, k_space_in, pos_space_out
                         return ifft2_phys_spatial(E_in, self.axis_tot)
                else:
                     # large in, large out, pos_space_in
                     if k_space_out:
                         # small in, large out, pos_space_in, k_space_out
                         return fft2_phys_spatial(E_in, self.axis_tot)
                     else:
                         # large in, large out, pos_space_in, pos_space_out
                         return E_in
        

###############################################################################
# clsLightField
# represents any input light field
###############################################################################    
class clsLightField(ABC):
    """ Superclass for any light field """
    def __init__(self, grid):
        self.__grid = grid  
        self._field = None
        self.__k_space = False
        self.__fov_only = False
        self.__name = ""
    
    def stretch_x(self, factor, center_pos = 0):
        """
        stretches the field in x direction around the 
        horizontal position center_pos by a factor `factor`.  

        :param factor:    factor by which to stretch (if <0 then compress)
        :center_pos:      The reference x-position about which we stretch
        """
        if factor == 1:
            # nothing to do
            return
        
        if self._field is None:
            # no image to stretch
            return
        
        field = self.get_field_tot(False)
        self.set_field(self.grid.stretch_x(field, center_pos, factor), False)        
        
    
    def stretch_y(self, factor, center_pos = 0):
        """
        stretches the field in y direction around the 
        vertical position center_pos by a factor `factor`.  

        :param factor:      factor by which to stretch (if <0 then compress)
        :param center_pos:  The reference y-position about which we stretch
        """
        if factor == 1:
            # nothing to do
            return
        
        if self._field is None:
            # no image to stretch
            return
        
        field = self.get_field_tot(False) 
        self.set_field(self.grid.stretch_y(field, center_pos, factor), False)   
    
    def shift(self, x_shift, y_shift):
        """
        Shifts the image in x and/or y direction
        returns xshift and yshift pixels
        """
        
        x_shft_px_signed = self.__grid.dist_to_pixels(x_shift)
        y_shft_px_signed = self.__grid.dist_to_pixels(y_shift)
        if self._field is None:
            # no image to shift
            return x_shft_px_signed, y_shft_px_signed 
        
        if x_shft_px_signed == 0 and y_shft_px_signed == 0:
            # nothing to do
            return x_shft_px_signed, y_shft_px_signed 
        
        pos_space = (not self.__k_space)   # current img in position space?
        full_size = (not self.__fov_only)  # current image full size?
        if not (pos_space and full_size):
            # make current image full_size and position space
            self.set_field(self.get_field_tot(False), False)
        
        x_shift_pixels = abs(x_shft_px_signed) 
        y_shift_pixels = abs(y_shft_px_signed)
        x_shift_right = (x_shift >= 0)
        y_shift_up = (y_shift >= 0)
                
        if x_shift_pixels >= self.__grid.res_tot:
            # image gets shifted so far that empty image remains
            self._field = None
            return x_shft_px_signed, y_shft_px_signed
        
        if y_shift_pixels >= self.__grid.res_tot:
            # image gets shifted so far that empty image remains
            self._field = None
            return x_shft_px_signed, y_shft_px_signed
        
        if y_shift_pixels>0:
            if y_shift_up:
                self._field = np.vstack(
                    (np.zeros((y_shift_pixels, self._field.shape[1])), 
                     self._field[:-y_shift_pixels]))
            else:
                self._field = np.vstack(
                    (self._field[y_shift_pixels:], 
                     np.zeros((y_shift_pixels, self._field.shape[1]))))

        if x_shift_pixels>0:
            if x_shift_right:
                self._field = np.hstack(
                    (np.zeros((self._field.shape[0], x_shift_pixels)), 
                     self._field[:, :-x_shift_pixels]))
                
            else:
                self._field = np.hstack(
                    (self._field[:, x_shift_pixels:], 
                     np.zeros((self._field.shape[0], x_shift_pixels))))
                
        return x_shft_px_signed, y_shft_px_signed

    def clone(self):
        """ returns a clone of this clsLightField object"""
        new_field = clsLightField(self.grid)
        new_field.name = self.__name
        if not self._field is None:
            new_field.set_field(self._field.copy(), self.__k_space)
        return new_field
    
    def add(self, other_field):
        """adds another field to this field """
        if other_field.empty:
            # the other field is empty: don't change this field
            return 
        
        if self._field is None:
            # this field is empty: just take the other field
            self.__k_space = other_field.k_space
            #print("self.__kspace", self.__kspace)
            self.__fov_only = other_field.fov_only
            #print("self.__fov_only", self.__fov_only)
            if other_field.fov_only:
                self._field = other_field.get_field_fov(self.__k_space).copy()
            else:
                self._field = other_field.get_field_tot(self.__k_space).copy()
                #print("self.__field", self.__field)
            
        else:
            # neither this nor the other field is empty
            if self.__fov_only and other_field.fov_only:
                # both fields are in FOV resoultion
                self._field += other_field.get_field_fov(self.__k_space)
            
            elif (not self.__fov_only) and (not other_field.fov_only):
                # both fields are in total resolution
                self._field += other_field.get_field_tot(self.__k_space)
                
            else:
                # one of the fields is FOV, the other total resolution
                if self.__fov_only:
                    # if this field is FOV -> upgrade to total resolution
                    self._field = self.get_field_tot(self.k_space)
                    self.__fov_only = False
                # now add the other field
                self._field += other_field.get_field_tot(self.__k_space)
    
    
    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name):
        self.__name = name
    
    @property 
    def empty(self):
        return (self._field is None) 
    
    @property
    def fov_only(self):
        return self.__fov_only
    
    @property
    def grid(self):
        return self.__grid
    
    @property
    def k_space(self):
        return self.__k_space    
    
    def get_field_tot(self, k_space_out: bool, process = 0):    
        """ returns the field as numpy array in total resolution 
        process: 0 ... complex, 1 .... real, 2 ... imaginary, 
        3 .... abs, 4 ... phase, 5 ... abs^2 """
        E_out = self.__grid.convert(self._field, self.__k_space, k_space_out, False)
        return self.__process_field(E_out, process)
           
    def get_field_fov(self, k_space_out: bool, process = 0):    
        """ returns the field numpy array in field of view resolution """
        E_out =  self.__grid.convert(self._field, self.__k_space, k_space_out, True)  
        return self.__process_field(E_out, process)
       
    def get_field_tot_vec(self, column_vec = True):
        """ returns the field in total resolution as mode-vector """
        return self.grid.arr_to_vec(self.get_field_tot(True), True, column_vec)
         
    def get_field_fov_vec(self, column_vec = True):
        """ returns the field in FOV resolution as mode-vector """
        return self.grid.arr_to_vec(self.get_field_fov(True), True, column_vec)    

    def set_field(self, field, k_space: bool):
        """ sets the field in FOV or total resolution (field: np array) """
        res = self.grid.get_res_arr(field)
        if res == 0:
            # unkonwn resolution
            print ("Error! Unknown resolution!")
            return
        
        self._field = field
        self.__k_space = k_space   
        self.__fov_only = (res == 1)        
   
    def set_field_vec(self, vec):
        """ sets the field based on the mode vector vec """
        self._field = None
        self.__k_space = False 
        self.__fov_only = False
        if vec is None:
            return
        if not is_matrix(vec):
            if vec == 0:
                return
        
        res = self.grid.get_res_vec(vec)
        if res == 0:
            # unkonwn resolution
            print ("Error! Unknown resolution!")
            return

        self._field = self.grid.vec_to_arr(vec, True)
        self.__k_space = True   
        self.__fov_only = (res == 1)
        
    def __process_field(self, E_in, process):
        """ returns the field in total resolution 
        process: 0 ... complex, 1 .... real, 2 ... imaginary, 
        3 .... abs, 4 ... phase, 5 ... abs^2 """
        if process == 1:
            return E_in.real
        elif process == 2:
            return E_in.imag
        elif process == 3:
            return abs(E_in)
        elif process == 4:
            return np.angle(E_in)
        elif process == 5:
            return abs(E_in)**2
        else:
            return E_in    
        
    def apply_TR_mat(self, TR):
        """ 
        applies a transmission or reflection matrix and returns the
        resulting light field
        """
        res = self.grid.get_res_TR(TR)
        if res==0:            
            print ("Error. TR matrix size does match neither full nor FOV resolution")
            return None
        
        new_field = clsLightField(self.grid)
        if res==1:
            # FOV resolution
            new_field.set_field_vec(mat_mul(TR, self.get_field_fov_vec()))
        else:
            # total resolution
            new_field.set_field_vec(mat_mul(TR, self.get_field_tot_vec()))
                
        return new_field 
        
    
    def intensity_integral_fov(self, aperture = 0):
        """ 
        returns the intesity integral over the field of view
        if aperture is provided than only over the area not coevred by aperture
        """
        if self._field is None:
            return 0        
        field_i = self.get_field_fov(False, 5)
        ax = self.grid.axis_fov
        if aperture>0:
            x, y = np.meshgrid(ax, ax)
            #if self.grid.fov_pos>0:
            #    pix_width = ax[1]-ax[0]
            #    x_off = self.grid.pos_offset_x * pix_width
            #    y_off = self.grid.pos_offset_y * pix_width
            #    x -= x_off
            #    y -= y_off
            R = np.sqrt(x**2+y**2);
            ap_R = aperture / 2
            ap_mask = (R <= ap_R).astype(int)  
            field_i *= ap_mask
        
        
        return np.trapz(np.trapz(field_i, axis=1, x=ax), axis=0, x=ax)
    
    def intensity_integral_tot(self, aperture = 0):
        """ 
        returns the intesity integral over the field of view
        if aperture is provided than only over the area not coevred by aperture
        """
        if self._field is None:
            return 0
        field_i = self.get_field_tot(False, 5)
        ax = self.grid.axis_tot
        
        if aperture>0:
            x, y = np.meshgrid(ax, ax)
            if self.grid.fov_pos>0:
                pix_width = ax[1]-ax[0]
                x_off = self.grid.pos_offset_x * pix_width
                y_off = self.grid.pos_offset_y * pix_width
                x -= x_off
                y -= y_off
            R = np.sqrt(x**2+y**2);
            ap_R = aperture / 2
            ap_mask = (R <= ap_R).astype(int)  
            field_i *= ap_mask
        #self.set_field(field_i, False)
        return np.trapz(np.trapz(field_i, axis=1, x=ax), axis=0, x=ax)
    
    @property 
    def intensity_sum_fov(self):
        """ returns the sum over all intensity pixels """
        if self._field is None:
            return 0
        return np.sum(self.get_field_fov(False, 5))
    
    @property 
    def intensity_sum_tot(self):
        """ returns the sum over all intensity pixels """
        if self._field is None:
            return 0
        return np.sum(self.get_field_tot(False, 5))
    
    def plot_field(self, what_to_plot, fov_only=True, save_path=None, 
                   c_map = 'hot', vmax_limit=None, norm = None, vmax = None):
        """what_to_plot ... 1 .... real, 2 ... imaginary, 
           3 .... abs, 4 ... phase, 5 ... abs^2 (intensity),  """
           
        # Note: The first (i.e. zeroth) line of the field array in position 
        # space corresponds to the bottom line to be plotted
                
        if what_to_plot<1:
            what_to_plot = 1
        elif what_to_plot>5:
            what_to_plot = 5
            
        if what_to_plot == 1:
            bar_txt = "amplitude (real part)"
        elif what_to_plot == 2:
            bar_txt = "amplitude (imaginary part)"
        elif what_to_plot == 3:
            bar_txt = "amplitude (abs. value)"
        elif what_to_plot == 4:
            bar_txt = "phase"
        else:
            bar_txt = "intensity"
            
        if c_map == "custom":
            rgb_table = pd.read_csv('cmap_custom.csv', header=None)            
            rgb_values = rgb_table.values   # Normalize RGB values 
            c_map = ListedColormap(rgb_values)
            #c_map = LinearSegmentedColormap.from_list('cmap_custom', 
            #        [(0, 0, 0), (0.19039, 0.9468, 0.601279) ])
            # (0.1614331, 0.731442, 0.92679)
            
        min_FOV, max_FOV, min_tot, max_tot, unit = self.grid.get_ax_plot_info()
        fig, pax = plt.subplots(dpi=150)
        
        if fov_only:
            field = self.get_field_fov(False, what_to_plot)
        else:
            field = self.get_field_tot(False, what_to_plot)
        
        if not vmax is None:
            vmax_limit = vmax
        elif not vmax_limit is None:           
            if np.max(field) > vmax_limit:
                vmax_limit = None
        
        
        if fov_only:            
            ax_lo = min_FOV
            ax_hi = max_FOV
        else:
            ax_lo = min_tot
            ax_hi = max_tot
        
        if norm is None:
            norm_obj = None
        else:
            if isinstance(c_map, str):
                c_map = plt.get_cmap(c_map)
            norm_obj = Normalize(vmin=0, vmax = norm)
            colors = c_map(np.linspace(0, norm, 256))
            c_map = LinearSegmentedColormap.from_list('half_hot', colors)
        
        if vmax_limit is None:            
            im = pax.imshow(field, 
                            extent=[ax_lo, ax_hi, ax_lo, ax_hi],
                            cmap=c_map, origin='lower',
                            norm = norm_obj)            
        else:
            im = pax.imshow(field, 
                            extent=[ax_lo, ax_hi, ax_lo, ax_hi],
                            cmap=c_map, origin='lower', 
                            vmax=vmax_limit,
                            norm = norm_obj)
            
        # Use AutoLocator for automatic reasonable ticks
        pax.xaxis.set_major_locator(AutoLocator())
        pax.yaxis.set_major_locator(AutoLocator())
          
        # Add unit label to axes
        pax.set_xlabel(unit)
        pax.set_ylabel(unit)

        # Add title if provided
        if not self.name == "":
            plt.title(self.name)

        # Add colorbar
        cbar = plt.colorbar(im, label=bar_txt)
        formatter = ScalarFormatter(useOffset=False)  # Disable offset
        formatter.set_scientific(False)  # Disable scientific notation
        cbar.formatter = formatter
        cbar.update_ticks()  # Update ticks to reflect new formatter settings
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


###############################################################################
# clsPlaneWaveMixField
# represents a light field composed of a superposition of one or more
# plane waves
###############################################################################
class clsPlaneWaveMixField(clsLightField):
    def __init__(self, grid):
        super().__init__(grid)
        self.set_field(self.grid.empty_grid(True) , False)
    
    @property
    def fov_only(self):
        return super().fov_only
    
    @fov_only.setter
    def fov_only(self, fo):
        if super().fov_only != fo:
            if fo == True:
                self.set_field(self.get_field_fov(False), False)
            else:
                self.set_field(self.get_field_tot(False), False)
        
    def add_fourier_basis_func(self, nx, ny, amplitude):
        """ 
        adds a fourier basis function with index nx, ny and given amplidude
        """
        psi = self.grid.fourier_basis_func(nx, ny, not self.fov_only, True)
        self._field += amplitude * np.fft.fft2(np.fft.ifftshift(psi))
        
###############################################################################
# clsGaussBeam
# represents a gaussian Beam 
############################################################################### 
class clsGaussBeam(clsLightField):
    def __init__(self, grid):
        super().__init__(grid)
        self.__w0 = None # beam waist
          
    # Function to calculate beam radius at distance z
    def beam_radius(self, z):
        return self.__w0 * np.sqrt(1 + (z * self.grid.cavity.Lambda / (np.pi * self.__w0**2))**2)

    # Function to calculate radius of curvature of the wavefront at z
    def radius_of_curvature(self, z):
        if z == 0:
            return np.inf  # Return infinity if z is zero
        else:
            return z + (np.pi * self.__w0**2 / (self.grid.cavity.Lambda * z))

    # Function to calculate Gouy phase shift
    def gouy_phase(self, z):
        return np.arctan(z * self.grid.cavity.Lambda / (np.pi * self.__w0**2))
    
    def create_beam(self, waist, x_offset = 0, y_offset = 0, x_angle_deg=0, y_angle_deg=0, z=0):
        self.__w0 = waist
        ax = self.grid.axis_tot
        x, y = np.meshgrid(ax, ax)
        
        x_angle = x_angle_deg / 180 * math.pi
        y_angle = y_angle_deg / 180 * math.pi
        
        fx = math.cos(x_angle)
        fy = math.cos(y_angle)
            
        R_squared = ((x - x_offset)*fx)**2+((y - y_offset)*fy)**2;
        
        w_z = self.beam_radius(z) # w_0 for z=0
        R_z = self.radius_of_curvature(z) # inf for z = 0
        zeta_z = self.gouy_phase(z) # 0 for z=0
        
        # corresponds to np.exp(-r_squared / w_0**2) for z=0
        amplitude = (self.__w0 / w_z) * np.exp(-R_squared / w_z**2)
        amplitude *= math.sqrt(fx)
        amplitude *= math.sqrt(fy)
        k = 2 * np.pi / self.grid.cavity.Lambda  # Wave number
        phase = np.exp(-1j * (k * z + k * R_squared / (2 * R_z) - zeta_z)) # 1 for z = 0
        field = amplitude * phase
        
        # now apply tilts
        if x_angle_deg != 0 or y_angle_deg != 0:
            ax = self.grid.axis_tot
            x, y = np.meshgrid(ax, ax)
            k0 = 2 * np.pi / self.grid.cavity.Lambda
            
        if x_angle_deg != 0:
            alpha = x_angle_deg / 180 * math.pi
            phase_mask =  np.exp(1j * k0 * x * math.tan(alpha))
            field *= phase_mask
        
        if y_angle_deg != 0:
            alpha = y_angle_deg / 180 * math.pi
            phase_mask =  np.exp(1j * k0 * y * math.tan(alpha))
            field *= phase_mask
        
        self.set_field(field, False)
       
    
###############################################################################
# clsSpeckleField
# represents a random speckle field 
############################################################################### 
class clsSpeckleField(clsLightField):
    def __init__(self, grid):
        super().__init__(grid)
        self.__n_max = 0 # the largest value sqrt(nx**2+ny**2) can take on
        self.__aperture = 0
        self.__no_of_modes = 0
        
    def create_field(self, no_of_modes: int, aperture: float, seed: float, 
                     fov_equiv = False, consider_pos_offset=True):
        """
        Creates a random speckle field with random f-space FFT modes of equal
        amplitude, with a circular shape in position space
        no_of_modes ... number of random modes in k-space
        aperture ... diameter of cisruclar shape in position space
        seed ... if not -1, then random numbers are initialized with this seed
        fov_equiv ... if true, then no_of_modes is equavalent to this 
                      value in fov resoltion
                      
        """
        if seed >= 0:
            np.random.seed(seed)
                    
        if fov_equiv:            
            no_of_modes = int(no_of_modes * self.grid.factor**2)
        
        N = self.grid.res_tot
        n_max = np.sqrt(no_of_modes/np.pi)
        
        self.__no_of_modes = no_of_modes
        
        if aperture>0:
            ax = self.grid.axis_tot
            x, y = np.meshgrid(ax, ax)
            if consider_pos_offset:
                pix_width = ax[1]-ax[0]
                x_off = self.grid.pos_offset_x * pix_width
                y_off = self.grid.pos_offset_y * pix_width
                x -= x_off
                y -= y_off
            R = np.sqrt(x**2+y**2);
            ap_R = aperture / 2
            ap_mask = (R <= ap_R).astype(int)  
            
        field = self.__create_speckle_field(N, n_max, False)
        if aperture>0:
            field *= ap_mask
            field = self.grid.limit_mode_numbers(field, n_max, False, False)
            field /= np.max(abs(field))
            
        self.set_field(field, False)
        
        self.__n_max = n_max
        self.__aperture = aperture
        
    
        
        
    def create_field_eq_distr(self, no_of_modes: int, n_max, aperture: float, 
                              seed: float, Lambda=0, consider_pos_offset=True):
        """
        Creates a random speckle field
        no_of_modes ... no of random modes in k-space
        alpha_max ... maximum value sqrt(n_x**2 + n_y**2)        
        """
        if seed >= 0:
            np.random.seed(seed)
        
        if Lambda == 0:
            Lambda = self.grid.cavity.Lambda
            
        if n_max<=0:
            self.__n_max = abs(self.grid.n_axis_tot[-1])
        else:
            self.__n_max = n_max
        alpha_max = self.get_max_angle(Lambda, 1)
        self.__no_of_modes = no_of_modes
        
        self.__aperture = max(0,aperture)        
         
        mode_vec = self.grid.get_sorted_mode_numbers(False, self.__n_max, True)
        # add column with zeros to count how often an entry was entries
        mode_vec  = np.hstack((mode_vec, np.zeros(mode_vec.shape[0],int).reshape(-1, 1)))
        # count distinct number of n_sqared entries (third column)
        unique_n_squared, counts = np.unique(mode_vec[:,2], return_counts=True)
        
        # calculate corresponding distinct angles
        N = self.grid.res_tot
        L = self.grid.axis_tot[-1] - self.grid.axis_tot[0]  # Reduced length of axis in position-space
        L = L * N / (N - 1)  # Nominal length of axis in position-space                
        k_tot = 2 * np.pi / Lambda
        kxy = np.sqrt(unique_n_squared) * 2 * np.pi / L
        kz = k_tot**2 - kxy**2 
        kz = np.where(kz < 0, 0, kz)
        kz = np.sqrt(kz)
        unique_alpha = np.arccos(kz/k_tot)
        
        E_speckles_k = np.zeros((N,N), dtype=self.grid.cavity.dtype_c)
        p = self.grid.cavity.progress
        p.tic_reset(no_of_modes, False, "creating random speckle field")
        #print("alpha_max",alpha_max/np.pi*180)
        for i in range(no_of_modes):
            # random angle
            alpha_rnd = np.random.uniform(0, alpha_max)
            # find closest angle
            closest_val, idx1 = find_closest_value(unique_alpha, alpha_rnd)
            # closest n_sqared
            closest_n_squared = unique_n_squared[idx1]
            # find index vector with all values matching closest_n_squared
            f = mode_vec[mode_vec[:, 2] == closest_n_squared]
            # select randomly one of these
            idx_rnd = np.random.randint(0, f.shape[0])
            nx =  f[idx_rnd, 0]
            ny =  f[idx_rnd, 1]
            # flag mode as used
            matching_row = (mode_vec[:, 0] == nx) & (mode_vec[:, 1] == ny)
            mode_vec[matching_row, 3] += 1
            #if mode_vec[matching_row, 3]>1:
            #    print("nx, ny, count", nx, ny, mode_vec[matching_row, 3])
            
            # add a mode with random phase to E_speckles array
            row, col = self.grid.get_row_col_idx_from_nx_ny(True, nx, ny)
            E_speckles_k[row, col] = np.sqrt(mode_vec[matching_row, 3]) * \
                np.exp(2*np.pi*1j*np.random.rand())
            
            p.tic()
            
        E_speckles = self.grid.ifft2_phys_spatial(E_speckles_k)
        
        if aperture>0:
            ap_mask = self.grid.get_aperture_mask(aperture, 4, 
                                    consider_pos_offset=consider_pos_offset)
            E_speckles *= ap_mask
            E_speckles = self.grid.limit_mode_numbers(E_speckles, self.__n_max, 
                                                      False, False)
           
        E_speckles /=  np.max(abs(E_speckles))        
        self.set_field(E_speckles, False) 
        
    def get_aperture(self):
        return self.__aperture
    
    def get_n_max(self):
        return self.__n_max
    
    def get_req_emebed_factor(self, dist, Lambda, nr, alpha_deg = -1):
        """"
        returns how much larger the emebding grid must be to
        accomodate the maximum k angle at distance dist
        """        
        if alpha_deg == -1:
            alpha = self.get_max_angle(Lambda, nr)
        else:
            alpha = alpha_deg / 180 * math.pi
        s = dist * math.tan(alpha)
        
        if self.grid.fov_pos == 0:
            # no offset (speckle field centered in total grid)
            if self.__aperture <=0:
                length_fov = self.grid.length_fov
            else:
                length_fov = self.__aperture
        
        else:
            # offset (speckle field con one side of total grid)
            if self.__aperture <=0:
                length_fov = 2 * self.grid.length_fov
            else:
                length_fov = self.__aperture + self.grid.length_fov
        
        length_tot = length_fov + 2 * s
        return length_tot / self.grid.length_fov
    
    
    def get_max_angle(self, Lambda, nr):
        """ 
        returns the maximum k-vector angle in radians
        Lambda .. wavelength
        nr ... real part of refractive index
        """
        N = self.grid.res_tot
        L = self.grid.axis_tot[-1] - self.grid.axis_tot[0]  # Reduced length of axis in position-space
        L = L * N / (N - 1)  # Nominal length of axis in position-space
        kxy_max = self.__n_max * 2 * np.pi / L
        
        # Calculate k_tot
        k_tot = 2 * np.pi / Lambda
        # Calculate kz-vector components (0 for complex kz vectors)
        kz = (nr * k_tot)**2 - kxy_max**2 
        if kz < 0:
            kz = 0
        else:
            kz = np.sqrt(kz)
        alpha = np.arccos(kz/k_tot)
        
        return(alpha)
    
    def get_max_angle_deg(self, Lambda, nr):
        """ 
        returns the maximum k-vector angle in degrees
        Lambda .. wavelength
        nr ... real part of refractive index
        """
        return self.get_max_angle(Lambda, nr)/np.pi*180
    
    def get_no_of_modes(self):
        """ 
        returns number of modes used in creating the field
        """
        return self.__no_of_modes
    
    def __create_speckle_field(self, N: int, mode_limit: int, k_space_out: bool):
        # scattering-medium's transmission (speckle k-space representation)
        
        T_medium = np.exp(2*np.pi*1j*np.random.rand(N, N))
        #print(T_medium)
        
        if k_space_out:
            E_speckles_k = self.grid.limit_mode_numbers(T_medium, mode_limit, True, True)
            E_speckles = self.grid.ifft2_phys_spatial(E_speckles_k)
            E_speckles /=  np.max(abs(E_speckles))
        else:
            E_speckles = self.grid.limit_mode_numbers(T_medium, mode_limit, True, False)
            E_speckles /= np.max(abs(E_speckles))
            
        return E_speckles


    def plot_angle_distribution(self, fov_only: bool, no_of_modes=-1, save_path = None, Lambda = -1):
        if Lambda == -1:
            Lambda = self.grid.cavity.Lambda
        
        if fov_only:
            intensities = np.abs(self.get_field_fov_vec(False))**2           
        else:
            intensities = np.abs(self.get_field_tot_vec(False))**2
           
        
        #normalized intensities vector
        cutoff = np.max(intensities)/1000
        intensities = np.where(intensities < cutoff, 0, intensities)
        nonzero_count = np.count_nonzero(intensities)
        intensities /= np.sum(intensities)
        if no_of_modes < 0:
            intensities *= 100
        elif no_of_modes == 0:
            intensities *= nonzero_count
        else:
            intensities *= no_of_modes
            
        mode_vec = self.grid.get_sorted_mode_numbers(fov_only, -1, True)
        mode_vec = mode_vec[intensities!=0]
        intensities = intensities[intensities!=0]
        # unique n_squared values
        unique_n_sqared = np.unique(mode_vec[:,2])
        # Initialize the counts array
        counts = np.zeros_like(unique_n_sqared, dtype=float)
        
        # Compute weighted counts
        p = self.grid.cavity.progress
        p.tic_reset(unique_n_sqared.shape[0], False, "calculating histogram")
        for i, n_squared in enumerate(unique_n_sqared):
            counts[i] = np.sum(intensities[mode_vec[:,2] == n_squared])
            p.tic()
        
        angles = np.array([self.grid.get_angle_from_nxy(not fov_only, np.sqrt(ns), 0, Lambda, 1) for ns in unique_n_sqared]) 
        angles /= np.pi
        angles *= 180
        
        hist, bin_edges = np.histogram(angles, bins=10, weights=counts)
        fig, pax = plt.subplots(dpi=150)
        # plt.figure(dpi=150)  # Width, Height in inches
        plt.hist(angles, bins=bin_edges, weights=counts, edgecolor='black', color='orange')
        plt.xlabel('Angle in Degrees')
        if no_of_modes <0:
            plt.ylabel('% Modes')
        else:
            plt.ylabel('Modes')
        plt.title('Histogram of k-Vector Angles')
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
        
        return hist, bin_edges, intensities
    
###############################################################################
# clsTestImage
# represents a test Image Light Field
############################################################################### 
class clsTestImage(clsLightField):
    def __init__(self, grid):
        super().__init__(grid)
    
    def create_test_image(self, image: int, 
                          flip_horizontal = False, flip_vertical = False):
        """
        Creates a test image
        ----
        image ...... 1: small centered T, 
                     2: large centered T, 
                     3: large centered T with phase shifts, 
                     4: large off-center T, 
                     5: large off-center T with phase shifts 
                     6: 1/5 square beam
        """
        pixels = self.grid.res_fov
        field = np.zeros((pixels, pixels), dtype=self.grid.cavity.dtype_c)
    
        if image == 1:
            # small T
            a = int(round(2000/4096*pixels))
            b = int(round(2200/4096*pixels))
            c = int(round(2020/4096*pixels))
            d = int(round(1940/4096*pixels))
            e = int(round(2070/4096*pixels))
            field[a-1:b, a-1:c] = 1  # horizontal bar
            field[a-1:c, d-1:e] = 1  # vertical bar            
    
        elif image == 2:
            # large T
            a = int(round(1300/4096*pixels))
            b = int(round(1700/4096*pixels))
            c = int(round((4096-1300)/4096*pixels))
            d = int(round(3000/4096*pixels))
            e = int(round(1900/4096*pixels))
            f = int(round(2200/4096*pixels))
            field[a-1:b, a-1:c] = 1  # horizontal bar
            field[a-1:d, e-1:f] = 1  # vertical bar
    
        elif image == 3:
            # large T with phase shifts
            a = int(1300/4096*pixels)
            b = int(1700/4096*pixels)
            c = int((4096-1300)/4096*pixels)
            d = int(3000/4096*pixels)
            e = int(1900/4096*pixels)
            f = int(2200/4096*pixels)
            # vertical bar
            X = np.exp(1j * ((np.arange(1, d - b + 1)[:, np.newaxis]) / (d - b) * 2 * np.pi))
            X = np.repeat(X, f-e+1, axis=1)
            field[b:d, e-1:f] = X
            # horzontal bar
            X = np.exp(1j*np.arange(1, c - a + 2)/(c-a+1)*np.pi)
            X = np.tile(X , (b-a+1, 1))
            field[a-1:b, a-1:c] = X
    
        elif image == 4:
            # large off-center T 
            offset = -900 
            a = max(1, int((1300+offset)/4096*pixels))
            b = int((1700+offset)/4096*pixels)
            c = int((4096-1300+offset)/4096*pixels)
            d = int((3000+offset)/4096*pixels)
            e = int((1900+offset)/4096*pixels)
            f = int((2200+offset)/4096*pixels)
            field[b:d, e-1:f] = 1 # vertical bar
            field[a-1:b, a-1:c] = 1 # horizontal bar
    
        elif image == 5:
            # large off-center T with pahse shifts
            offset = -900 
            a = max(1, int((1300+offset)/4096*pixels))
            b = int((1700+offset)/4096*pixels)
            c = int((4096-1300+offset)/4096*pixels)
            d = int((3000+offset)/4096*pixels)
            e = int((1900+offset)/4096*pixels)
            f = int((2200+offset)/4096*pixels)
            # vertical bar
            X = np.exp(1j * ((np.arange(1, d - b + 1)[:, np.newaxis]) / (d - b) * 2 * np.pi))
            X = np.repeat(X, f-e+1, axis=1)
            field[b:d, e-1:f] = X
            # horzontal bar
            X = np.exp(1j*np.arange(1, c - a + 2)/(c-a+1)*np.pi)
            X = np.tile(X , (b-a+1, 1))
            field[a-1:b, a-1:c] = X
    
        elif image == 6:
            # 1/5 square beam
            ax = np.linspace(-pixels/2, pixels/2, pixels)
            dx = ax[1] - ax[0]
            w = (max(ax) - min(ax) + dx) / 5 + dx
            X2, Y2 = np.meshgrid(ax, ax)
            field = np.where((np.abs(X2/w) <= 1/2) & (np.abs(Y2/w) <= 1/2), 1, 0)

        if not flip_vertical:
            # the image so far is vertically flipped; 
            # hence we need to flip it to have it not flipped 
            field = np.flipud(field)
            
        if flip_horizontal:
            field = np.fliplr(field)
        
        self.set_field(field, False)

###############################################################################
# clsOptComponent
# represents super-class for optical components 
# (two or four path)
###############################################################################    
class clsOptComponent(ABC):
    """ Superclass for two-path and four-path optical components """
    def __init__(self, name, cavity):
        self.__name = name
        self._lambda = 633 / 1000000000
        self._lambda_nm = 633
        self.__cavity = None
        self.__grid = None   
        self.idx = -1
        self.__file_cache_M_bmat = False # should the transfer matrix and its inverse be file cached?
        self.__mem_cache_M_bmat = False # should the transfer matrix and its inverse be memory cached?
        self._M_bmat_tot_mem_cached = False # is the transfer block matrix cached?
        self._inv_M_bmat_tot_mem_cached = False # is the inverse transfer block matrix cached?
        self._M_bmat_tot = None # memory cache for the transfer block matrix
        self._inv_M_bmat_tot = None # memory cache for the transfer block matrix        
        if not cavity is None:
            self._connect_to_cavity(cavity, cavity.grid, -1)
        
    def _connect_to_cavity(self, cavity, grid, idx):
       self.__cavity = cavity
       self.__grid = grid
       self._lambda = cavity.Lambda
       if self.idx == -1:
           self.idx = idx
    
    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name):
        self.__name = name 
        
    @property
    def Lambda_nm(self):
        """ returns wavelength in nm """
        return self._lambda_nm
    
    @Lambda_nm.setter    
    def Lambda_nm(self, lambda_nm: float):
        """ sets wavelength in nm """
        self.lambda_nm = lambda_nm
        self._lambda = lambda_nm / 1000000000
        self.clear_mem_cache()
    
    @property
    def Lambda(self):
        """ returns wavelength in m """
        return self._lambda
    
    @Lambda.setter    
    def Lambda(self, Lambda: float):
        """ sets wavelength in nm """
        self._lambda = Lambda
        self._lambda_nm = Lambda * 1000000000
        self.clear_mem_cache()
           
    @property
    def grid(self):
        """ returns the grid instance """
        return self.__grid
    
    @property
    def cavity(self):
        """ returns the grid instance """
        return self.__cavity
    
    @property
    def file_cache_M_bmat(self):
        return self.__file_cache_M_bmat
    
    @file_cache_M_bmat.setter
    def file_cache_M_bmat(self, c:bool):
        self.__file_cache_M_bmat = c        
        if c:
            self.__mem_cache_M_bmat = False
    
    @property
    def mem_cache_M_bmat(self):
        return self.__mem_cache_M_bmat
    
    @mem_cache_M_bmat.setter
    def mem_cache_M_bmat(self, c:bool):
        self.__mem_cache_M_bmat = c
        if c:
            self.__file_cache_M_bmat = False   
    
    @property
    @abstractmethod    
    def k_space_in_prefer(self):    
        """ returns true, if k_space input is faster """
        pass
    
    @property
    @abstractmethod    
    def k_space_in_dont_care(self):    
        """ returns true, if there is no preference for either 
        position-space or k-space input"""
        pass
    
    @property
    @abstractmethod    
    def k_space_out_prefer(self):    
        """ returns true, if k_space output is faster """
        pass
    
    @property
    @abstractmethod    
    def k_space_out_dont_care(self):    
        """ returns true, if there is no preference for either 
        position-space or k-space output"""
        pass 

    def load_M_bmat_tot(self):
        """ loads block transfer matrix from file"""
        self.cavity.progress.push_print("loading transfer matrix M of '"+self.name+"'")
        filename = self.cavity._full_file_name("M_mat",self.idx)                
        self._M_bmat_tot = joblib.load(filename)
        if isinstance(self._M_bmat_tot, clsBlockMatrix):
            self._M_bmat_tot.keep_tmp_files = True
        self._M_bmat_tot_mem_cached = True
        self.cavity.progress.pop()
        
    def load_inv_M_bmat_tot(self):
        """ loads inverse block transfer matrix from file"""
        self.cavity.progress.push_print("loading inverse transfer matrix inv(M) of '"+self.name+"'")
        filename = self.cavity._full_file_name("invM_mat",self.idx)                
        self._inv_M_bmat_tot = joblib.load(filename)
        if isinstance(self._inv_M_bmat_tot, clsBlockMatrix):
            self._inv_M_bmat_tot.keep_tmp_files = True
        self._inv_M_bmat_tot_mem_cached = True
        self.cavity.progress.pop()        

    def save_M_bmat_tot(self):
        """ saves block transfer matrix to file"""
        self.cavity.progress.push_print("saving transfer matrix M of '"+self.name+"'")
        filename = self.cavity._full_file_name("M_mat",self.idx)            
        if isinstance(self._M_bmat_tot, clsBlockMatrix):
            self._M_bmat_tot.keep_tmp_files = True
        joblib.dump(self._M_bmat_tot, filename)
        self.cavity.progress.pop()
        
    def save_inv_M_bmat_tot(self):
        """ saves inverse block transfer matrix to file"""
        self.cavity.progress.push_print("saving inverse transfer matrix inv(M) of '"+self.name+"'")
        filename = self.cavity._full_file_name("invM_mat",self.idx)               
        if isinstance(self._inv_M_bmat_tot, clsBlockMatrix):
            self._inv_M_bmat_tot.keep_tmp_files = True
        joblib.dump(self._inv_M_bmat_tot, filename)
        self.cavity.progress.pop()        
        
    def delete_M_bmat_tot(self):
        """ deletes the block transfer matrix and inverse block transfer matrix file"""        
        
        self.cavity.delete_mat("M_mat", self.idx, "deleting transfer matrix file of "+self.name)
        self.cavity.delete_mat("invM_mat", self.idx, "deleting inverse transfer matrix file of "+self.name)                        

    def clear_mem_cache(self):
        """ 
        Deletes the M-Matrix and/or the inverse M matrix from memory cache
        Returns True, if something was deleted
        """
        if self._M_bmat_tot_mem_cached and self._inv_M_bmat_tot_mem_cached:
            self.cavity.progress.push_print(
                "deleting transfer matrix M and inverse inv(M) of '"+
                self.name+"' from memory cache") 
            self.cavity.progress.pop()
        elif self._M_bmat_tot_mem_cached :
            self.cavity.progress.push_print(
                "deleting transfer matrix M of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        elif self._inv_M_bmat_tot_mem_cached :
            self.cavity.progress.push_print(
                "deleting inverse transfer matrix inv(M) of '"+
                self.name+"' from memory cache") 
            self.cavity.progress.pop()
        self._M_bmat_tot_mem_cached = False
        self._inv_M_bmat_tot_mem_cached = False
        if isinstance(self._M_bmat_tot, clsBlockMatrix):
            del self._M_bmat_tot
        if isinstance(self._inv_M_bmat_tot, clsBlockMatrix):
            del self._inv_M_bmat_tot
        self._M_bmat_tot = None 
        self._inv_M_bmat_tot = None 

    @property
    def inv_M_bmat_tot(self):        
        """ 
        returns the inverse 2x2 block transfer matrix (total resolution)
        """
        if self._inv_M_bmat_tot_mem_cached:            
            # already cached in memory
            return self._inv_M_bmat_tot
        
        if self.file_cache_M_bmat:
            # file caching active! Check, if file exists
            filename = self.cavity._full_file_name("invM_mat",self.idx)
            if os.path.exists(filename):
                # file exists! Load from file
                self.load_inv_M_bmat_tot()
                return self._inv_M_bmat_tot
        
        self.cavity.progress.push_print("calculating inverse transfer matrix M")
        invM =self.calc_inv_M_bmat_tot() 
        if self.mem_cache_M_bmat:
            self._inv_M_bmat_tot = invM
            self._inv_M_bmat_tot_mem_cached = True                
        
        self.cavity.progress.pop()
        
        if self.file_cache_M_bmat:
            self._inv_M_bmat_tot = invM
            self.save_inv_M_bmat_tot()
            if isinstance(self._M_bmat_tot, clsBlockMatrix):
                del self._M_bmat_tot
            if isinstance(self._inv_M_bmat_tot, clsBlockMatrix):
                del self._inv_M_bmat_tot
            self._M_bmat_tot = None
            self._inv_M_bmat_tot = None
        
        return invM 
    
###############################################################################
# clsOptComponent2port
# represents any 2-port optical component (e.g. lens, mirror, propagation)
###############################################################################    
class clsOptComponent2port(clsOptComponent):
    """ Superclass for two-port optical components """
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        
        
    @property
    @abstractmethod    
    def symmetric(self):    
        """ returns true, if left-right-symmetric """
        pass
    
    @property
    @abstractmethod    
    def T_LTR_mat_tot(self):    
        """ returns the left-to-right transmission matrix (tot resolution) """
        pass
    
    @property
    @abstractmethod    
    def T_RTL_mat_tot(self):    
        """ returns the right-to-left transmission matrix (tot resolution) """
        pass
    
    @property
    @abstractmethod    
    def R_L_mat_tot(self):    
        """ returns the left reflection matrix (tot resolution) """
        pass
    
    @property
    @abstractmethod    
    def R_R_mat_tot(self):    
        """ returns the right reflection matrix (tot resolution) """
        pass
        
    @abstractmethod    
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        """
        Propagates the total input field E_in in positive z-direction (L->R)                
        ----------
        E_in : np.array or float
            input field (in either position-space or k-space).
            can also be a float if k_space_in = True
        k_space_in : bool
            if true:  E_in is a k-space array.
            if false: E_in is a position-space array
        k_space_out : bool
            if true:  output is a k-space array.
            if false: output is a position-space array.
        direction : Dir
            diretktion of propagation (Dir.LTR or Dir.RTL)
        Returns
        -------
        np.array
            field after propagation in k-space or position-space.
        """
    
    @property
    def S_bmat_tot(self):
        """ 
        returns the 2x2 block scatering matrix (total resolution)
        as list [[S11, S12], [S21, S22]] 
        """
        if self.cavity.use_bmatrix_class:
            S = clsBlockMatrix(2, self.cavity.use_swap_files_in_bmatrix_class, 
                               self.cavity.tmp_folder)
            S.set_block(0, 0, self.R_L_mat_tot)
            S.set_block(0, 1, self.T_RTL_mat_tot)
            S.set_block(1, 0, self.T_LTR_mat_tot)
            S.set_block(1, 1, self.R_R_mat_tot)
            
            return S
        else:
            return [[self.R_L_mat_tot, self.T_RTL_mat_tot] , [self.T_LTR_mat_tot, self.R_R_mat_tot]]

    @property
    def dist_phys(self):
        """ returns physical propgation distance in meters """
        return 0.0
    
    @property
    def dist_opt(self):
        """ returns optical propgation distance in meters """
        return 0.0

    def _S_to_M(self, S, p = None, msg = ""):        
        """ 
        converts the scattering matrix S into the transfer matrix M
        both S and M are block matrices
        if p is set to a clsProgressPrinter instance, then tics are sent
        if msg != "" the the progress printer is resetted
        """
        if not p is None:
            if not msg == "":
                p.tic_reset(5, False, msg)
        
        if isinstance(S, clsBlockMatrix):
            M = clsBlockMatrix(2, S.file_caching, S.tmp_dir)
            
            #S21_inv = M[1,1]
            M.set_block(1, 1, mat_inv(S.get_block(1,0)))
            gc.collect()
            if not p is None:
                p.tic()
            
            #S11_S21_inv = M[0, 1]
            M.set_block(0, 1, mat_mul(S.get_block(0,0), M.get_block(1,1)) )
            gc.collect()
            if not p is None:
                p.tic()
            
            a = mat_mul(M.get_block(0,1), S.get_block(1,1))
            gc.collect()
            if not p is None:
                p.tic()
            M.set_block(0, 0, mat_minus(S.get_block(0,1), a))            
            del a
            gc.collect()
            if not p is None:
                p.tic()
                
            b = mat_mul(M.get_block(1,1), S.get_block(1,1))
            gc.collect()
            M.set_block(1, 0, -b)
            del b
            gc.collect()                
            if not p is None:
                p.tic()
                
            return M
            
        else:
        
            S11, S12, S21, S22 = S[0][0], S[0][1], S[1][0], S[1][1]
            S21_inv = mat_inv(S21)                
            S11_S21_inv = mat_mul(S11, S21_inv)
            
            M11 = mat_minus(S12, mat_mul(S11_S21_inv, S22))        
            if not p is None:
                p.tic()
            
            M12 = S11_S21_inv        
            if not p is None:
                p.tic()
            
            M21 = -mat_mul(S21_inv, S22)
            if not p is None:
                p.tic()
            
            M22 = S21_inv
            if not p is None:
                p.tic()
            
            return [[M11, M12], [M21, M22]]
      
    
    def calc_inv_M_bmat_tot(self):
        """ calculates the inverse transfer block matrix """
        return bmat2_inv(self.M_bmat_tot, self.cavity.progress, "inverting M")
    
    @property
    def M_bmat_tot(self):
        """ returns the 2x2 block transfer matrix (total resolution) """
        if self._M_bmat_tot_mem_cached:            
            # already cached in memory
            return self._M_bmat_tot
        
        if self.file_cache_M_bmat:
            # file caching active! Check, if file exists
            filename = self.cavity._full_file_name("M_mat",self.idx)
            if os.path.exists(filename):
                # file exists! Load from file
                self.load_M_bmat_tot()
                return self._M_bmat_tot
        
        self.cavity.progress.push_print("calculating transfer matrix M") 
                        
        if self.cavity.use_bmatrix_class:
            S = clsBlockMatrix(2, self.cavity.use_swap_files_in_bmatrix_class,
                               self.cavity.tmp_folder)                        
                
            S.set_block(0, 0, self.R_L_mat_tot)            
            S.set_block(0, 1, self.T_RTL_mat_tot)            
            S.set_block(1, 0, self.T_LTR_mat_tot)
            S.set_block(1, 1, self.R_R_mat_tot)     
            
            # in case of file-caching in the block matrix class, 
            # we don't need the mem-cached instances 
            # of self.R_L_mat_tot, self.T_RTL_mat_tot, etc any more
            if S.file_caching:
                self.clear_mem_cache()                         
                        
            M = self._S_to_M(S, self.cavity.progress, "converting S-matrix to M-matrix")            
            S = None
            
        else:
            S11 = self.R_L_mat_tot
            S12 = self.T_RTL_mat_tot
            S21 = self.T_LTR_mat_tot
            S22 = self.R_R_mat_tot
            S = [[S11, S12],[S21, S22]]
            
            M = self._S_to_M(S, self.cavity.progress, "converting S-matrix to M-matrix")
            
            S = None
            S11 = None
            S12 = None
            S21 = None
            S22 = None
                
        
        if self.mem_cache_M_bmat:
            self._M_bmat_tot = M
            self._M_bmat_tot_mem_cached = True                
        
        elapsed_time = self.cavity.progress.pop()
        
        #print("XXX")
        if self.file_cache_M_bmat:
            #print("YYY",elapsed_time, self.cavity.file_cache_min_calc_time)
            if elapsed_time > self.cavity.file_cache_min_calc_time:
                self._M_bmat_tot = M
                self.save_M_bmat_tot()
                self._M_bmat_tot = None
                self._inv_M_bmat_tot = None
        
        return M 


###############################################################################
# clsTransmissionTilt
# Tilts the light beam either LTR or RTL
############################################################################### 
class clsTransmissionTilt(clsOptComponent2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__direction =  Dir.LTR
        self.__x_angle_deg = None # angle in deg by which the field is titled in x-direction
        self.__y_angle_deg = None # angle in deg by which the field is titled in y-direction
        self.__x_angle = 0 # angle in rad by which the field is titled in x-direction
        self.__y_angle = 0 # angle in rad by which the field is titled in y-direction
        self.__x_zero_line = 0 # x-coordinate at which there is zero x-phase-shift
        self.__y_zero_line = 0 # y-coordinate at which there is zero y-phase-shift
        self.__x_mask = 1
        self.__y_mask = 1
        self.__T_mat_tot = None
        
    @property
    def x_zero_line(self):
        """
        x-coordinate at which there is zero x-phase-shift
        """
        return self.__x_zero_line

    @x_zero_line.setter
    def x_zero_line(self, x_pos):
        self.__x_zero_line = x_pos
        self.__y_mask = None
        self.__T_mat_tot = None
        

    @property
    def y_zero_line(self):
        """
        y-coordinate at which there is zero y-phase-shift
        """
        return self.__y_zero_line

    @y_zero_line.setter
    def y_zero_line(self, y_pos):
        self.__y_zero_line = y_pos
        self.__x_mask = None
        self.__T_mat_tot = None

    @property
    def x_angle_deg(self):
        """ 
        angle in degree by which the light field is titled in x-direction
        """
        if self.__x_angle_deg is None:
            return self.__x_angle / math.pi * 180
        else:
            return self.__x_angle_deg    
    
    @x_angle_deg.setter
    def x_angle_deg(self, alpha):
        if not self.cavity is None:
            self.cavity.clear()        
        if abs(alpha)>45:
            print("Error. Tilt angle must be <= 45°")
        else:
            self.__x_angle_deg = alpha
            self.__x_angle = None
            self.__x_mask = None
            self.__T_mat_tot = None    
   
    @property
    def y_angle_deg(self):
        """ 
        angle in degree by which the lioght field is titled in y-direction
        """
        if self.__y_angle_deg is None:
            return self.__y_angle / math.pi * 180
        else:
            return self.__y_angle_deg
    
    @y_angle_deg.setter
    def y_angle_deg(self, alpha):
        if not self.cavity is None:
            self.cavity.clear()        
        if abs(alpha)>45:
            print("Error. Tilt angle must be <= 45°")
        else:
            self.__y_angle_deg = alpha
            self.__y_angle = None
            self.__y_mask = None
            self.__T_mat_tot = None

    @property
    def x_angle(self):
        """ 
        angle in rad by which the light field is titled in x-direction
        """
        if self.__x_angle is None:
            return self.__x_angle_deg / 180 * math.pi 
        else:
            return self.__x_angle    
        
        
    @x_angle.setter
    def x_angle(self, alpha):
        if not self.cavity is None:
            self.cavity.clear()        
        if abs(alpha)>math.pi/4:
            print("Error. Tilt angle must be <= pi/4")
        else:
            self.__x_angle = alpha
            self.__x_angle_deg = None
            self.__x_mask = None
            self.__T_mat_tot = None    
        
    @property
    def y_angle(self):
        """ 
        angle in rad by which the light field is titled in y-direction
        """
        if self.__y_angle is None:
            return self.__y_angle_deg / 180 * math.pi 
        else:
            return self.__y_angle    

    @y_angle.setter
    def y_angle(self, alpha):
        if not self.cavity is None:
            self.cavity.clear()        
        if abs(alpha)>math.pi/4:
            print("Error. Tilt angle must be <= pi/4")
        else:
            self.__y_angle = alpha
            self.__y_angle_deg = None
            self.__y_mask = None
            self.__T_mat_tot = None    

    @property 
    def direction(self):
        """ propagation direction in which the light field will be tilted """
        return self.__direction
    
    @direction.setter
    def direction(self, x: Dir):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__direction = x
        
    @property
    def x_mask(self):
        """ returns the tilt-mask in x-direction"""
        if self.__x_mask is None:
            self.__calc_x_mask()
        return self.__x_mask
    
    @property
    def y_mask(self):
        """ returns the tilt-mask """
        if self.__y_mask is None:
            self.__calc_y_mask()
        return self.__y_mask
    
    def __calc_x_mask(self):        
        if self.x_angle == 0:
            self.__x_mask = 1
            
        else:
            # create coordinate grid 
            ax = self.grid.axis_tot
            if self.__x_zero_line != 0:
                ax = ax - self.__x_zero_line            
            x, _ = np.meshgrid(ax, ax)
            
            # Calculate k
            k0 = 2 * np.pi / self._lambda
            alpha = self.x_angle #_deg / 180 * math.pi
            self.__x_mask =  np.exp(1j * k0 * x * math.tan(alpha))
    
    def __calc_y_mask(self):        
        if self.y_angle == 0:
            self.__y_mask = 1
            
        else:
            # create coordinate grid             
            ax = self.grid.axis_tot            
            if self.__y_zero_line != 0:
                ax = ax - self.__y_zero_line            
            _, y = np.meshgrid(ax, ax)
            
            # Calculate k
            k0 = 2 * np.pi / self._lambda
            alpha = self.y_angle  #_deg / 180 * math.pi
            self.__y_mask =  np.exp(1j * k0 * y * math.tan(alpha))
                                    
    
    @property
    def symmetric(self):
        return False
    
    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
        
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space

    @property    
    def k_space_in_dont_care(self):    
        return False

    @property  
    def k_space_out_dont_care(self):    
        return False    
    
    @property 
    def T_LTR_mat_tot(self):    
        """ returns the left-to-right transmission matrix (total resolution)"""
        if self.direction == Dir.LTR:
            # in LTR direction there can be a tilt 
            if self.x_angle_deg == 0 and self.y_angle_deg == 0:
                # but actually it is a 0 degree tilt
                return 1
            else:
                # calculate tilt transfer matrix if neccessary, and return it
                if self.__T_mat_tot is None:
                    self.calc_T_mat_tot()
                return self.__T_mat_tot 
        else:
            # no tilt in LTR direction
            return 1
    
    @property    
    def T_RTL_mat_tot(self):    
        """ returns the right-to-left transmission matrix (total resolution)"""
        if self.direction == Dir.RTL:
            # in RTL direction there can be a tilt 
            if self.x_angle_deg == 0 and self.y_angle_deg == 0:
                # but actually it is a 0 degree tilt
                return 1
            else:
                # calculate tilt transfer matrix if neccessary, and return it
                if self.__T_mat_tot is None:
                    self.calc_T_mat_tot()
                return self.__T_mat_tot 
        else:
            # no tilt in RTL direction
            return 1
    
    def _calc_T_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        if self.x_angle_deg != 0:
            psi *= self.__x_mask
        if self.y_angle_deg != 0:
            psi *= self.__y_mask
        gc.collect()
        return self.grid.arr_to_vec(psi, False, False) 
    
    def calc_T_mat_tot(self):
        """
        Creates a transmission matrix for propagation through tilt mask 
        (total resolution)
        """        
        # simulate propagation for all modes in parallel        
        self.cavity.progress.push_print("calculating transmission matrix T")
        
        if self.x_angle_deg == 0 and self.y_angle_deg == 0:
            self.__T_mat_tot = 1
        else:
            if self.x_angle_deg != 0:
                if self.__x_mask is None:
                    self.__calc_x_mask()
            if self.y_angle_deg != 0:
                if self.__y_mask is None:
                    self.__calc_y_mask()
            NN = self.grid.res_tot**2
            
            keep_clsBlockMatrix_alive(True)
            result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_T_mat_tot_helper)(i) for i in range(NN))
            keep_clsBlockMatrix_alive(False)
            self.__T_mat_tot = np.column_stack(result)
            
            
        self.cavity.progress.pop()
    
    @property  
    def R_L_mat_tot(self):    
        """ returns the left reflection matrix (total resolution)"""
        return 0
    
    @property
    def R_R_mat_tot(self):    
        """ returns the right reflection matrix (total resolution)"""
        return 0
    
    
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        """
        Propagates the input field E_in in positive z-direction (LTR) 
        or negative z-Direction (RTL)
        """
        if self.x_angle_deg == 0 and self.y_angle_deg == 0:
            if k_space_in == k_space_out:
                return E_in
            
        ax = self.grid.axis_tot
        if k_space_in:
            # input in k-space: Convert to position-space
            E_in = ifft2_phys_spatial(E_in, ax)
        
        # apply tilt-mask
        if direction == self.direction:
            if self.x_angle_deg != 0:
                E_in *= self.x_mask
            if self.y_angle_deg != 0:
                E_in *= self.y_mask
        
        if k_space_out:
            # output in k-space: Convert to k-space
            E_in = fft2_phys_spatial(E_in, ax)
        
        return E_in
    
    def clear_mem_cache(self):   
        super().clear_mem_cache()
        if not self.__x_mask is None:
            self.__x_mask = None
            self.cavity.progress.push_print(
                "deleting phase mask (x-shift) of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if not self.__y_mask is None:
            self.__y_mask = None
            self.cavity.progress.push_print(
                "deleting phase mask (y-shift) of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if not self.__T_mat_tot is None:
            self.__T_mat_tot = None
            self.cavity.progress.push_print(
                "deleting transmission matrix T of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
            


###############################################################################
# clsMirrorBase2port
# base clas for flat and curved mirror
############################################################################### 
class clsMirrorBase2port(clsOptComponent2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self._tilt_mask_x_L = None
        self._tilt_mask_x_R = None
        self._tilt_mask_y_L = None
        self._tilt_mask_y_R = None
        self.__sym_phase = True # symmetric phase behavior
        self.__R = 1 # power reflection from left and right
        self.__T = 0 # power transmission in either direction
        self.__r = -1 # complex reflection coefficient from left and right
        self.__t = 0 # complex transmission coefficient from either side    
        self.__LTR_transm_behaves_like_refl_left = False
        self.__LTR_transm_behaves_like_refl_right = False
        self.__LTR_transm_behaves_neutral = False
        self.__RTL_transm_behaves_like_refl_left = False
        self.__RTL_transm_behaves_like_refl_right = False
        self.__RTL_transm_behaves_neutral = False
        self.__no_reflection_phase_shift = False
        self.__left_side_relevant = True
        self.__right_side_relevant = True
        self.__rot_around_x_deg = 0 # rotation around x-axis in degrees
        self.__rot_around_y_deg = 0 # rotation around y-axis in degrees
        self.__rot_around_x = None  # rotation around x-axis in rad
        self.__rot_around_y = None  # rotation around y-axis in rad
        self.__project_according_to_angles = False
        self.__consider_tilt_astigmatism = False
        self.__apply_tilt_masks = True
        self.__incident_angle_y_deg = 0 # incident angle in y direction
        self.__incident_angle_y = None 
    
    def clear_mem_cache(self):   
        super().clear_mem_cache()
        if (not self._tilt_mask_x_L is None) or (not self._tilt_mask_x_R is None):
            self._tilt_mask_x_L = None
            self._tilt_mask_x_R = None
            self.cavity.progress.push_print(
                "deleting phase mask (x-shift) of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if (not self._tilt_mask_y_L is None) or (not self._tilt_mask_y_R is None):
            self._tilt_mask_y_L = None
            self._tilt_mask_y_R = None
            self.cavity.progress.push_print(
                "deleting phase mask (y-shift) of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
    
    def _set_rt(self):
        if self.__sym_phase:
            self.__r = -self.__R - 1j * math.sqrt(self.__R) * math.sqrt(1 - self.__R)
            #self.__t = 1 + self.__r
            R = 1-self.__T
            self.__t = 1-R - 1j * math.sqrt(R) * math.sqrt(1 - R)
        else:
            self.__r = math.sqrt(self.__R)
            self.__t = math.sqrt(self.__T)

    @property
    def R(self):
        """ power reflection on either side """
        return self.__R
    
    @R.setter
    def R(self, R_new):
        if R_new>1:
            print("Warning: R must be <= 1. Setting R to 1.")
            R_new = 1
        elif R_new<0:
            print("Warning: R must be >= 0. Setting R to 0.")
            R_new = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__R = R_new
        self.__T = 1 - R_new
        self._set_rt()
    
    @property
    def T(self):
        """ power transmission on either side """
        return self.__T
        
    @T.setter
    def T(self, T_new):
        if T_new>1:
            print("Warning: T must be <= 1. Setting T to 1.")
            T_new = 1
        elif T_new<0:
            print("Warning: T must be >= 0. Setting T to 0.")
            T_new = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__T = T_new
        self.__R = 1 - T_new
        self._set_rt()
    
    def set_T_non_energy_conserving(self, T):
        """" 
        allows to set an unphysical (non-energy-conerving) power transmissivity
        e.g. to simulate perfectly reflecting mirrors with R=1 which would
        cause the transfer matrix to become singular
        """
        if not self.cavity is None:
            self.cavity.clear()       
        self.__T = T
        self._set_rt()
    
    @property
    def r_L(self):
        if self.no_reflection_phase_shift:
            # the flag that there should be no phase shift is set
            if self.__sym_phase:
                # but self.__r is complex; return sqrt(self.__R) instead
                return math.sqrt(self.__R)
            else:
                # self.__r is alread real-valued, 
                # it's faster to return this value
                return self.__r
        else:
            # normal reflection behavior
            return self.__r
            
    @property
    def r_R(self):
        if self.no_reflection_phase_shift:
            # the flag that there should be no phase shift is set
            if self.__sym_phase:
                # but self.__r is complex; return sqrt(self.__R) instead
                return math.sqrt(self.__R)
            else:
                # self.__r is alread real-valued, 
                # it's faster to return this value
                return self.__r
        else:
            # normal reflection behavior
            if self.__sym_phase:
                return self.__r 
            else:
                return -self.__r
    
    @property
    def t_LTR(self):
        return self.__t 
    
    @property
    def t_RTL(self):
        return self.__t 
    
    @property
    def sym_phase(self):
        return self.__sym_phase
        
    def set_phys_behavior(self, sym_phase: bool):
        """ 
        If sym_phase == True (default), then the mirror behaves
        physically insofar, as if it has the same reflection coefficient 
        on the left and right side, both will get the same phase. 
        If sym_phase == False, the mirror follows the convention that r_L, t_LTR 
        and t_RTL are real positive numbers, and r_R is a real negative number 
        """
        if not self.cavity is None:
            self.cavity.clear()       
        self.__sym_phase = sym_phase
        self._set_rt()          
    
    @property
    def apply_tilt_masks(self):
        """ If true, tilt masks are applied for tilted mirrors """
        return self.__apply_tilt_masks
    
    @apply_tilt_masks.setter
    def apply_tilt_masks(self, new_val):
        if not self.cavity is None:
            self.cavity.clear()    
        if new_val == False:
            self._tilt_mask_x_L = None
            self._tilt_mask_x_R = None
            self._tilt_mask_y_L = None
            self._tilt_mask_y_R = None
        self.__apply_tilt_masks = new_val
        
    
    @property
    def project_according_to_angles(self):
        """ 
        If true, the size of the exiting light field is changed, because it is 
        projected onto the mirror according to the incident angle and the 
        mirror tilt
        """
        return self.__project_according_to_angles
    
    @project_according_to_angles.setter
    def project_according_to_angles(self, new_val):
        self.__project_according_to_angles = new_val
        
    @property
    def consider_tilt_astigmatism(self):
        """ 
        If true, the size of the exiting light field is changed, because it is 
        projected onto the mirror according to the incident angle and the 
        mirror tilt
        """
        return self.__consider_tilt_astigmatism
    
    @consider_tilt_astigmatism.setter
    def consider_tilt_astigmatism(self, new_val):
        self.__consider_tilt_astigmatism = new_val
    
    def get_projection_factor_y1(self, side: Side):
        """ 
        Factor by which the field projected onto the mirror plane
        gets larger or smaller in y-direction compared to the incident field
        considering mirror tilt and incident angle.
        Only cnsidered when project_according_to_incident_angle = True
        """
        mirr_tilt = self.rot_around_x
        if side == Side.RIGHT:
            mirr_tilt = -mirr_tilt
        
        factor = math.cos(self.incident_angle_y)
        factor /= math.cos(mirr_tilt + self.incident_angle_y)
            
        return factor
    
    def get_projection_factor_y2(self, side: Side):
        """ 
        Factor by which the reflected outgoing field gets larger or smaller
        in y-direction compared to the incident field 
        considering mirror tilt and incident angle.
        Only cnsidered when project_according_to_incident_angle = True
        """
        mirr_tilt = self.rot_around_x
        if side == Side.RIGHT:
            mirr_tilt = -mirr_tilt
        
        factor = math.cos(self.incident_angle_y)
        factor /= math.cos(2*mirr_tilt + self.incident_angle_y)
            
        return factor    
    
    @property
    def rot_around_x_deg(self):
        """ 
        angle in degree by which the mirror is rotated around the x axis
        positive values means: Left reflection moves towards positive y direction
        ("up"), and the right reflection moves towards negative y values ("down")
        """
        if self.__rot_around_x_deg is None:
            return self.__rot_around_x / math.pi * 180
        else:
            return self.__rot_around_x_deg
     
    @rot_around_x_deg.setter
    def rot_around_x_deg(self, alpha):        
        if abs(alpha)>=45:
            print("Error. Tilt angle must be < 45°")
        else:
            if not self.cavity is None:
                self.cavity.clear()       
            self.__rot_around_x_deg = alpha
            self.__rot_around_x = None            
   
    @property
    def rot_around_y_deg(self):
        """ 
        angle in degree by which the mirror is rotated arund the y axis
        positive values manes: Left reflection moves towards negative x direction
        ("left" if seen from the right side, or "right" if seen from the left side), 
        and the right reflection moves towards positive x values ("right"
        if seen from the right side, or "left" if seen from the left side)
        """
        if self.__rot_around_y_deg is None:
            return self.__rot_around_y / math.pi * 180
        else:
            return self.__rot_around_y_deg
    
    @rot_around_y_deg.setter
    def rot_around_y_deg(self, alpha):
        if abs(alpha)>=45:
            print("Error. Tilt angle must be < 45°")
        else:
            if not self.cavity is None:
                self.cavity.clear()       
            self.__rot_around_y_deg = alpha
            self.__rot_around_y = None            

    @property
    def rot_around_x(self):
        """ 
        angle in radians by which the mirror is rotated around the x axis
        positive values means: Left reflection moves towards positive y direction
        ("up"), and the right reflection moves towards negative y values ("down")
        """
        if self.__rot_around_x is None:
            return self.__rot_around_x_deg / 180 * math.pi
        else:
            return self.__rot_around_x
     
    @rot_around_x.setter
    def rot_around_x(self, alpha):        
        if abs(alpha)>=math.pi/4:
            print("Error. Tilt angle must be < pi/4")
        else:
            if not self.cavity is None:
                self.cavity.clear()       
            self.__rot_around_x = alpha
            self.__rot_around_x_deg = None    

    @property
    def rot_around_y(self):
        """ 
        angle in radians by which the mirror is rotated around the x axis
        positive values means: Left reflection moves towards positive y direction
        ("up"), and the right reflection moves towards negative y values ("down")
        """
        if self.__rot_around_y is None:
            return self.__rot_around_y_deg / 180 * math.pi
        else:
            return self.__rot_around_y
     
    @rot_around_y.setter
    def rot_around_y(self, alpha):        
        if abs(alpha)>=math.pi/4:
            print("Error. Tilt angle must be < pi/4")
        else:
            if not self.cavity is None:
                self.cavity.clear()       
            self.__rot_around_y = alpha
            self.__rot_around_y_deg = None

    
    @property
    def incident_angle_y_deg(self):
        """ 
        incident angle in y-direction in degree
        positive values mean up-tilt
        """
        if self.__incident_angle_y_deg is None:
            return self.__incident_angle_y / math.pi * 180
        else:
            return self.__incident_angle_y_deg
     
    @incident_angle_y_deg.setter
    def incident_angle_y_deg(self, alpha):        
        if abs(alpha)>=45:
            print("Error. Tilt angle must be < 45°")
        else:
            if not self.cavity is None:
                self.cavity.clear()       
            self.__incident_angle_y_deg = alpha
            self.__incident_angle_y = None            
            
    @property
    def incident_angle_y(self):
        """ 
        incident angle in y-direction in rad
        positive values mean down-tilt
        """
        if self.__incident_angle_y is None:
            return self.__incident_angle_y_deg / 180 * math.pi
        else:
            return self.__incident_angle_y
     
    @incident_angle_y.setter
    def incident_angle_y(self, alpha):        
        if abs(alpha)>=math.pi/2:
            print("Error. Tilt angle must be < pi/2")
        else:
            if not self.cavity is None:
                self.cavity.clear()       
            self.__incident_angle_y = alpha
            self.__incident_angle_y_deg = None


    @property
    def no_reflection_phase_shift(self):
        """ If True, then there is no reflection phase shift 
        """
        return self.__no_reflection_phase_shift
    
    @no_reflection_phase_shift.setter
    def no_reflection_phase_shift(self, new_val: bool):
        if not self.cavity is None:
            self.cavity.clear() 
        self.__no_reflection_phase_shift = new_val
    
    @property
    def LTR_transm_behaves_like_refl_left(self):
        """ If True, then the LTR transmissive behavior acts like the reflection 
        from left, and the reflection from left is zero """
        return self.__LTR_transm_behaves_like_refl_left

    @LTR_transm_behaves_like_refl_left.setter
    def LTR_transm_behaves_like_refl_left(self, new_val: bool):
        if not self.cavity is None:
            self.cavity.clear() 
        if new_val:
            if not self.__left_side_relevant:
                print("The setting .left_side_relevant was set to False.")
                print("Switching setting .left_side_relevant to True for consistency.")
                self.left_side_relevant = True
            self.__LTR_transm_behaves_like_refl_left = True
            self.__LTR_transm_behaves_like_refl_right= False
            self.__LTR_transm_behaves_neutral = False
        else:
            self.__LTR_transm_behaves_like_refl_left = False
                    
    @property
    def LTR_transm_behaves_like_refl_right(self):        
        """ If True, then the LTR transmissive behavior acts like the reflection 
        from right, and the reflection from left is zero """
        return self.__LTR_transm_behaves_like_refl_right
    
    @LTR_transm_behaves_like_refl_right.setter
    def LTR_transm_behaves_like_refl_right(self, new_val: bool):
        if not self.cavity is None:
            self.cavity.clear() 
        if new_val:
            if not self.__right_side_relevant:
                print("The setting .right_side_relevant was set to False.")
                print("Switching setting .right_side_relevant to True for consistency.")
                self.right_side_relevant = True
            self.__LTR_transm_behaves_like_refl_right = True
            self.__LTR_transm_behaves_like_refl_left = False
            self.__LTR_transm_behaves_neutral = False
        else:
            self.__LTR_transm_behaves_like_refl_right = False

    @property
    def LTR_transm_behaves_neutral(self):
        """ If True, then the LTR transmissive behavior is neutral (does nothing) 
        and the reflection from left is zero """
        return self.__LTR_transm_behaves_neutral

    @LTR_transm_behaves_neutral.setter
    def LTR_transm_behaves_neutral(self, new_val: bool):
        if not self.cavity is None:
            self.cavity.clear() 
        if new_val:            
            self.__LTR_transm_behaves_like_refl_left = False
            self.__LTR_transm_behaves_like_refl_right= False
            self.__LTR_transm_behaves_neutral = True
        else:
            self.__LTR_transm_behaves_neutral = False

    @property
    def RTL_transm_behaves_like_refl_left(self):
        """ If True, then the RTL transmissive behavior acts like the reflection 
        from left, and the reflection from right is zero """
        return self.__RTL_transm_behaves_like_refl_left

    @RTL_transm_behaves_like_refl_left.setter
    def RTL_transm_behaves_like_refl_left(self, new_val: bool):
        if not self.cavity is None:
            self.cavity.clear() 
        if new_val:
            if not self.__left_side_relevant:
                print("The setting .left_side_relevant was set to False.")
                print("Switching setting .left_side_relevant to True for consistency.")
                self.left_side_relevant = True
            self.__RTL_transm_behaves_like_refl_left = True
            self.__RTL_transm_behaves_like_refl_right= False
            self.__RTL_transm_behaves_neutral = False
        else:
            self.__RTL_transm_behaves_like_refl_left = False
                    
    @property
    def RTL_transm_behaves_like_refl_right(self):        
        """ If True, then the RTL transmissive behavior acts like the reflection 
        from right, and the reflection from right is zero """
        return self.__RTL_transm_behaves_like_refl_right
    
    @RTL_transm_behaves_like_refl_right.setter
    def RTL_transm_behaves_like_refl_right(self, new_val: bool):
        if not self.cavity is None:
            self.cavity.clear() 
        if new_val:
            if not self.__right_side_relevant:
                print("The setting .right_side_relevant was set to False.")
                print("Switching setting .right_side_relevant to True for consistency.")
                self.right_side_relevant = True
            self.__RTL_transm_behaves_like_refl_right = True
            self.__RTL_transm_behaves_like_refl_left = False
            self.__RTL_transm_behaves_neutral = False
        else:
            self.__RTL_transm_behaves_like_refl_right = False
        
    @property
    def RTL_transm_behaves_neutral(self):
        """ If True, then the LTR transmissive behavior is neutral (does nothing) 
        and the reflection from left is zero """
        return self.__RTL_transm_behaves_neutral

    @RTL_transm_behaves_neutral.setter
    def RTL_transm_behaves_neutral(self, new_val: bool):
        if not self.cavity is None:
            self.cavity.clear() 
        if new_val:            
            self.__RTL_transm_behaves_like_refl_left = False
            self.__RTL_transm_behaves_like_refl_right= False
            self.__RTL_transm_behaves_neutral = True
        else:
            self.__RTL_transm_behaves_neutral = False
    
    
    @property
    def left_side_relevant(self):
        """ If False, the reflection on the left side may be calculated 
        incorrectly (e.g. for tilted mirrors) for sake of speed """
        return self.__left_side_relevant
    
    @left_side_relevant.setter
    def left_side_relevant(self, r: bool):
        if not self.cavity is None:
            self.cavity.clear()                
        self.__left_side_relevant = r
    
    @property
    def right_side_relevant(self):
        """ If False, the reflection on the right side may be calculated 
        incorrectly (e.g. for tilted mirrors) for sake of speed """
        return self.__right_side_relevant
    
    @right_side_relevant.setter
    def right_side_relevant(self, r: bool):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__right_side_relevant = r     
    
    @property
    def mirror_tilted(self):
        """ returns true, if the mirror is tilted"""
        return (self.rot_around_x != 0 or self.rot_around_y != 0)
        
    @property
    def left_refl_size_adjust(self):
        """ Returns true, if the outgoing filed will be adjusted in size
        because of projetion effects """
        out_field_stretch = False                
        if self.project_according_to_angles:            
            out_field_stretch = (self.get_projection_factor_y2(Side.LEFT) != 1)
        return out_field_stretch
    
    @property
    def right_refl_size_adjust(self):
        """ Returns true, if the outgoing filed will be adjusted in size
        because of projection effects """
        out_field_stretch = False                
        if self.project_according_to_angles:            
            out_field_stretch = (self.get_projection_factor_y2(Side.RIGHT) != 1)
        return out_field_stretch
    
    @property
    def left_mask_adjust(self):
        """ Returns true, if the left reflection mask or LTR lens mask needs
        to be stretched because of projection effects """
        stretch = False                
        if self.consider_tilt_astigmatism:            
            stretch = (self.get_projection_factor_y1(Side.LEFT) != 1)
        return stretch
    
    @property
    def right_mask_adjust(self):
        """ Returns true, if the right reflection mask or RTL lens mask needs
        to be stretched because of projection effects """
        stretch = False                
        if self.consider_tilt_astigmatism:            
            stretch = (self.get_projection_factor_y1(Side.RIGHT) != 1)
        return stretch
    
    def calc_tilt_masks_x(self):
        # shift in x-direction (depends on y rotation)
        if self.rot_around_y == 0 or (not self.apply_tilt_masks):
            self._tilt_mask_x_L = 1
            self._tilt_mask_x_R = 1
        else:
            ax = self.grid.axis_tot
            x, _ = np.meshgrid(ax, ax)
            k0 = 2 * np.pi / self._lambda # wave number
            alpha = self.rot_around_y # tilt angle in radians
            # positive y-rotation causes left reflection to move in negative x direction
            if self.left_side_relevant:
                self._tilt_mask_x_L =  np.exp(1j * k0 * x * math.tan(-alpha * 2))
            # positive y-rotation causes right reflection to move in positive x direction
            if self.right_side_relevant:
                self._tilt_mask_x_R =  np.exp(1j * k0 * x * math.tan(alpha * 2))
    
    def calc_tilt_masks_y(self):
        # shift in y-direction (depends on x rotation)
        if self.rot_around_x == 0 or (not self.apply_tilt_masks):
            self._tilt_mask_y_L = 1
            self._tilt_mask_y_R = 1
        else:
            ax = self.grid.axis_tot
            _, y = np.meshgrid(ax, ax)
            k0 = 2 * np.pi / self._lambda # wave number
            alpha = self.rot_around_x # tilt angle in radians
            # positive x-rotation causes left reflection to move in positive y direction
            if self.left_side_relevant:
                self._tilt_mask_y_L =  np.exp(1j * k0 * y * math.tan(alpha * 2))
            # positive x-rotation causes right reflection to move in negative y direction
            if self.right_side_relevant:
                self._tilt_mask_y_R =  np.exp(1j * k0 * y * math.tan(-alpha * 2))
    
    @property
    def tilt_mask_x_L(self):
        """ returns the tilt-mask in x-direction on the left side """
        if self._tilt_mask_x_L is None and self.left_side_relevant:
            self.calc_tilt_masks_x()
        return self._tilt_mask_x_L
    
    @property
    def tilt_mask_x_R(self):
        """ returns the tilt-mask in x-direction on the right side """
        if self._tilt_mask_x_R is None and self.right_side_relevant:
            self.calc_tilt_masks_x()
        return self._tilt_mask_x_R
    
    @property
    def tilt_mask_y_L(self):
        """ returns the tilt-mask in y-direction on the left side """
        if self._tilt_mask_y_L is None and self.left_side_relevant:
            self.calc_tilt_masks_y()
        return self._tilt_mask_y_L
    
    @property
    def tilt_mask_y_R(self):
        """ returns the tilt-mask in y-direction on the right side """
        if self._tilt_mask_y_R is None and self.right_side_relevant:
            self.calc_tilt_masks_y()
        return self._tilt_mask_y_R
    
###############################################################################
# represents a (smitransparent) curved mirror
############################################################################### 
class clsCurvedMirror(clsMirrorBase2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__f_R_L = 0.05 # left reflection focal length
        self.__f_R_L_mm = 50 # left reflection focal length in mm
        self.__f_R_R = math.inf # right reflection focal length
        self.__f_R_R_mm = math.inf # right reflection focal length in mm
        self.__n = 1.5 # refractive index (relevant for passing light)
        self.__mirror_type_spherical = True
        self.__mirror_type_perfect = False
        self.__mirror_mask_L = None
        self.__mirror_mask_R = None
        self.__lens_mask = None
        self.__RT_matrices = None
        self._R_L_mat_available = False
        self._R_R_mat_available = False
        self._T_mat_available = False            
        self.par_RT_calc = True # parallel calcuation of R and T matrices 
        self._proj_factor_y2_L = 1
        self._proj_factor_y2_R = 1
        self._proj_factor_y1_L = 1
        self._proj_factor_y1_R = 1

         
    @clsMirrorBase2port.rot_around_x_deg.setter
    def rot_around_x_deg(self, alpha):        
        clsMirrorBase2port.rot_around_x_deg.fset(self, alpha)
        self.__tilt_mask_x_L = None
        self.__tilt_mask_x_R = None
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None
       
    @clsMirrorBase2port.rot_around_y_deg.setter
    def rot_around_y_deg(self, alpha):
        clsMirrorBase2port.rot_around_y_deg.fset(self, alpha)
        self.__tilt_mask_y_L = None
        self.__tilt_mask_y_R = None
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None
        
    @clsMirrorBase2port.rot_around_x.setter
    def rot_around_x(self, alpha):        
        clsMirrorBase2port.rot_around_x.fset(self, alpha)  
        self.__tilt_mask_x_L = None
        self.__tilt_mask_x_R = None
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None
     
    @clsMirrorBase2port.rot_around_y.setter
    def rot_around_y(self, alpha):        
        clsMirrorBase2port.rot_around_y.fset(self, alpha)
        self.__tilt_mask_x_L = None
        self.__tilt_mask_x_R = None
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None        
    
    @property
    def _RT_matrices(self):
        """ 
        0,0 ... R_L_mat_tot
        0,1 ... T_mat_tot
        1,1 ... R_R_mat_tot
        """
        
        delete_old_instance = False
        create_new_instance = False
        if self.__RT_matrices is None:
            # no current instance: create a new instance
            create_new_instance = True
            
        else:
            # there is already an instance
            if self.cavity.use_swap_files_in_bmatrix_class:
                # it should be set to file caching
                if not self.__RT_matrices.file_caching:
                    # but it is not!
                    delete_old_instance = True
                    create_new_instance = True
                
                elif self.__RT_matrices.tmp_dir != self.cavity.tmp_folder:
                    # the current instance is set to file caching (as it should)
                    # but it refers to a wrong tmp folder!
                    delete_old_instance = True
                    create_new_instance = True
                
            else:
                # the current instance should *not* be set to file caching
                if self.__RT_matrices.file_caching:
                    # but it is!
                    delete_old_instance = True
                    create_new_instance = True
            
        if delete_old_instance:
            self._R_L_mat_available = False
            self._R_R_mat_available = False
            self._T_mat_available = False
            self.__RT_matrices = None
            gc.collect()
        
        if create_new_instance:
            self.__RT_matrices = clsBlockMatrix(2, 
                                            self.cavity.use_swap_files_in_bmatrix_class,
                                            self.cavity.tmp_folder)
        
        return self.__RT_matrices

    
    @clsMirrorBase2port.left_side_relevant.setter
    def left_side_relevant(self, r: bool):
        clsMirrorBase2port.left_side_relevant.fset(self, r)
        self._R_L_mat_available = False
        self._RT_matrices.set_block(0, 0, 0)
        self._T_mat_available = False
        self._RT_matrices.set_block(0, 1, 0)        
        
    @clsMirrorBase2port.right_side_relevant.setter
    def right_side_relevant(self, r: bool):
        clsMirrorBase2port.right_side_relevant.fset(self, r)
        self._R_R_mat_available = False
        self._RT_matrices.set_block(1, 1, 0)
        self._T_mat_available = False
        self._RT_matrices.set_block(0, 1, 0)    
            
    def _set_rt(self):
        super()._set_rt()
        
        self._R_L_mat_available = False
        self._RT_matrices.set_block(0, 0, 0)
            
        self._R_R_mat_available = False
        self._RT_matrices.set_block(1, 1, 0)

        self._T_mat_available = False
        self._RT_matrices.set_block(0, 1, 0)
    
    @property 
    def n(self):
        """ Refractive index (for passing light) """
        return self.__n
    
    @n.setter
    def n(self, n_new):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__n = n_new
                    
    @property
    def lens_mask(self):
        """ returns the lens-mask (for light transmission) """
        if self.__lens_mask is None:
            self.__calc_lens_mask()
        return self.__lens_mask
    
    @property
    def mirror_mask_L(self):
        """ returns the mirror-mask for the left facet """
        if self.__mirror_mask_L is None:
            self.__calc_mirror_mask(Side.LEFT)
        return self.__mirror_mask_L
    
    @property
    def mirror_mask_R(self):
        """ returns the mirror-mask for the right facet """
        if self.__mirror_mask_R is None:
            self.__calc_mirror_mask(Side.RIGHT)
        return self.__mirror_mask_R
    
    @property
    def radius_L(self):
        """ 
        radius of left surface 
        (positive for convex surface, negative for concave surface)
        """            
        return -2*self.f_R_L
    
    @property
    def radius_R(self):
        """ 
        radius of right surface 
        (negative for convex surface, positve for concave surface)
        """            
        return 2*self.f_R_R
    
    @property
    def f_T(self):
        """ returns the focal length for transmitted light """
        if self.radius_L == self.radius_R or self.n==1:
            # two equally curved surfaces 
            # (one concave, the other convex; or both flat)
            # or refractive index = 1 -> no lens effect
            return math.inf
        else:
            return 1/((self.n-1)*(1/self.radius_L - 1/self.radius_R))
    
    def __calc_lens_mask(self):
        # TODO: Consider tilt
        if (not self.left_side_relevant) or (not self.right_side_relevant):
            #only one side relevamnt -> tranmission behavior not relevant
            self.__lens_mask = 1
            return
        
        # Create lens mask for transmitted light
        if self.__f_R_L == math.inf and self.__f_R_R == math.inf:
            # two flat surfaces
            self.__lens_mask = 1
            return
        
        # Create coordinate grid 
        ax = self.grid.axis_tot
        x, y = np.meshgrid(ax, ax)
        # Calculate k
        k0 = 2 * np.pi / self._lambda
        
        # at least one of the surfaces is curved
        f_T = self.f_T
        if math.isinf(f_T):
            self.__lens_mask = 1
        elif self.__mirror_type_spherical:
            # spherical mirror                
            self.__lens_mask = np.exp(-1j * k0 / (2 * f_T) * (x**2 + y**2))                                
        else:
            # perfect aspherical mirror                
            self.__lens_mask = np.exp(-1j * k0 * 
                                  (np.sqrt(f_T**2 + x**2 + y**2) - f_T))
               
    def __calc_mirror_mask(self, side: Side):
        
        if side == Side.LEFT and not self.left_side_relevant:
            self.__mirror_mask_L = 1
            return
        
        if side == Side.RIGHT and not self.right_side_relevant:
            self.__mirror_mask_R = 1
            return
        
        # Create coordinate grid 
        ax_x = self.grid.axis_tot
        ax_y = self.grid.axis_tot
        
        if side == Side.LEFT:
            if self.left_mask_adjust:
                ax_y = ax_y * self.get_projection_factor_y1(Side.LEFT)
                
        if side == Side.RIGHT:
            if self.right_mask_adjust:
                ax_y = ax_y * self.get_projection_factor_y1(Side.RIGHT)
        
        x, y = np.meshgrid(ax_x, ax_y)
        # Calculate k
        k0 = 2 * np.pi / self._lambda
        
        # Create mirror mask
        if self.__mirror_type_spherical:
            # spherical mirror
            if side == Side.LEFT:
                self.__mirror_mask_L = np.exp(-1j * k0 / (2 * self.__f_R_L) * 
                                      (x**2 + y**2))
            else:
                self.__mirror_mask_R = np.exp(-1j * k0 / (2 * self.__f_R_R) * 
                                      (x**2 + y**2))
                
        else:
            # perfect aspherical mirror
            if side == Side.LEFT:
                self.__mirror_mask_L = np.exp(-1j * k0 * 
                                      (np.sqrt(self.__f_R_L**2 + x**2 + y**2) 
                                       - self.__f_R_L))
            else:
                self.__mirror_mask_R = np.exp(-1j * k0 * 
                                      (np.sqrt(self.__f_R_R**2 + x**2 + y**2) 
                                       - self.__f_R_R))
    
    def clear_mem_cache(self):   
        super().clear_mem_cache()
            
        if not self.__lens_mask is None:            
            self.__lens_mask = None
            self.cavity.progress.push_print(
                "deleting lens phase mask of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
            
        if (not self.__mirror_mask_L is None) or (not self.__mirror_mask_R is None):            
            self.__mirror_mask_L = None
            self.__mirror_mask_R = None
            self.cavity.progress.push_print(
                "deleting mirror phase masks of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
            
        if self._R_L_mat_available or self._R_R_mat_available:           
            self._R_L_mat_available = False
            self._RT_matrices.set_block(0, 0, 0)
            self._R_R_mat_available = False
            self._RT_matrices.set_block(1, 1, 0)
            self.cavity.progress.push_print(
                "deleting left and right reflection matrix of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
            
        if self._T_mat_available:  
            self._T_mat_available = False
            self._RT_matrices.set_block(0, 1, 0)
            self.cavity.progress.push_print(
                "deleting transmission matrix of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
    
    @property
    def mirror_type_spherical(self):
        """ returns True if it is a spherical mirror """
        return self.__mirror_type_spherical  
    
    @mirror_type_spherical.setter
    def mirror_type_spherical(self, spherical: bool):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__mirror_type_spherical = spherical
        self.__mirror_type_perfect = not spherical
        self.clear_mem_cache()

    @property
    def mirror_type_perfect(self):
        """ returns True if it is a perfect lens """
        return self.__mirror_type_perfect  
    
    @mirror_type_perfect.setter
    def mirror_type_perfect(self, perfect: bool):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__mirror_type_perfect =  perfect
        self.__mirror_type_spherical = not perfect
        self.clear_mem_cache()
    
    @property
    def symmetric(self):
        if self.mirror_tilted:
            return False
        elif self.left_refl_size_adjust:
            return False
        elif self.right_refl_size_adjust:
            return False
        else:
            return (self.__f_R_L == self.__f_R_R)

    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
        
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space

    @property    
    def k_space_in_dont_care(self):    
        return False

    @property  
    def k_space_out_dont_care(self):    
        return False

    @property
    def f_R_L(self):
        """ 
        focal length of the left side mirror in m 
        f>0: concave mirror (focussing), f<0: convex mirror
        """
        return self.__f_R_L
    
    @property
    def f_R_R(self):
        """ 
        focal length of the right side mirror in m 
        f>0: concave mirror (focussing), f<0: convex mirror
        """
        return self.__f_R_R
    
    @property
    def f_R_R_mm(self):
        """ 
        the focal length of the right side mirror in mm 
        f_mm>0: concave mirror (focussing), f_mm<0: convex mirror
        """
        return self.__f_R_R_mm
    
    @f_R_R.setter
    def f_R_R(self, f: float):
        if f==0:
            print("Error: focal length 0 is not allowed!")
            return
        if not self.cavity is None:
            self.cavity.clear()        
        self.__f_R_R = f
        self.__f_R_R_mm = f * 1000
        self.__mirror_mask_R = None
        self.__lens_mask = None
    
    @f_R_R_mm.setter
    def f_R_R_mm(self, f_mm: float):
        if f_mm==0:
            print("Error: focal length 0 is not allowed!")
            return
        if not self.cavity is None:
            self.cavity.clear()        
        self.__f_R_R_mm = f_mm
        self.__f_R_R = f_mm / 1000
        self.__mirror_mask_R = None
        self.__lens_mask = None
        
    @property
    def f_R_L_mm(self):
        """ 
        the focal length of the left side mirror in mm 
        f_mm>0: concave mirror (focussing), f_mm<0: convex mirror
        """
        return self.__f_R_L_mm
    
    @f_R_L.setter
    def f_R_L(self, f: float):
        if f==0:
            print("Error: focal length 0 is not allowed!")
            return
        if not self.cavity is None:
            self.cavity.clear()        
        self.__f_R_L = f
        self.__f_R_L_mm = f * 1000
        self.__mirror_mask_L = None
        self.__lens_mask = None
    
    @f_R_L_mm.setter
    def f_R_L_mm(self, f_mm: float):
        if f_mm==0:
            print("Error: focal length 0 is not allowed!")
            return
        if not self.cavity is None:
            self.cavity.clear()        
        self.__f_R_L_mm = f_mm
        self.__f_R_L = f_mm / 1000
        self.__mirror_mask_L = None
        self.__lens_mask = None
        
    def is_convex(self, side: Side):
        if side == Side.LEFT:
            if self.f_R_L == math.inf:
                return False
            else:
                return (self.f_R_L<0)
        else:
            if self.f_R_R == math.inf:
                return False
            else:
                return (self.f_R_R<0)
        
    def is_concave(self, side: Side):
        if side == Side.LEFT:
            if self.f_R_L == math.inf:
                return False
            else:
                return (self.f_R_L>0)
        else:
            if self.f_R_R == math.inf:
                return False
            else:
                return (self.f_R_R>0)


    def reflect(self, E_in, k_space_in, k_space_out, side: Side):
        """ 
        reflects the input field E_in on the left or right side 
        if .transm_behaves_like_refl_left or .transm_behaves_like_refl_right
        then transmission acts as reflection, and there is no reflection and
        just zero is returned
        """
        
        if side == Side.LEFT:
            if self.LTR_transm_behaves_like_refl_left:
                # no left reflection in "LTR transmission acts as reflection"-mode
                return 0
            
            elif self.LTR_transm_behaves_like_refl_right:
                # no left reflection in "LTR transmission acts as reflection"-mode
                return 0
            
            elif self.LTR_transm_behaves_neutral:
                # no left reflection in "LTR transmission acts neutral"-mode
                return 0
            
        elif side == Side.RIGHT:
            if self.RTL_transm_behaves_like_refl_left:
                # no right reflection in "RTL transmission acts as reflection"-mode
                return 0
            
            elif self.RTL_transm_behaves_like_refl_right:
                # no right reflection in "RTL transmission acts as reflection"-mode
                return 0
            
            elif self.RTL_transm_behaves_neutral:
                # no right reflection in "RTL transmission acts neutral"-mode
                return 0        
        
        # normal reflection behavior
        ax = self.grid.axis_tot
        if k_space_in:
            # input in k-space: Convert to position-space
            E_in = ifft2_phys_spatial(E_in, ax)
            
        if side == Side.LEFT:
            # reflection on the left side
            E_in *= self.r_L
                
            # apply curved mirror mask left
            if self.__f_R_L != math.inf:
                E_in *= self.mirror_mask_L 
                
            # apply tilt masks left
            if self.rot_around_y != 0:
                E_in *= self.tilt_mask_x_L 
            if self.rot_around_x != 0:
                E_in *= self.tilt_mask_y_L   
                
            # if activated, change size of outgoing light field in y-direction 
            # according to mirror tilt and incident angle
            if self.project_according_to_angles:
                proj_factor = self.get_projection_factor_y2(Side.LEFT)
                if  proj_factor != 1:
                    E_in = self.grid.stretch_y(E_in, 0, proj_factor)
                
        elif side == Side.RIGHT:
            # reflection on the right side
            E_in *= self.r_R
            
            # apply curved mirror mask right
            if self.__f_R_R != math.inf:
                E_in *= self.mirror_mask_R 
                
            # apply tilt masks right
            if self.rot_around_y != 0:
                E_in *= self.tilt_mask_x_R 
            if self.rot_around_x != 0:
                E_in *= self.tilt_mask_y_R 
                
            # if activated, change size of outgoing light field in y-direction 
            # according to mirror tilt and incident angle
            if self.project_according_to_angles:
                proj_factor = self.get_projection_factor_y2(Side.RIGHT)
                if  proj_factor != 1:
                    E_in = self.grid.stretch_y(E_in, 0, proj_factor)
        
        if k_space_out:
            # output in k-space: Convert to k-space
            E_in = fft2_phys_spatial(E_in, ax)               

        return E_in

    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        """
        Propagates the input field E_in in positive z-direction (LTR) 
        or negative z-Direction (RTL) by simply applying the lens mask and the
        t_LTR or t_RTL transmission factor (for "normal" mirror behavior)
        if .transm_behaves_like_refl_left or .transm_behaves_like_refl_right
        is True, then transmission imitates the reflection behavior
        """
        ax = self.grid.axis_tot
        
        if (direction == Dir.LTR and self.LTR_transm_behaves_neutral) or \
            (direction == Dir.RTL and self.RTL_transm_behaves_neutral):
            # Transmission in this direction behaves neutral (does nothing)
            currently_in_k_space = k_space_in
            
        else:  
            if k_space_in:
                # input in k-space: Convert to position-space
                E_in = ifft2_phys_spatial(E_in, ax)
            
            currently_in_k_space = False
                
            if (direction == Dir.LTR and self.LTR_transm_behaves_like_refl_left) or \
                (direction == Dir.RTL and self.RTL_transm_behaves_like_refl_left):
                # Transmission behaves like reflection from left side
                E_in *= self.r_L
                    
                # apply curved mirror mask left
                if self.__f_R_L != math.inf:
                    E_in *= self.mirror_mask_L 
                    
                # apply tilt masks left
                if self.rot_around_y != 0:
                    E_in *= self.tilt_mask_x_L 
                if self.rot_around_x != 0:
                    E_in *= self.tilt_mask_y_L  
                
                # if activated, change size of outgoing light field in y-direction 
                # according to mirror tilt and incident angle
                if self.project_according_to_angles:
                    proj_factor = self.get_projection_factor_y2(Side.LEFT)
                    if  proj_factor != 1:
                        E_in = self.grid.stretch_y(E_in, 0, proj_factor)
            
            elif (direction == Dir.LTR and self.LTR_transm_behaves_like_refl_right) or \
                (direction == Dir.RTL and self.RTL_transm_behaves_like_refl_right):
                # Transmission behaves like reflection from right side
                E_in *= self.r_R
            
                # apply curved mirror mask right
                if self.__f_R_R != math.inf:
                    E_in *= self.mirror_mask_R 
                    
                # apply tilt masks right
                if self.rot_around_y != 0:
                    E_in *= self.tilt_mask_x_R 
                if self.rot_around_x != 0:
                    E_in *= self.tilt_mask_y_R  
                    
                # if activated, change size of outgoing light field in y-direction 
                # according to mirror tilt and incident angle
                if self.project_according_to_angles:
                    proj_factor = self.get_projection_factor_y2(Side.RIGHT)
                    if  proj_factor != 1:
                        E_in = self.grid.stretch_y(E_in, 0, proj_factor)
            
            else:
                # normal transmission (never affected by rotation)
                
                # apply lens-mask
                E_in *= self.lens_mask
                
                # transmission factor
                if direction == Dir.LTR:
                    E_in *= self.t_LTR
                else:
                    E_in *= self.t_RTL
        
        if currently_in_k_space:
             # result is currently in k-space
             if not k_space_out:
                 # but output should be in position-space
                 E_in = ifft2_phys_spatial(E_in, self.grid.axis_tot)
        
        else:
             # result is currently in position-space
             if k_space_out:
                 # but output should be in k-space
                 E_in = fft2_phys_spatial(E_in, self.grid.axis_tot)     
        
        return E_in
    
    def _calc_R_L_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        psi *= self.r_L
        
        if self.left_side_relevant:
            if self.__f_R_L != math.inf:
                psi *= self.__mirror_mask_L 
            if self.rot_around_y != 0:
                psi *= self._tilt_mask_x_L 
            if self.rot_around_x != 0:
                psi *= self._tilt_mask_y_L 
            if self._proj_factor_y2_L != 1:
                psi = self.grid.stretch_y(psi, 0, self._proj_factor_y2_L)
                
        return self.grid.arr_to_vec(psi, False, False) 
    
    def calc_R_L_mat_tot(self):
        """
        Calulates the left reflection matrix 
        """
        # simulate propagation for all modes in parallel   
        if self.left_side_relevant:
            self.cavity.progress.push_print("calculating left reflection matrix R_L")
            if self.LTR_transm_behaves_like_refl_left and self.RTL_transm_behaves_like_refl_left:
                self.cavity.progress.print("(which governs LTR and RTL transmission behavior)")
            elif self.LTR_transm_behaves_like_refl_left:
                self.cavity.progress.print("(which governs LTR transmission behavior)")
            elif self.RTL_transm_behaves_like_refl_left:
                self.cavity.progress.print("(which governs RTL transmission behavior)")
        
        if not self.left_side_relevant: 
            # left side not relevant: assume flat, untilted mirror
            self._RT_matrices.set_block(0, 0, self.r_L)
            self._R_L_mat_available = True
            
        elif self.__f_R_L == math.inf and not (self.mirror_tilted 
                                               or self.left_refl_size_adjust):
            # flat surface on left side and no mirror tilt or reflection size adjustment: 
            # Flat mirror
            self._RT_matrices.set_block(0, 0, self.r_L)
            self._R_L_mat_available = True
            
        else:
            # either curved suface, or tilted mirror, or reflection size adjustment
            
            # calculate mirror mask, if neccessary
            if self.__f_R_L != math.inf:
                if self.__mirror_mask_L is None :
                    self.__calc_mirror_mask(Side.LEFT)
                    
            # calculate tilt masks, if neccessary
            if self._tilt_mask_x_L is None or self._tilt_mask_y_L is None:
                if self.rot_around_y != 0:
                    self.calc_tilt_masks_x()
                if self.rot_around_x != 0:
                    self.calc_tilt_masks_y()
                    
            NN = self.grid.res_tot**2
            
            # this will be used by _calc_R_L_mat_tot_helper
            if self.left_refl_size_adjust:
                self._proj_factor_y2_L = self.get_projection_factor_y2(Side.LEFT)
            else:
                self._proj_factor_y2_L = 1
            
            if self.par_RT_calc:
                #########################################
                # Parallel version
                #########################################
                keep_clsBlockMatrix_alive(True)
                result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_R_L_mat_tot_helper)(i) for i in range(NN))
                keep_clsBlockMatrix_alive(False)
                R_L_mat_tot = np.column_stack(result)
            
            else:
                ##########################################
                # non-parallel version
                ##########################################
                R_L_mat_tot = np.empty((NN,NN), dtype=self.cavity.dtype_c)        
                i = 0
                self.cavity.progress.tic_reset(NN, True, "processing...")
                
                for nx, ny in self.grid.mode_numbers_tot:            
                    psi = self.grid.fourier_basis_func(nx, ny, True, False)
                    psi *= self.r_L
                    if self.left_side_relevant:
                        if self.__f_R_L != math.inf:
                            psi *= self.__mirror_mask_L 
                        if self.rot_around_y != 0:
                            psi *= self._tilt_mask_x_L 
                        if self.rot_around_x != 0:
                            psi *= self._tilt_mask_y_L 
                        if self._proj_factor_y2_L != 1:
                            psi = self.grid.stretch_y(psi, 0, self._proj_factor_y2_L)
                    R_L_mat_tot[:,i] = self.grid.arr_to_vec(psi, False, False)
                    i += 1
                    self.cavity.progress.tic()
        
            self._RT_matrices.set_block(0, 0, R_L_mat_tot)
            self._R_L_mat_available = True
            R_L_mat_tot = None
        
        gc.collect()
        if self.left_side_relevant: 
            self.cavity.progress.pop()
    
    
    @property    
    def R_L_mat_tot(self):
        """ returns the right reflection matrix (total resolution) """
        if self.LTR_transm_behaves_like_refl_left:
            # LTR transmission should behave like reflection from left
            # this means that there is no reflection on the left side
            return 0
        
        elif self.LTR_transm_behaves_like_refl_right: 
            # LTR transmission should behave like reflection from right
            # this means, that there is no reflection on the left side
            return 0
        
        elif self.LTR_transm_behaves_neutral: 
            # LTR transmission should behave neutral
            # this means, that there is no reflection on the left side
            return 0
        
        else:
            return self._R_L_mat_tot_helper()
    
    def _R_L_mat_tot_helper(self):    
        """ 
        returns the left reflection matrix (total resolution)
        NOT CONSIDERING .__transm_behaves_like_refl_left and
        .__transm_behaves_like_refl_right
        """
        
        if not self.left_side_relevant:
            # left side not relevant: Assume Flat mirror
            return self.r_L
        
        elif self.__f_R_L == math.inf and not (self.mirror_tilted 
                                               or self.left_refl_size_adjust):
            # flat surface on left side and no tilt, no size adjustment
            # assume Flat mirror
            return self.r_L
        
        else:
            if not self._R_L_mat_available:
                self.calc_R_L_mat_tot()
            return self._RT_matrices.get_block(0, 0)
        
    def _calc_R_R_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        psi *= self.r_R
        
        if self.right_side_relevant:
            if self.__f_R_R != math.inf:
                psi *= self.__mirror_mask_R 
            if self.rot_around_y != 0:
                psi *= self._tilt_mask_x_R
            if self.rot_around_x != 0:
                psi *= self._tilt_mask_y_R 
            if self._proj_factor_y2_L != 1:
                psi = self.grid.stretch_y(psi, 0, self._proj_factor_y2_R)
                
        return self.grid.arr_to_vec(psi, False, False) 
    
    def calc_R_R_mat_tot(self):
        """
        Calulates the right reflection matrix 
        """
        # simulate propagation for all modes in parallel    
        if self.right_side_relevant:
            self.cavity.progress.push_print("calculating right reflection matrix R_R")
            if self.LTR_transm_behaves_like_refl_right and self.RTL_transm_behaves_like_refl_right:
                self.cavity.progress.print("(which governs LTR and RTL transmission behavior)")
            elif self.LTR_transm_behaves_like_refl_right:
                self.cavity.progress.print("(which governs LTR transmission behavior)")
            elif self.RTL_transm_behaves_like_refl_right:
                self.cavity.progress.print("(which governs RTL transmission behavior)")
        
        if not self.right_side_relevant: 
            # Right side not relevant: assume flat, untilted mirror
            self._RT_matrices.set_block(1, 1, self.r_R)
            self._R_R_mat_available = True
        
        elif self.__f_R_R == math.inf and not (self.mirror_tilted 
                                               or self.right_refl_size_adjust):
            # flat surface on right side and no mirror tilt or output field strech
            # assume Flat untilted mirror
            self._RT_matrices.set_block(1, 1, self.r_R)
            self._R_R_mat_available = True
            
        else:
            # either curved suface, or tilted mirror.
            
            # calculate mirror mask, if neccessary
            if self.__f_R_R != math.inf:
                if self.__mirror_mask_R is None :
                    self.__calc_mirror_mask(Side.RIGHT)
                    
            # calculate tilt masks, if neccessary
            if self._tilt_mask_x_R is None or self._tilt_mask_y_R is None:
                if self.rot_around_y != 0:
                    self.calc_tilt_masks_x()
                if self.rot_around_x != 0:
                    self.calc_tilt_masks_y()
            
            NN = self.grid.res_tot**2
            # this will be used by _calc_R_R_mat_tot_helper
            if self.right_refl_size_adjust:
                self._proj_factor_y2_R = self.get_projection_factor_y2(Side.RIGHT)
            else:
                self._proj_factor_y2_R = 1
            
            if self.par_RT_calc:
                #########################################
                # Parallel version
                #########################################
                keep_clsBlockMatrix_alive(True)
                result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_R_R_mat_tot_helper)(i) for i in range(NN))
                keep_clsBlockMatrix_alive(False)
                R_R_mat_tot = np.column_stack(result)
            
            else:
                ##########################################
                # non-parallel version
                ##########################################
                R_R_mat_tot = np.empty((NN,NN), dtype=self.cavity.dtype_c)        
                i = 0
                self.cavity.progress.tic_reset(NN, True, "processing...")
                for nx, ny in self.grid.mode_numbers_tot:            
                    psi = self.grid.fourier_basis_func(nx, ny, True, False)
                    psi *= self.r_R
                    if self.right_side_relevant:
                        if self.__f_R_R != math.inf:
                            psi *= self.__mirror_mask_R 
                        if self.rot_around_y != 0:
                            psi *= self._tilt_mask_x_R 
                        if self.rot_around_x != 0:
                            psi *= self._tilt_mask_y_R 
                        if self._proj_factor_y2_R != 1:
                            psi = self.grid.stretch_y(psi, 0, self._proj_factor_y2_R)
                    R_R_mat_tot[:,i] = self.grid.arr_to_vec(psi, False, False)
                    i += 1
                    self.cavity.progress.tic()
            
            self._RT_matrices.set_block(1, 1, R_R_mat_tot)
            self._R_R_mat_available = True
            R_R_mat_tot = None
            
        gc.collect()
        if self.right_side_relevant: 
            self.cavity.progress.pop()
 
    
    @property    
    def R_R_mat_tot(self):
        """ returns the right reflection matrix (total resolution)"""
        if self.RTL_transm_behaves_like_refl_left:
            # RTL transmission should behave like reflection from left
            # this means that there is no reflection on the right side
            return 0
        
        elif self.RTL_transm_behaves_like_refl_right: 
            # RTL transmission should behave like reflection from right
            # this means, that there is no reflection on the right side
            return 0
        
        elif self.RTL_transm_behaves_neutral: 
            # RTL transmission should behave neutral
            # this means, that there is no reflection on the right side
            return 0
        
        else:
            return self._R_R_mat_tot_helper()
    
    def _R_R_mat_tot_helper(self):    
        """ 
        returns the right reflection matrix (total resolution)
        NOT CONSIDERING .__transm_behaves_like_refl_left and
        .__transm_behaves_like_refl_right
        """
        if not self.right_side_relevant:
            # right side not relevant: Assume Flat mirror
            return self.r_R
        
        elif self.__f_R_R == math.inf and not (self.mirror_tilted 
                                               or self.right_refl_size_adjust):
            # flat surface on right side and no tilt, no size adjustment
            # assume Flat mirror
            return self.r_R
        
        else:
            if not self._R_R_mat_available:
                self.calc_R_R_mat_tot()
            return self._RT_matrices.get_block(1, 1)
    
    def _calc_T_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        psi *= self.__t
        if self.left_side_relevant and self.right_side_relevant:
            psi *= self.__lens_mask 
        return self.grid.arr_to_vec(psi, False, False) 
    
    def calc_T_mat_tot(self):
        """
        Calulates the transnmission matrix 
        """
        # simulate propagation for all modes in parallel      
        if self.left_side_relevant and self.right_side_relevant:
            self.cavity.progress.push_print("calculating transmission matrix T")
        
        if self.f_T == math.inf:
            # infinite focal length does nothing
            self._RT_matrices.set_block(0, 1, self.__t)
            self._T_mat_available = True
        elif not self.left_side_relevant:
            # left side not relevant -> don't care about transmission
            self._RT_matrices.set_block(0, 1, self.__t)
            self._T_mat_available = True
        elif not self.right_side_relevant:
            # right side not relevant -> don't care about transmission
            self._RT_matrices.set_block(0, 1, self.__t)
            self._T_mat_available = True
        else:
            if self.__lens_mask is None :
                self.__calc_lens_mask()
            NN = self.grid.res_tot**2
            
            if self.par_RT_calc:
                #########################################
                # Parallel version
                #########################################
                keep_clsBlockMatrix_alive(True)
                result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_T_mat_tot_helper)(i) for i in range(NN))
                keep_clsBlockMatrix_alive(False)
                T_mat_tot = np.column_stack(result)
            
            else:
                ##########################################
                # non-parallel version
                ##########################################
                T_mat_tot = np.empty((NN,NN), dtype=self.cavity.dtype_c)        
                i = 0
                self.cavity.progress.tic_reset(NN, True, "processing...")
                for nx, ny in self.grid.mode_numbers_tot:            
                    psi = self.grid.fourier_basis_func(nx, ny, True, False)
                    psi *= self.__t
                    if self.left_side_relevant and self.right_side_relevant:
                        psi *= self.__lens_mask 
                    T_mat_tot[:,i] = self.grid.arr_to_vec(psi, False, False)
                    i += 1
                    self.cavity.progress.tic()
                
            self._RT_matrices.set_block(0, 1, T_mat_tot)
            self._T_mat_available = True
            T_mat_tot = None
            
        gc.collect()
        if self.left_side_relevant and self.right_side_relevant: 
            self.cavity.progress.pop()

    @property  
    def T_LTR_mat_tot(self):    
        """ returns the left-to-right transmission matrix (total resolution)"""
        if self.LTR_transm_behaves_like_refl_left:
            # Transmission should behave like reflection from left
            return self._R_L_mat_tot_helper()
         
        elif self.LTR_transm_behaves_like_refl_right: 
            # Transmission should behave like reflection from right
            return self._R_R_mat_tot_helper()
         
        elif self.LTR_transm_behaves_neutral: 
            # LTR transmission should do nothing
            return 1
         
        else:
            # regular transmission behavior
            if self.f_T == math.inf:
                # no lensing
                return self.__t
            elif not self.right_side_relevant:
                # right side not relevant -> don't care about exact transmission behavior
                return self.__t
            else:
                if not self._T_mat_available:
                    self.calc_T_mat_tot()
                return self._RT_matrices.get_block(0, 1)
    
    @property
    def T_RTL_mat_tot(self):    
        """ returns the right-to-left transmission matrix (total resolution)"""
        if self.RTL_transm_behaves_like_refl_left:
             # Transmission should behave like reflection from left
             return self._R_L_mat_tot_helper()
         
        elif self.RTL_transm_behaves_like_refl_right: 
             # Transmission should behave like reflection from right
             return self._R_R_mat_tot_helper()
        
        elif self.RTL_transm_behaves_neutral: 
            # RTL transmission should do nothing
            return 1    
        
        else:
            # regular transmission behavior
            if self.f_T == math.inf:
                # no lensing
                return self.__t
            elif not self.left_side_relevant:
                # left side not relevant -> don't care about exact transmission behavior
                return self.__t
            else:
                if not self._T_mat_available:
                    self.calc_T_mat_tot()
                return self._RT_matrices.get_block(0, 1)


###############################################################################
# clsGrating
# represents a thin grating (periodic absoption or phase changes)
###############################################################################        
class clsGrating(clsOptComponent2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__grate_mask = None
        self.__T_mat_tot = None
        self.__absorbtion_grating = True
        self.__phase_grating = False
        self.__dx = math.inf
        self.__dy = math.inf
        self.__x_max = math.inf
        self.__y_max = math.inf
        self.__min_val = 0
        self.__max_val = 1
    
    @property
    def dx(self):
        return self.__dx 
    
    @property
    def dy(self):
        return self.__dy 

    @property
    def x_max(self):
        return self.__x_max 
    
    @property
    def y_max(self):
        return self._y_max 
    
    @property
    def min_val(self):
        return self.__min_val 
    
    @property
    def max_val(self):
        return self.__max_val 
    
    @property
    def absorbtion_grating(self):
        """ 
        If true, the grating is an absorption grating 
        (alternatively absorbing / not absorbing)
         """
        return self.__absorbtion_grating
    
    @absorbtion_grating.setter
    def absorbtion_grating(self, x: bool):
        if not self.cavity is None:
            self.cavity.clear()        
        if self.__absorbtion_grating != x:
            self.__grate_mask = None
            self.__T_mat_tot = None
        self.__absorbtion_grating = x
        self.__phase_grating = not x
    
    @property
    def phase_grating(self):
        """ 
        If true, the grating is an phase grating 
        (peridically modulating the phase )
         """
        return self.__phase_grating
    
    @phase_grating.setter
    def phase_grating(self, x: bool):
        if not self.cavity is None:
            self.cavity.clear()        
        if self.__phase_grating != x:
            self.__grate_mask = None
            self.__T_mat_tot = None
        self.__phase_grating =  x
        self.__absorbtion_grating = not x
    
    @property
    def grate_mask(self):
        if self.__grate_mask is None:
            self.__calc_grating_mask()
        return self.__grate_mask
    
    @property
    def symmetric(self):
        return True
    
    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
        
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space

    @property    
    def k_space_in_dont_care(self):    
        return False

    @property  
    def k_space_out_dont_care(self):    
        return False

    def set_cos_grating(self, dx, x_max, dy, y_max, opt_dim: bool, min_val, max_val):
        """
        dx: period in x-direction  (math.inf for no grating in x-direction)
        x_max: maximum size in +-x-direction (math.inf for full size)
        dy: period in y-direction  (math.inf for no grating in y-direction)
        y_max: maximum size in +-y-direction (math.inf for full size)
        opt_dim: optimize dimension to fit grid
        min_val: minimum value in grating
        max_val: maximum value in grating
        returns (optimized) values dx, x_max, dy, y_max
        """
        if not self.cavity is None:
            self.cavity.clear()        
        if min_val > max_val:  
            min_val, max_val = max_val, min_val
        
        if opt_dim:
            ax = self.grid.axis_tot
            x, y = np.meshgrid(ax, ax)
            if dx != math.inf:
                dx_new = find_best_match(ax, dx)
                x_max *= dx_new / dx
                dx = dx_new
            if dy != math.inf:
                dy_new = find_best_match(ax, dy)
                y_max *= dy_new / dy
                dy = dy_new
            
        self.__dx = dx
        self.__dy = dy
        self.__x_max = x_max
        self.__y_max = y_max
        self.__min_val = min_val
        self.__max_val = max_val
                
        return dx, x_max, dy, y_max
    
    def __calc_grating_mask(self):
        ax = self.grid.axis_tot
        x, y = np.meshgrid(ax, ax)
        
        if self.__dx == math.inf and self.__dy == math.inf:
            # no grid
            self.__grate_mask = 1
        
        elif self.__dx == math.inf:
            # grid only in y-direction
            self.__grate_mask = self.__min_val + (self.__max_val-self.__min_val) * \
            (1-np.cos(2*np.pi*y/self.__dy))/2 * \
            (np.abs(y) <= self.__y_max).astype(float)
        
        elif self.__dy == math.inf:
            # grid only in x-direction
            self.__grate_mask = self.__min_val + (self.__max_val-self.__min_val) * \
            (1-np.cos(2*np.pi*x/self.__dx))/2 * \
            (np.abs(x) <= self.__x_max).astype(float)
        else:
            # grid in x and y direction
            self.__grate_mask = self.__min_val + (self.__max_val-self.__min_val) * \
                (1-np.cos(2*np.pi*x/self.__dx))/2 * \
                (1-np.cos(2*np.pi*y/self.__dy))/2 * \
                (np.abs(x) <= self.__x_max).astype(float) * \
                (np.abs(y) <= self.__y_max).astype(float)
    
        if self.__phase_grating:
            if is_matrix(self.__grate_mask):
                self.__grate_mask = np.exp(1j*self.__grate_mask)
    
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        """
        Propagates the input field E_in in positive z-direction (LTR) 
        or negative z-Direction (RTL)
        """
        if self.__grate_mask is None:
            self.__calc_grating_mask()
        
        ax = self.grid.axis_tot
        if k_space_in:
            # input in k-space: Convert to position-space
            E_in = ifft2_phys_spatial(E_in, ax)
        
        # apply lens-mask
        if self.__absorbtion_grating:
            E_in *= self.__grate_mask
        else:
            E_in *= self.__grate_mask
        
        if k_space_out:
            # output in k-space: Convert to k-space
            E_in = fft2_phys_spatial(E_in, ax)
        
        return E_in
    
    @property  
    def R_L_mat_tot(self):    
        """ returns the left reflection matrix (total resolution)"""
        return 0
    
    @property
    def R_R_mat_tot(self):    
        """ returns the right reflection matrix (total resolution)"""
        return 0
    
    @property  
    def T_LTR_mat_tot(self):    
        """ returns the left-to-right transmission matrix (total resolution)"""
        if self.__T_mat_tot is None:
            self.calc_T_mat_tot()
        return self.__T_mat_tot
    
    @property
    def T_RTL_mat_tot(self):    
        """ returns the right-to-left transmission matrix (total resolution)"""
        if self.__T_mat_tot is None:
            self.calc_T_mat_tot()
        return self.__T_mat_tot

    def clear_mem_cache(self):   
        super().clear_mem_cache()
        if not self.__grate_mask is None:            
            self.__grate_mask = None
            self.cavity.progress.push_print(
                "deleting grating mask of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if not self.__T_mat_tot is None:
            self.__T_mat_tot = None
            self.cavity.progress.push_print(
                "deleting transmission matrix T of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()

    def _calc_T_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        psi *= self.grate_mask
        return self.grid.arr_to_vec(psi, False, False)  

    def calc_T_mat_tot(self):
        """
        Creates a transmission matrix for propagation through grating 
        (total resolution)
        """        
        # simulate propagation for all modes in parralell        
        self.cavity.progress.push_print("calculating transmission matrix T")
        NN = self.grid.res_tot**2
        
        keep_clsBlockMatrix_alive(True)
        result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_T_mat_tot_helper)(i) for i in range(NN))
        keep_clsBlockMatrix_alive(False)
        self.__T_mat_tot = np.column_stack(result)
        
        
        self.cavity.progress.pop()    
        
        
###############################################################################
# clsSoftAperture
# thin soft-border aperture
###############################################################################
class clsSoftAperture(clsOptComponent2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__aperture = 0 # aperture in m (0 means no aperture)
        self.__aperture_mm = 0 # aperture in mm (0 means no aperture)
        self.__black_value = 0
        self.__aperture_mask = None
        self.__transition_width = 0
        self.__T_mat_tot = None
        
    @property
    def symmetric(self):
        return True
    
    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
    
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space
    
    @property    
    def k_space_in_dont_care(self):    
        return False

    @property  
    def k_space_out_dont_care(self):    
        return False
    
    def set_params(self, aperture, transition_width, black_value):
        if not self.cavity is None:
            self.cavity.clear()        
        if aperture<0:
            aperture = 0
        if transition_width <0:
            transition_width = 0
        
        self.__aperture = aperture
        self.__aperture_mm = aperture * 1000
        self.__transition_width = transition_width
        self.black_value = black_value
        self.clear_mem_cache()
    
    @property
    def black_value(self):
        """ returns the aperture in m """
        return self.__black_value
    
    @black_value.setter
    def black_value(self, x):
        if not self.cavity is None:
            self.cavity.clear()        
        if x<0:
            x = 0
        if x>1:
            x = 1
        self.__black_value = x
        self.clear_mem_cache()
    
    @property
    def transition_width(self):
        """ returns the aperture in m """
        return self.__transition_width
    
    @transition_width.setter
    def transition_width(self, x: float):
        if not self.cavity is None:
            self.cavity.clear()        
        if x < 0:
            x = 0
        self.__transition_width = x
        self.clear_mem_cache()
    
    @property
    def aperture(self):
        """ returns the aperture in m """
        return self.__aperture
    
    @property
    def aperture_mm(self):
        """ returns the aperture in mm """
        return self.__aperture_mm
    
    @aperture.setter
    def aperture(self, a: float):
        if not self.cavity is None:
            self.cavity.clear()        
        if a < 0:
            a = 0
        self.__aperture = a
        self.__aperture_mm = a * 1000
        self.clear_mem_cache()
        
    @aperture_mm.setter
    def aperture_mm(self, a_mm: float):
        if not self.cavity is None:
            self.cavity.clear()        
        if a_mm < 0:
            a_mm = 0
        self.__aperture_mm = a_mm
        self.__aperture = a_mm / 1000
        self.clear_mem_cache()
    
    @property
    def aperture_mask(self):
        """ returns the lens-mask """
        if self.__aperture_mask is None:
            self.__calc_aperture_mask()
        return self.__aperture_mask
    
    def __calc_aperture_mask(self):
        # add aperture, if required
        if self.__aperture>0:
            
            if self.__transition_width == 0:
                self.__aperture_mask = self.grid.get_aperture_mask(
                    self.__aperture, 4, self.__black_value)
            else:
                self.__aperture_mask = self.grid.get_soft_aperture_mask(
                    self.__aperture, self.__transition_width/2, 
                    self.__black_value)
        else:
            self.__aperture_mask = 1
    
    def _calc_T_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        psi *= self.aperture_mask
        return self.grid.arr_to_vec(psi, False, False)  
    
    def calc_T_mat_tot(self):
        """
        Creates a transmission matrix for propagation through lens 
        (total resolution)
        """        
        # simulate propagation for all modes in parralell        
        self.cavity.progress.push_print("calculating transmission matrix T")
        NN = self.grid.res_tot**2
        
        keep_clsBlockMatrix_alive(True)
        result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_T_mat_tot_helper)(i) for i in range(NN))
        keep_clsBlockMatrix_alive(False)
        self.__T_mat_tot = np.column_stack(result)
               
        self.cavity.progress.pop()
       
    @property 
    def T_LTR_mat_tot(self):    
        """ returns the left-to-right transmission matrix (total resolution)"""
        if self.__T_mat_tot is None:
            self.calc_T_mat_tot()
        return self.__T_mat_tot
    
    @property    
    def T_RTL_mat_tot(self):    
        """ returns the right-to-left transmission matrix (total resolution)"""
        if self.__T_mat_tot is None:
            self.calc_T_mat_tot()
        return self.__T_mat_tot
    
    @property  
    def R_L_mat_tot(self):    
        """ returns the left reflection matrix (total resolution)"""
        return 0
    
    @property
    def R_R_mat_tot(self):    
        """ returns the right reflection matrix (total resolution)"""
        return 0
    
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        """
        Propagates the input field E_in in positive z-direction (LTR) 
        or negative z-Direction (RTL)
        """
        ax = self.grid.axis_tot
        if k_space_in:
            # input in k-space: Convert to position-space
            E_in = ifft2_phys_spatial(E_in, ax)
        
        # apply lens-mask
        E_in *= self.aperture_mask
        
        if k_space_out:
            # output in k-space: Convert to k-space
            E_in = fft2_phys_spatial(E_in, ax)
        
        return E_in
    
    def clear_mem_cache(self):   
        super().clear_mem_cache()
        if not self.__aperture_mask is None:            
            self.__aperture_mask = None
            self.cavity.progress.push_print(
                "deleting aperture phase mask of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if not self.__T_mat_tot is None:
            self.__T_mat_tot = None
            self.cavity.progress.push_print(
                "deleting transmission matrix T of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        gc.collect()
        
###############################################################################
# clsThinLens
# represents a thin lens
############################################################################### 
class clsThinLens(clsOptComponent2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__lens_type_spherical = True
        self.__lens_type_perfect = False
        self.__f = 0.05 # focal length in m
        self.__f_mm = 50 # focal length in mm
        self.__aperture = 0 # aperture in m (0 means no aperture)
        self.__aperture_black_value = 0
        self.__aperture_mm = 0 # aperture in mm (0 means no aperture)
        self.__aperture_anti_alias = True
        self.__lens_mask = None
        
        self.__RT_matrices = None
        self._R_mat_available = False
        self._T_mat_available = False
        self.__reflection_mask = None
        self.__reflection_mask1 = None
        self.__reflection_mask2 = None
        self.__reflection_aperture = 0
        self.__reflection_aperture_epsilon = 0
        self.__reflection_aperture_black = 0
        self.__R_residual = 0
        self.__T_residual = 1
        self.__r_residual = 0
        self.__t_residual = 1
        self.__sym_phase = True
        self.par_RT_calc = True # parallel calcuation of R and T matrices 
            
    @property
    def _RT_matrices(self):
        """ 
        0,0 ... R_mat_tot
        0,1 ... T_mat_tot
        """
        
        delete_old_instance = False
        create_new_instance = False
        if self.__RT_matrices is None:
            # no current instance: create a new instance
            create_new_instance = True
            
        else:
            # there is already an instance
            if self.cavity.use_swap_files_in_bmatrix_class:
                # it should be set to file caching
                if not self.__RT_matrices.file_caching:
                    # but it is not!
                    delete_old_instance = True
                    create_new_instance = True
                
                elif self.__RT_matrices.tmp_dir != self.cavity.tmp_folder:
                    # the current instance is set to file caching (as it should)
                    # but it refers to a wrong tmp folder!
                    delete_old_instance = True
                    create_new_instance = True
                
            else:
                # the current instance should *not* be set to file caching
                if self.__RT_matrices.file_caching:
                    # but it is!
                    delete_old_instance = True
                    create_new_instance = True
            
        if delete_old_instance:
            self._R_mat_available = False
            self._T_mat_available = False
            self.__RT_matrices = None
            gc.collect()
        
        if create_new_instance:
            self.__RT_matrices = clsBlockMatrix(2, 
                                               self.cavity.use_swap_files_in_bmatrix_class,
                                               self.cavity.tmp_folder)
        
        return self.__RT_matrices
    
    def set_residual_reflection(self, R, reflection_aperture, 
                               epsilon, black_value):
        """
        sets residual reflection values:
        R ... residual refletivity on each facet
        reflection_aperture .. aperture for fading out the reflection to zero
        epsion:
        - for any radius < (aperture/2 - epsilon) the aperture value is 1
        - for any radius > (aperture/2 + epsilon) the aperture value is black_value
        - for any radius between aperture/2 - epsilon and aperture/2 + epsilon
          the value is between black_value and 1
        black_value: Value near zero 
        """
        if not self.cavity is None:
            self.cavity.clear()        
        if reflection_aperture<0:
            reflection_aperture = 0
        if black_value<0:
            black_value = 0
        if black_value>1:
            black_value = 1
        self.__R_residual = R
        self.__T_residual = 1 - R
        self.__set_rt()
        self.__reflection_aperture = reflection_aperture
        self.__reflection_aperture_black = black_value
        self.__reflection_aperture_epsilon = epsilon
        self.clear_mem_cache()
    
    def __set_rt(self):
        if self.__sym_phase:
            self.__r_residual = -self.__R_residual - 1j * math.sqrt(
                self.__R_residual) * math.sqrt(1 - self.__R_residual)
            #self.__t = 1 + self.__r
            R = 1-self.__T_residual
            self.__t_residual = 1-R - 1j * math.sqrt(R) * math.sqrt(1 - R)
        else:
            self.__r_residual = math.sqrt(self.__R_residual)
            self.__t_residual = math.sqrt(self.__T_residual)

        self.clear_mem_cache()
    
    def set_phys_behavior(self, sym_phase: bool):
        """ 
        If sym_phase == True (default), then the mirror behaves
        physically insofar, as if it has the same reflection coefficient 
        on the left and right side, both will get the same phase. 
        If sym_phase == False, the mirror follows the convention that r_L, t_LTR 
        and t_RTL are real positive numbers, and r_R is a real negative number 
        """
        if not self.cavity is None:
            self.cavity.clear()        
        self.__sym_phase = sym_phase
        self.__set_rt()
    
    @property
    def R_residual(self):
        return self.__R_residual  
    
    @property
    def r_residual(self):
        return self.__r_residual
    
    @property
    def t_residual(self):
        return self.__t_residual
    
    @property
    def reflection_aperture(self):
        return self.__reflection_aperture 
    
    @property
    def reflection_aperture_black(self):
        return self.__reflection_aperture_black
    
    @property
    def reflection_aperture_epsilon(self):
        return self.__reflection_aperture_epsilon
    
    @property
    def symmetric(self):
        return True
    
    @property
    def sym_phase(self):
        return self.__sym_phase
    
    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
        
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space

    @property    
    def k_space_in_dont_care(self):    
        return False

    @property  
    def k_space_out_dont_care(self):    
        return False

    def calc_inv_M_bmat_tot(self):
        """ 
        calculates the inverse transfer block matrix of the lens in a fast way 
        """
        M = self.M_bmat_tot
        if self.__aperture <= 0 and self.__R_residual == 0:     
            if self.cavity.use_bmatrix_class:
                M_inv = clsBlockMatrix(2, self.cavity.use_swap_files_in_bmatrix_class,
                                   self.cavity.tmp_folder)
                M_inv.set_block(0,0, M.get_block(1,1))
                M_inv.set_block(1,1, M.get_block(0,0))
                return M_inv
                
            else:
                return [[M[1][1],0],[0,M[0][0]]] 
        else:
            return bmat2_inv(M, self.cavity.progress, "inverting M")
        
    def _calc_T_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        psi *= self.__lens_mask
        return self.grid.arr_to_vec(psi, False, False)  
    
    def _calc_R_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        psi *= self.__reflection_mask
        return self.grid.arr_to_vec(psi, False, False)  
    
    def _calc_R1_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        psi *= self.__reflection_mask1
        return self.grid.arr_to_vec(psi, False, False)  
    
    def _calc_R2_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        psi *= self.__reflection_mask2
        return self.grid.arr_to_vec(psi, False, False)  
    
    def calc_R_mat_tot(self):
        """
        Creates a reflection matrix for residual reflections on lens facets 
        (total resolution)
        """ 
                
        self.cavity.progress.push_print("calculating reflection matrix R")
        NN = self.grid.res_tot**2
        _ = self.reflection_mask
                       
        if not self.par_RT_calc:
            #########################################
            # non-parallel version
            #########################################
            R_mat_tot = np.empty((NN,NN), dtype=self.cavity.dtype_c)        
            i = 0
            self.cavity.progress.tic_reset(NN, True, "processing...")
            for nx, ny in self.grid.mode_numbers_tot:            
                psi = self.grid.fourier_basis_func(nx, ny, True, False)
                psi *= self.__reflection_mask
                R_mat_tot[:,i] = self.grid.arr_to_vec(psi, False, False)
                i += 1
                self.cavity.progress.tic()
        
        else:
            #########################################
            # parallel version
            #########################################
            keep_clsBlockMatrix_alive(True)
            result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_R_mat_tot_helper)(i) for i in range(NN))
            keep_clsBlockMatrix_alive(False)
            R_mat_tot =  np.column_stack(result) 

        self._RT_matrices.set_block(0, 0, R_mat_tot)
        self._R_mat_available = True
        R_mat_tot = None        
        
        gc.collect()
        self.cavity.progress.pop()
        
        
    def calc_T_mat_tot(self):
        """
        Creates a transmission matrix for propagation through lens 
        (total resolution)
        """        
        
        
        # simulate propagation for all modes in parralell        
        self.cavity.progress.push_print("calculating transmission matrix T")
        NN = self.grid.res_tot**2
        
        #start_time = time.perf_counter() 
        ##  PARALLEL Variante 1
        #pool = clsPoolSingleton(self.cavity.mp_pool_processes)
        #result = pool.map(self._calc_T_mat_tot_helper, range(NN))  
        #self.__T_mat = np.column_stack(result)
        #print("done")
        
        if not self.par_RT_calc:
            #########################################
            # non-parallel version
            #########################################
            T_mat_tot = np.empty((NN,NN), dtype=self.cavity.dtype_c)        
            
            if self.__lens_mask is None:
                self.__calc_lens_mask()
            
            i = 0
            self.cavity.progress.tic_reset(NN, True, "processing...")
            for nx, ny in self.grid.mode_numbers_tot:            
                psi = self.grid.fourier_basis_func(nx, ny, True, False)
                psi *= self.__lens_mask
                T_mat_tot[:,i] = self.grid.arr_to_vec(psi, False, False)
                i += 1
                self.cavity.progress.tic()
        
        else:
            #########################################
            # Parallel version (Variante 2)
            #########################################
            if self.__lens_mask is None:
                self.__calc_lens_mask()
                
            keep_clsBlockMatrix_alive(True)
            result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_T_mat_tot_helper)(i) for i in range(NN))
            keep_clsBlockMatrix_alive(False)
            T_mat_tot = np.column_stack(result)
        
        # PARALLEL Variante 3
        #self.__T_mat_tot = np.empty((NN,NN), dtype=complex)
        #start_time = time.perf_counter()
        #with ProcessPoolExecutor() as executor:
        #    futures = [executor.submit(test_worker, i, 
        #                               self.grid.mode_numbers_tot,
        #                               self.lens_mask, self.grid.res_tot, 
        #    futures = as_completed(futures)
        #    for future in futures:
        #        i, r = future.result()
        #        self.__T_mat_tot[:, i] = r
        #elapsed_time = time.perf_counter() - start_time
        #print(f"Elapsed time: {elapsed_time} seconds")        
        
        self._RT_matrices.set_block(0, 1, T_mat_tot)
        self._T_mat_available = True
        T_mat_tot = None
        
        gc.collect()
        self.cavity.progress.pop()
        
    @property 
    def T_LTR_mat_tot(self):    
        """ returns the left-to-right transmission matrix (total resolution)"""
        if not self._T_mat_available:
            self.calc_T_mat_tot()
        return self._RT_matrices.get_block(0, 1)
    
    @property    
    def T_RTL_mat_tot(self):    
        """ returns the right-to-left transmission matrix (total resolution)"""
        if not self._T_mat_available:
            self.calc_T_mat_tot()
        return self._RT_matrices.get_block(0, 1)
    
    @property  
    def R_L_mat_tot(self):    
        """ returns the left reflection matrix (total resolution)"""
        if self.__R_residual == 0:
            return 0
        else:
            if not self._R_mat_available:
                self.calc_R_mat_tot()
            return self._RT_matrices.get_block(0, 0)
            
    @property
    def R_R_mat_tot(self):    
        """ returns the right reflection matrix (total resolution)"""
        if self.__R_residual == 0:
            return 0
        else:
            if not self._R_mat_available:
                self.calc_R_mat_tot()
            if self.__sym_phase:
                return self._RT_matrices.get_block(0, 0)
            else:
                return -self._RT_matrices.get_block(0, 0)
    
    @property
    def f(self):
        """ returns the focal length in m """
        return self.__f
    
    @property
    def f_mm(self):
        """ returns the focal length in mm """
        return self.__f_mm
    
    @f.setter
    def f(self, f: float):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__f = f
        self.__f_mm = f * 1000
        self.clear_mem_cache()
    
    @f_mm.setter
    def f_mm(self, f_mm: float):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__f_mm = f_mm
        self.__f = f_mm / 1000
        self.clear_mem_cache()
    
    @property
    def aperture_black_value(self):
        if not self.cavity is None:
            self.cavity.clear()        
        return self.__aperture_black_value
    
    @aperture_black_value.setter
    def aperture_black_value(self, x):
        if not self.cavity is None:
            self.cavity.clear()        
        if x<0:
            x = 0
        if x>1:
            x = 1
        self.__aperture_black_value = x
        self.clear_mem_cache()
    
    @property
    def aperture(self):
        """ returns the aperture in m """
        return self.__aperture
    
    @property
    def aperture_mm(self):
        """ returns the aperture in mm """
        return self.__aperture_mm
    
    @aperture.setter
    def aperture(self, a: float):
        if not self.cavity is None:
            self.cavity.clear()        
        if a < 0:
            a = 0
        self.__aperture = a
        self.__aperture_mm = a * 1000
        self.clear_mem_cache()
        
    @aperture_mm.setter
    def aperture_mm(self, a_mm: float):
        if not self.cavity is None:
            self.cavity.clear()        
        if a_mm < 0:
            a_mm = 0
        self.__aperture_mm = a_mm
        self.__aperture = a_mm / 1000
        self.clear_mem_cache()
        
    @property
    def aperture_anti_alias(self):
        return self.__aperture_anti_alias
    
    @aperture_anti_alias.setter
    def aperture_anti_alias(self, aa: bool):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__aperture_anti_alias = aa
        
    def set_aperture_based_on_NA(self, NA: float):
        """ sets the aperture based on the numerical aperture """
        if not self.cavity is None:
            self.cavity.clear()        
        self.aperture = 2 * self.__f * np.tan(np.arcsin(NA))
    
    @property
    def lens_type_spherical(self):
        """ returns True if it is a spherical lens """
        return self.__lens_type_spherical  
    
    @lens_type_spherical.setter
    def lens_type_spherical(self, spherical: bool):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__lens_type_spherical = spherical
        self.__lens_type_perfect = not spherical
        self.clear_mem_cache()

    @property
    def lens_type_perfect(self):
        """ returns True if it is a perfect lens """
        return self.__lens_type_perfect  
    
    @lens_type_perfect.setter
    def lens_type_perfect(self, perfect: bool):
        if not self.cavity is None:
            self.cavity.clear()        
        self.__lens_type_perfect =  perfect
        self.__lens_type_spherical = not perfect
        self.clear_mem_cache()
        
    
    @property
    def lens_mask(self):
        """ returns the lens-mask """
        if self.__lens_mask is None:
            self.__calc_lens_mask()
        return self.__lens_mask
    
    
    @property
    def reflection_mask(self):
        """ returns the lens-mask """
        if self.__reflection_mask is None:
            if self.__reflection_mask1 is None or self.__reflection_mask2 is None:
                self.__calc_reflection_masks()
            self.__reflection_mask = self.__r_residual * (self.__reflection_mask1 + self.__reflection_mask2 ) / np.sqrt(2)
            #self.__reflection_mask = self.__r_residual * (self.__reflection_mask1 + self.__reflection_mask2 * 1j) * (0.5-0.5j)
            self.__reflection_mask1 = None
            self.__reflection_mask2 = None
            gc.collect()
            
            if self.__reflection_aperture > 0:
                self.__reflection_mask *= self.grid.get_soft_aperture_mask(
                    self.__reflection_aperture, 
                    self.__reflection_aperture_epsilon, 
                    self.__reflection_aperture_black)
        return self.__reflection_mask
    
    @property
    def reflection_mask1(self):
        """ returns the lens-mask """
        if self.__reflection_mask1 is None:
            self.__calc_reflection_masks()
        return self.__reflection_mask1
    
    @property
    def reflection_mask2(self):
        """ returns the lens-mask """
        if self.__reflection_mask2 is None:
            self.__calc_reflection_masks()
        return self.__reflection_mask2
    
    def clear_mem_cache(self):   
        super().clear_mem_cache()
        if not self.__lens_mask is None:            
            self.__lens_mask = None
            self.cavity.progress.push_print(
                "deleting lens phase mask of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if not self.__reflection_mask is None:            
            self.__reflection_mask = None
            self.cavity.progress.push_print(
                "deleting reflection mask of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if not self.__reflection_mask1 is None:            
            self.__reflection_mask1 = None
            self.cavity.progress.push_print(
                "deleting reflection mask 1 of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if not self.__reflection_mask2 is None:            
            self.__reflection_mask2 = None
            self.cavity.progress.push_print(
                "deleting reflection mask 2 of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if self._T_mat_available:
            self._T_mat_available = False
            self._RT_matrices.set_block(0, 1, 0)
            self.cavity.progress.push_print(
                "deleting transmission matrix T of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        if self._R_mat_available:
            self._R_mat_available = False
            self._RT_matrices.set_block(0, 0, 0)
            self.cavity.progress.push_print(
                "deleting reflection matrix R of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        gc.collect()
   
        
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        """
        Propagates the input field E_in in positive z-direction (LTR) 
        or negative z-Direction (RTL)
        """
        ax = self.grid.axis_tot
        if k_space_in:
            # input in k-space: Convert to position-space
            E_in = ifft2_phys_spatial(E_in, ax)
        
        # apply lens-mask
        E_in *= self.lens_mask
        
        if k_space_out:
            # output in k-space: Convert to k-space
            E_in = fft2_phys_spatial(E_in, ax)
        
        return E_in
    
    def __calc_lens_mask(self):
        # Create coordinate grid 
        ax = self.grid.axis_tot
        x, y = np.meshgrid(ax, ax)
        # Calculate k
        k0 = 2 * np.pi / self._lambda
        
        # Create lens mask
        if self.__lens_type_spherical:
            # thin spherical lens
            self.__lens_mask = np.exp(-1j * k0 / (2 * self.__f) * 
                                      (x**2 + y**2))
        else:
            # thin aspherical perfect lens
            self.__lens_mask = np.exp(-1j * k0 * 
                                      (np.sqrt(self.__f**2 + x**2 + y**2) 
                                       - self.__f))
        
        if self.__R_residual > 0:
            self.__lens_mask *= self.t_residual  
        
        # add aperture, if required
        if self.__aperture>0:
            if self.__aperture_anti_alias:
                lens_pupil = self.grid.get_aperture_mask(
                    self.__aperture, 4, self.__aperture_black_value)
            else:
                lens_pupil = self.grid.get_aperture_mask(
                    self.__aperture, 0, self.__aperture_black_value)
            
            self.__lens_mask *= lens_pupil
    
    def __calc_reflection_masks(self):
        if self.__R_residual == 0:
            self.__reflection_mask1 = None
            self.__reflection_mask2 = None
            return
        
        ax = self.grid.axis_tot
        x, y = np.meshgrid(ax, ax)
        k0 = 2 * np.pi / self._lambda
        
        # reflection on outside facet
        f = -self.__f/2
        if self.__lens_type_spherical:
            # thin spherical lens
            self.__reflection_mask1 = np.exp(-1j * k0 / (2 * f) * 
                                            (x**2 + y**2)) 
        else:
            # thin aspherical perfect lens
            self.__reflection_mask1 = np.exp(-1j * k0 * 
                                      (np.sqrt(f**2 + x**2 + y**2) 
                                       - f)) 
        
        # reflection on inside facet
        f = f/3
        if self.__lens_type_spherical:
            # thin spherical lens
            self.__reflection_mask2 = np.exp(-1j * k0 / (2 * f) * 
                                            (x**2 + y**2)) 
        else:
            # thin aspherical perfect lens
            self.__reflection_mask2 = np.exp(-1j * k0 * 
                                      (np.sqrt(f**2 + x**2 + y**2) 
                                       - f)) 
        
        
###############################################################################
# Amplitude Scaling
# increases or deerduces the amplitude
###############################################################################
class clsAmplitudeScaling(clsOptComponent2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__scale_factor = 1
        
    @property
    def symmetric(self):
        return True

    @property
    def k_space_in_prefer(self):    
        return False 
    
    @property
    def k_space_out_prefer(self):    
        return False     

    @property
    def k_space_in_dont_care(self):    
        return True

    @property
    def k_space_out_dont_care(self):    
        return True
    
    @property 
    def amplitude_scale_factor(self):
        return self.__scale_factor
    
    @amplitude_scale_factor.setter
    def amplitude_scale_factor(self, f):
        self.__scale_factor = f
        
    @property 
    def T_LTR_mat_tot(self):    
        """ returns the left-to-right transmission matrix """
        return self.__scale_factor
    
    @property 
    def T_RTL_mat_tot(self):    
        """ returns the right-to-left transmission matrix """
        return self.__scale_factor
     
    @property  
    def R_L_mat_tot(self):    
        """ returns the left reflection matrix """
        return 0
    
    @property
    def R_R_mat_tot(self):    
        """ returns the right reflection matrix """
        return 0 
    
    @property
    def dist_phys(self):
        """ returns physical propgation distance in meters """
        return 0
    
    @property
    def dist_opt(self):
        """ returns optical propgation distance in meters """
        return 0
    
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        E_out = E_in * self.__scale_factor
        if k_space_in != k_space_out:
            fov_res = self.grid.is_fov_res(E_in) 
            E_out = self.grid.convert(E_out, k_space_in, k_space_out, fov_res)
        return E_out
    
    def reflect(self, E_in, k_space_in, k_space_out, side: Side):
        return 0
    
###############################################################################
# clsPropagation
# represents propagation over a certain distance
############################################################################### 
class clsPropagation(clsOptComponent2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__n = 1 # refractive index
        self.__dist_phys = 0.001 # physical propagation distance in meters
        self.__transfer_function = 1 # default: Fresnel Transfer Function
        self.__T_mat_tot = None

    
    def calc_inv_M_bmat_tot(self):
        """ 
        calculates the inverse transfer block matrix in a fast way
        """
        M = self.M_bmat_tot
        if self.__n.imag == 0:   
            if self.cavity.use_bmatrix_class:
                M_inv = clsBlockMatrix(2, 
                                       self.cavity.use_swap_files_in_bmatrix_class,
                                       self.cavity.tmp_folder)
                M_inv.set_block(0,0, M.get_block(1,1))
                M_inv.set_block(1,1, M.get_block(0,0))
                return M_inv
            else:
                return [[M[1][1],0],[0,M[0][0]]] 
        else:
            return bmat2_inv(M, self.cavity.progress, "inverting M")

    @property
    def symmetric(self):
        return True

    @property
    def k_space_in_prefer(self):    
        return True # preferred input: position-space
    
    @property
    def k_space_out_prefer(self):    
        return True  # preferred output: position-space
    
    @property
    def k_space_in_dont_care(self):    
        return False

    @property
    def k_space_out_dont_care(self):    
        return False

    def clear_mem_cache(self):   
        super().clear_mem_cache()
        if not self.__T_mat_tot is None:
            self.__T_mat_tot = None
            self.cavity.progress.push_print(
                "deleting transmission matrix T of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
       
        
    def calc_T_mat_tot(self):
        """
        Creates a transmission matrix for propagation through free space
        or absorber (with refractive index n) in total resolution
        """        
        # simulate propagation for all modes in parallel
        E_out = self._prop(1, True, True, False)        
        self.__T_mat_tot = diags(self.grid.arr_to_vec(E_out, True, False))        
        
    @property 
    def T_LTR_mat_tot(self):    
        """ returns the left-to-right transmission matrix """
        if self.__T_mat_tot is None:
            self.calc_T_mat_tot()
        return self.__T_mat_tot
    
    @property    
    def T_RTL_mat_tot(self):    
        """ returns the right-to-left transmission matrix """
        if self.__T_mat_tot is None:
            self.calc_T_mat_tot()
        return self.__T_mat_tot
    
    @property  
    def R_L_mat_tot(self):    
        """ returns the left reflection matrix """
        return 0
    
    @property
    def R_R_mat_tot(self):    
        """ returns the right reflection matrix """
        return 0
    
    
    @property
    def n(self):
        """ complex refractive index """
        return self.__n
    
    @property
    def transfer_function(self):
        """ 0: Rayleigh Sommerfeld, 1: Fresnel """
        return self.__transfer_function   
    
    @transfer_function.setter
    def transfer_function(self, TF: int):
        """
        TF ........ 0: use Rayleigh Sommerfeld Transfer function
                    1: use Fresnel Transfer Function 
        """
        if not self.cavity is None:
            self.cavity.clear()        
        self.__T_mat_tot = None
        if TF < 0:
            print("Warning: transfer_function must be 0 or 1.")
            print("Setting transfer_function distance to 0 (Rayleigh Sommerfeld).")
            self.__transfer_function = 0
        elif TF > 1:
            print("Warning: transfer_function must be 0 or 1.")
            print("Setting transfer_function distance to 1 (Fresnel).")
            self.__transfer_function = 1
        else:
            self.__transfer_function = TF
    
    @property
    def dist_phys(self):
        """ returns physical propgation distance in meters """
        return self.__dist_phys
    
    @property
    def dist_opt(self):
        """ returns optical propgation distance in meters """
        return self.__dist_phys  * complex(self.__n).real

    
    def set_dist_opt(self, d_opt: float):
        """ defines the pyhsical distance by setting the optical distance """
        if not self.cavity is None:
            self.cavity.clear()
        self.__dist_phys = d_opt / complex(self.__n).real
    
        
    def set_ni_based_on_T(self, T: float):
        """ Sets the imaginary part of the refractive index so that 
        (based on dist_phys and lambda) it results in a transmittivity T.
        Must be called again if dist_phys or lambda changes. """
        if not self.cavity is None:
            self.cavity.clear()
        k_c = 2 * np.pi / self.Lambda        
        ni = -np.log(T) / (2 * self.__dist_phys * k_c)
        self.__n = complex(self.__n).real + complex(0,ni)

    def set_params(self, dist_phys, n):
        """ 
        dist_phys: physical distance in z direction to propagate
        n: complex refration index of material (1 for free space)
        """
        self.__T_mat_tot = None
        if dist_phys<0:
            print("Warning: Negative propagation distance not allowed. ")
            print("Setting propagation distance to zero.")
            dist_phys = 0
        
        if not self.cavity is None:
            self.cavity.clear()        
        self.__dist_phys = dist_phys
        self.__n = n
    
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        return self._prop(E_in, k_space_in, k_space_out, False)
        
    def _prop(self, E_in, k_space_in: bool, k_space_out: bool, fov_only: bool):
        """
        Propagates the input field E_in in positive z-direction (L->R)
        either with Rayleigh Sommerfeld transfer function approach
        or with Fresnel Transfer Function approach
        E_in (in either position-space or k-space).
        can also be a float if k_space_in = True
        """
        if fov_only:
            ax = self.grid.axis_fov
            k_ax = self.grid.k_axis_fov
        else:
            ax = self.grid.axis_tot
            k_ax = self.grid.k_axis_tot
            
        # Create coordinate grid in k-space
        kx, ky = np.meshgrid(k_ax, k_ax)
        # Calculate k_tot
        k_tot = 2 * np.pi / self._lambda
        
        if self.__transfer_function == 0:
            # Rayleigh Sommerfeld transfer function approach. 
            # see: Voelz D, "Computational Fourier optics", (4.19)
            # requires prop_dist >> lambda 
            
            # Calculate kz-vector
            kz = np.sqrt(complex((self.__n * k_tot)**2) - kx**2 - ky**2
                         ).astype(self.cavity.dtype_c)
            # Angular spectrum of propagator in z-direction
            angs_prop_z = np.exp(complex(0, 1) * kz * self.__dist_phys
                                 ).astype(self.cavity.dtype_c)
            
        else:
            # Fresnel transfer function approach. (paraxial approximation)
            # Angular spectrum of propagator in z-direction
            angs_prop_z = np.exp(complex(0, 1) * 
                                 (self.__n * k_tot * self.__dist_phys - 
                                  self.__dist_phys * (kx**2 + ky**2) / 
                                  (2 * k_tot * self.__n))
                                 ).astype(self.cavity.dtype_c)

        if not k_space_in:
            #  input not in k-space, convert to k-space
            E_in = fft2_phys_spatial(E_in, ax)
            
        # Apply angular spectrum of propagator in k-space
        E_out = E_in * angs_prop_z ;
        
        if not k_space_out:
            # output demanded in position space, convert to position-space
            E_out = ifft2_phys_spatial(E_out, ax)
        
        return E_out
        
 
###############################################################################
# clsSplitPropagation
# represents propagation over a certain distance
# different parameters for top and bottom half
############################################################################### 
class clsSplitPropagation(clsOptComponent2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__n1 = 1 # refractive index top half
        self.__n2 = 1 # refractive index bottom half
        self.__dist_phys = 0.001 # physical propagation distance in meters
        self.__dist_corr1 = 0 # distance corection top half
        self.__dist_corr2 = 0 # distance corection top half
        self.__transfer_function = 1 # default: Fresnel Transfer Function
        self.__T_mat_tot = None

    @property
    def symmetric(self):
        return True

    @property
    def k_space_in_prefer(self):    
        return True # preferred input: position-space
    
    @property
    def k_space_out_prefer(self):    
        return True  # preferred output: position-space
    
    @property
    def k_space_in_dont_care(self):    
        return False

    @property
    def k_space_out_dont_care(self):    
        return False

    def clear_mem_cache(self):   
        super().clear_mem_cache()
        if not self.__T_mat_tot is None:
            self.__T_mat_tot = None
            self.cavity.progress.push_print(
                "deleting transmission matrix T of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
       
    def _calc_T_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, True)
        psi = self.prop(psi, True, False, Dir.LTR) 
        return self.grid.arr_to_vec(psi, False, False) 
       
    
    def calc_T_mat_tot(self):
        """
        Creates a transmission matrix for propagation through free space
        or absorber (with refractive index n) in total resolution
        """        
        # simulate propagation for all modes in parallel
        #E_out = self._prop(1, True, True, False)        
        #self.__T_mat_tot = diags(self.grid.arr_to_vec(E_out, True, False))        
        self.cavity.progress.push_print("calculating transmission matrix")
        NN = self.grid.res_tot**2
        keep_clsBlockMatrix_alive(True)
        result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_T_mat_tot_helper)(i) for i in range(NN))
        keep_clsBlockMatrix_alive(False)
        self.__T_mat_tot = np.column_stack(result)
        
        self.cavity.progress.pop()
        
        
    @property 
    def T_LTR_mat_tot(self):    
        """ returns the left-to-right transmission matrix """
        if self.__T_mat_tot is None:
            self.calc_T_mat_tot()
        return self.__T_mat_tot
    
    @property    
    def T_RTL_mat_tot(self):    
        """ returns the right-to-left transmission matrix """
        if self.__T_mat_tot is None:
            self.calc_T_mat_tot()
        return self.__T_mat_tot
    
    @property  
    def R_L_mat_tot(self):    
        """ returns the left reflection matrix """
        return 0
    
    @property
    def R_R_mat_tot(self):    
        """ returns the right reflection matrix """
        return 0
    
    
    @property
    def n1(self):
        """ complex refractive index top half"""
        return self.__n1
    
    @property
    def n2(self):
        """ complex refractive index bottom half"""
        return self.__n2
    
    @property
    def transfer_function(self):
        """ 0: Rayleigh Sommerfeld, 1: Fresnel """
        return self.__transfer_function   
    
    @transfer_function.setter
    def transfer_function(self, TF: int):
        """
        TF ........ 0: use Rayleigh Sommerfeld Transfer fuction
                    1: use Fresnel Transfer Function 
        """
        if not self.cavity is None:
            self.cavity.clear()        
        self.__T_mat_tot = None
        if TF < 0:
            print("Warning: transfer_function must be 0 or 1.")
            print("Setting transfer_function distance to 0 (Rayleigh Sommerfeld).")
            self.__transfer_function = 0
        elif TF > 1:
            print("Warning: transfer_function must be 0 or 1.")
            print("Setting transfer_function distance to 1 (Fresnel).")
            self.__transfer_function = 1
        else:
            self.__transfer_function = TF
    
    @property
    def dist_phys(self):
        """ returns physical propgation distance in meters """
        return self.__dist_phys
    
    @property
    def dist_opt1(self):
        """ returns optical propgation distance (top half) in meters """
        return (self.__dist_phys + self.__dist_corr1)  * complex(self.__n1).real

    @property
    def dist_opt2(self):
        """ returns optical propgation distance (ottom half) in meters """
        return (self.__dist_phys + self.__dist_corr2)  * complex(self.__n2).real
    
    @property
    def dist_opt(self):
        return max(self.dist_opt1, self.dist_opt2)
    
    def set_dist_opt(self, d_opt: float, top_bottom: int):
        """ defines the pyhsical distance by setting the optical distance """
        if not self.cavity is None:
            self.cavity.clear()        
        if top_bottom <=1:
            self.__dist_phys = d_opt / complex(self.__n1).real
        else:
            self.__dist_phys = d_opt / complex(self.__n2).real
    
        
    def set_ni_based_on_T(self, T: float, top_bottom: int):
        """ 
        Sets the imaginary part of the refractive index so that 
        (based on dist_phys and lambda) it results in a transmittivity T.
        Must be called again if dist_phys or lambda changes. 
        top_bottom ... if 1, top part, else bottom part
        """
        if not self.cavity is None:
            self.cavity.clear()        
        k_c = 2 * np.pi / self.Lambda        
        ni = -np.log(T) / (2 * self.__dist_phys * k_c)
        if top_bottom <= 1:
            # top
            self.__n1 = complex(self.__n1).real + complex(0,ni)
        else:
            # bottom
            self.__n2 = complex(self.__n2).real + complex(0,ni)

    def set_params(self, dist_phys, n1, n2, dist_corr1 = 0, dist_corr2 = 0):
        """ 
        dist_phys: physical distance in z direction to propagate
        n: complex refration index of material (1 for free space)
        """
        if not self.cavity is None:
            self.cavity.clear()        
        self.__T_mat_tot = None
        if dist_phys<0:
            print("Warning: Negative propagation distance not allowed. ")
            print("Setting propagation distance to zero.")
            dist_phys = 0
        
        self.__dist_phys = dist_phys
        self.__n1 = n1
        self.__n2 = n2
        self.__dist_corr1 = dist_corr1
        self.__dist_corr2 = dist_corr2
        
    
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        #first, calulate the whole field with top parameters
        E_out = self._prop(E_in, k_space_in, False, False, 1)
        # now with bottom aparameters
        E_out2 = self._prop(E_in, k_space_in, False, False, 2)
        
        # now replace bottom part of E_out
        lines = E_in.shape[0]
        lines_h = lines//2
        E_out[:lines_h,:] =  E_out2[:lines_h,:]
        
        if k_space_out:
            E_out = fft2_phys_spatial(E_out, self.grid.axis_tot)
        
        return E_out
        
    def _prop(self, E_in, k_space_in: bool, k_space_out: bool, fov_only: bool,
              top_bottom: int):
        """
        Propagates the input field E_in in positive z-direction (L->R)
        either with Rayleigh Sommerfeld transfer function approach
        or with Fresnel Transfer Function approach
        E_in (in either position-space or k-space).
        can also be a float if k_space_in = True
        top_bottom ... if 1, use parameters for top part, else bottom part
        """
        if fov_only:
            ax = self.grid.axis_fov
            k_ax = self.grid.k_axis_fov
        else:
            ax = self.grid.axis_tot
            k_ax = self.grid.k_axis_tot
        
        if top_bottom <= 1:
            n = self.__n1
            dist = self.__dist_phys + self.__dist_corr1
        else:
            n = self.__n2
            dist = self.__dist_phys + self.__dist_corr2
        
        # Create coordinate grid in k-space
        kx, ky = np.meshgrid(k_ax, k_ax)
        # Calculate k_tot
        k_tot = 2 * np.pi / self._lambda
        
        if self.__transfer_function == 0:
            # Rayleigh Sommerfeld transfer function approach. 
            # see: Voelz D, "Computational Fourier optics", (4.19)
            # requires prop_dist >> lambda 
            
            # Calculate kz-vector
            kz = np.sqrt(complex((n * k_tot)**2) - kx**2 - ky**2
                         ).astype(self.cavity.dtype_c)
            # Angular spectrum of propagator in z-direction
            angs_prop_z = np.exp(complex(0, 1) * kz * dist
                                 ).astype(self.cavity.dtype_c)
            
        else:
            # Fresnel transfer function approach. (paraxial approximation)
            # Angular spectrum of propagator in z-direction
            angs_prop_z = np.exp(complex(0, 1) * 
                                 (n * k_tot * dist - dist * (kx**2 + ky**2) / 
                                  (2 * k_tot * n))).astype(self.cavity.dtype_c)

        if not k_space_in:
            #  input not in k-space, convert to k-space
            E_in = fft2_phys_spatial(E_in, ax)
            
        # Apply angular spectrum of propagator in k-space
        E_out = E_in * angs_prop_z ;
        
        if not k_space_out:
            # output demanded in position space, convert to position-space
            E_out = ifft2_phys_spatial(E_out, ax)
        
        return E_out    
 

   
    
###############################################################################
# clsMirror
# represents a partially reflective flat mirror
############################################################################### 
class clsMirror(clsMirrorBase2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None         
        self._proj_factor_y2_L = 1
        self._proj_factor_y2_R = 1
            
     
    @clsMirrorBase2port.incident_angle_y_deg.setter
    def incident_angle_y_deg(self, alpha):     
        clsMirrorBase2port.incident_angle_y_deg.fset(self, alpha)        
        # TODO: clear mem           
     
    @clsMirrorBase2port.incident_angle_y.setter
    def incident_angle_y(self, alpha):        
        clsMirrorBase2port.incident_angle_y.fset(self, alpha) 
        # TODO: clear mem
    
    def clear_mem_cache(self):   
        super().clear_mem_cache()
        if (not self.__R_L_mat_tot is None) or (not self.__R_R_mat_tot is None):
            self.__R_L_mat_tot = None
            self.__R_R_mat_tot = None
            self.cavity.progress.push_print(
                "deleting reflection matrices R_L and R_R of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
            
        gc.collect()    
    
    
    @clsMirrorBase2port.left_side_relevant.setter
    def left_side_relevant(self, r: bool):
        #super().left_side_relevant = r  # Call the base class setter
        clsMirrorBase2port.left_side_relevant.fset(self, r)
        self.__R_L_mat_tot = None  
    
    @clsMirrorBase2port.right_side_relevant.setter
    def right_side_relevant(self, r: bool):
        clsMirrorBase2port.right_side_relevant.fset(self, r)  # Call the base class setter
        self.__R_R_mat_tot = None  
               
    @clsMirrorBase2port.rot_around_x_deg.setter
    def rot_around_x_deg(self, alpha):        
        clsMirrorBase2port.rot_around_x_deg.fset(self, alpha)        
        self.__tilt_mask_x_L = None
        self.__tilt_mask_x_R = None
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None   
    
    @clsMirrorBase2port.rot_around_y_deg.setter
    def rot_around_y_deg(self, alpha):
        clsMirrorBase2port.rot_around_y_deg.fset(self, alpha)        
        self.__tilt_mask_y_L = None
        self.__tilt_mask_y_R = None
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None

     
    @clsMirrorBase2port.rot_around_x.setter
    def rot_around_x(self, alpha):        
        clsMirrorBase2port.rot_around_x.fset(self, alpha)  
        self.__tilt_mask_x_L = None
        self.__tilt_mask_x_R = None
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None
     
    @clsMirrorBase2port.rot_around_y.setter
    def rot_around_y(self, alpha):        
        clsMirrorBase2port.rot_around_y.fset(self, alpha)
        self.__tilt_mask_x_L = None
        self.__tilt_mask_x_R = None
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None

    @property
    def symmetric(self):
        if self.mirror_tilted:
            return False
        elif self.left_refl_size_adjust:
            return False
        elif self.right_refl_size_adjust:
            return False
        else:
            return True
        
    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
    
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space
    
    @property
    def k_space_in_dont_care(self):    
        # we don't care if the input is in k-space or position space
        # only if the mirror is not tilted
        return (self.rot_around_x == 0 and self.rot_around_y == 0)

    @property
    def k_space_out_dont_care(self):    
        # we don't care if the output is in k-space or position space
        # only if the mirrir is not tilted
        return (self.rot_around_x == 0 and self.rot_around_y == 0)

        
    def _set_rt(self):
        super()._set_rt()
        self.__R_L_mat_tot = None
        self.__R_R_mat_tot = None
        
    @property    
    def T_LTR_mat_tot(self):
        if self.LTR_transm_behaves_like_refl_left:
            # LTR transmission should behave like reflection from left
            return self._R_L_mat_tot_helper()
        
        elif self.LTR_transm_behaves_like_refl_right: 
            # LTR transmission should behave like reflection from right
            return self._R_R_mat_tot_helper()
        
        elif self.LTR_transm_behaves_neutral: 
            # LTR transmission should do nothing
            return 1
        
        else:
            # normal behavior: 
            # transmission is just determined by transmittivity
            return self.t_LTR
    
    @property    
    def T_RTL_mat_tot(self):
        if self.RTL_transm_behaves_like_refl_left:
            # Transmission should behave like reflection from left
            return self._R_L_mat_tot_helper()
        
        elif self.RTL_transm_behaves_like_refl_right: 
            # Transmission should behave like reflection from right
            return self._R_R_mat_tot_helper()
        
        elif self.RTL_transm_behaves_neutral: 
            # LTR transmission should do nothing
            return 1
        
        else:
            # normal behavior: 
            # transmission is just determined by transmittivity
            return self.t_RTL        
    
    def _calc_R_L_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        
        psi *= self.r_L
            
        if self.left_side_relevant:
            if self.rot_around_y != 0:
                psi *= self._tilt_mask_x_L 
            if self.rot_around_x != 0:
                psi *= self._tilt_mask_y_L                 
            if self._proj_factor_y2_L != 1:
                psi = self.grid.stretch_y(psi, 0, self._proj_factor_y2_L)
                                
        return self.grid.arr_to_vec(psi, False, False) 
    
    def calc_R_L_mat_tot(self):
        """
        Calulates the left reflection matrix with mirror tilt
        """
        # simulate propagation for all modes in parallel    
        if self.left_side_relevant:
            self.cavity.progress.push_print("calculating left reflection matrix R_L")
            if self.LTR_transm_behaves_like_refl_left and self.RTL_transm_behaves_like_refl_left:
                self.cavity.progress.print("(which governs LTR and RTL transmission behavior)")
            elif self.LTR_transm_behaves_like_refl_left:
                self.cavity.progress.print("(which governs the LTR transmission behavior)")
            elif self.RTL_transm_behaves_like_refl_left:
                self.cavity.progress.print("(which governs the RTL transmission behavior)")                
                
        if not self.left_side_relevant:
            # left side not relevant. Just apply reflection factor
            self.__R_L_mat_tot = self.r_L
        
        elif not (self.mirror_tilted or self.left_refl_size_adjust):
            # neither mirror tilted nor outgoing field adjustment
            # just apply refleciton factor
            self.__R_L_mat_tot = self.r_L
            
        else:
            if self.mirror_tilted:
                if self._tilt_mask_x_L is None or self._tilt_mask_y_L is None:
                    if self.rot_around_y != 0:
                        self.calc_tilt_masks_x()
                    if self.rot_around_x != 0:
                        self.calc_tilt_masks_y()
            
            NN = self.grid.res_tot**2
            
            # this will be used by _calc_R_L_mat_tot_helper
            if self.left_refl_size_adjust:
                self._proj_factor_y2_L = self.get_projection_factor_y2(Side.LEFT)
            else:
                self._proj_factor_y2_L = 1
            
            keep_clsBlockMatrix_alive(True)                            
            result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_R_L_mat_tot_helper)(i) for i in range(NN))
            keep_clsBlockMatrix_alive(False)
            self.__R_L_mat_tot = np.column_stack(result)
            
        
        if self.left_side_relevant: 
            self.cavity.progress.pop()
    
    def _calc_R_R_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
            
        psi *= self.r_R
        
        if self.right_side_relevant:
            if self.rot_around_y != 0:
                psi *= self._tilt_mask_x_R 
            if self.rot_around_x != 0:
                psi *= self._tilt_mask_y_R 
            if self._proj_factor_y2_R != 1:
                psi = self.grid.stretch_y(psi, 0, self._proj_factor_y2_R)
        
        return self.grid.arr_to_vec(psi, False, False) 
    
    def calc_R_R_mat_tot(self):
        """
        Calulates the right reflection matrix with mirror tilt
        """
        # simulate propagation for all modes in parallel        
        if self.right_side_relevant: 
            self.cavity.progress.push_print("calculating right reflection matrix R_R")
            if self.LTR_transm_behaves_like_refl_right and self.RTL_transm_behaves_like_refl_right:
                self.cavity.progress.print("(which governs LTR and RTL transmission behavior)")
            elif self.LTR_transm_behaves_like_refl_right:
                self.cavity.progress.print("(which governs the LTR transmission behavior)")
            elif self.RTL_transm_behaves_like_refl_right:
                self.cavity.progress.print("(which governs the RTL transmission behavior)")
                                        
        if not self.right_side_relevant:
            # right side not relevant. Just apply reflection factor
            self.__R_R_mat_tot = self.r_R
        
        elif not (self.mirror_tilted or self.right_refl_size_adjust):
            # neither mirror tilted nor outgoing field adjustment
            # just apply refleciton factor
            self.__R_R_mat_tot = self.r_R
            
        else:
            # mirror tilt and/or outgoing field stretch neccessary
            if self.mirror_tilted:
                if self._tilt_mask_x_R is None or self._tilt_mask_y_R is None:
                    if self.rot_around_y != 0:
                        self.calc_tilt_masks_x()
                    if self.rot_around_x != 0:
                        self.calc_tilt_masks_y()
                
            NN = self.grid.res_tot**2
            
            # this will be used by _calc_R_R_mat_tot_helper
            if self.left_refl_size_adjust:
                self._proj_factor_y2_R = self.get_projection_factor_y2(Side.RIGHT)
            else:
                self._proj_factor_y2_R = 1
            
            keep_clsBlockMatrix_alive(True)
            result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_R_R_mat_tot_helper)(i) for i in range(NN))
            keep_clsBlockMatrix_alive(False)
            self.__R_R_mat_tot = np.column_stack(result)
            
            
        if self.right_side_relevant: 
            self.cavity.progress.pop()
    
    @property    
    def R_L_mat_tot(self):
        if self.LTR_transm_behaves_like_refl_left:
            # LTR transmission should behave like reflection from left
            # this means that there is no reflection on the left side
            return 0
        
        elif self.LTR_transm_behaves_like_refl_right: 
            # LTR transmission should behave like reflection from right
            # this means, that there is no reflection on the left side
            return 0
        
        elif self.LTR_transm_behaves_neutral: 
            # LTR transmission should behave neutral
            # this means, that there is no reflection on the left side
            return 0
        
        else:
            return self._R_L_mat_tot_helper()
    
    def _R_L_mat_tot_helper(self):
        if not (self.mirror_tilted or self.left_refl_size_adjust):
            # no tilt or size adjustment
            return self.r_L
        
        else:
            if self.__R_L_mat_tot is None:
                self.calc_R_L_mat_tot()
                    
            return self.__R_L_mat_tot
    
    @property    
    def R_R_mat_tot(self):
        if self.RTL_transm_behaves_like_refl_left:
            # RTL transmission should behave like reflection from left
            # this means that there is no reflection on the right side
            return 0
        
        elif self.RTL_transm_behaves_like_refl_right: 
            # RTL transmission should behave like reflection from right
            # this means, that there is no reflection on the right side
            return 0
        
        elif self.RTL_transm_behaves_neutral: 
            # RTL transmission should behave neutral
            # this means, that there is no reflection on the right side
            return 0
              
        else:
            return self._R_R_mat_tot_helper()
    
    def _R_R_mat_tot_helper(self):
        if not (self.mirror_tilted or self.right_refl_size_adjust):
            # no tilt
            return self.r_R
        
        else:
            if self.__R_R_mat_tot is None:
                self.calc_R_R_mat_tot()
                    
            return self.__R_R_mat_tot
    
    def reflect(self, E_in, k_space_in, k_space_out, side: Side):
        """ 
        reflects the input field E_in on the left or right side 
        if .transm_behaves_like_refl_left or .transm_behaves_like_refl_right
        then transmission acts as reflection, and there is no reflection and
        just zero is returned
        """
        if side == Side.LEFT:
            if self.LTR_transm_behaves_like_refl_left:
                # no left reflection in "LTR transmission acts as reflection"-mode
                return 0
            
            elif self.LTR_transm_behaves_like_refl_right:
                # no left reflection in "LTR transmission acts as reflection"-mode
                return 0
            
            elif self.LTR_transm_behaves_neutral:
                # no left reflection in "LTR transmission acts neutral"-mode
                return 0
            
        elif side == Side.RIGHT:
            if self.RTL_transm_behaves_like_refl_left:
                # no right reflection in "RTL transmission acts as reflection"-mode
                return 0
            
            elif self.RTL_transm_behaves_like_refl_right:
                # no right reflection in "RTL transmission acts as reflection"-mode
                return 0
            
            elif self.RTL_transm_behaves_neutral:
                # no right reflection in "RTL transmission acts neutral"-mode
                return 0
        
        # normal reflection behavior
        currently_in_k_space = k_space_in
        ax = self.grid.axis_tot
        if side == Side.LEFT:
            # reflection on the left side
            E_out = E_in * self.r_L
                        
            if self.mirror_tilted:    
                # we need to calulate mirror tilt
                if currently_in_k_space:
                    # input in k-space: Convert to position-space
                    # to prepare for mirror tilt
                    E_out = ifft2_phys_spatial(E_out, ax)
                    currently_in_k_space = False
            
                # apply tilt masks
                if self.rot_around_y != 0:
                    E_out *= self.tilt_mask_x_L 
                if self.rot_around_x != 0:
                    E_out *= self.tilt_mask_y_L  
                    
            # if activated, change size of outgoing light field in y-direction 
            # according to mirror tilt and incident angle
            if self.project_according_to_angles:
                proj_factor = self.get_projection_factor_y2(Side.LEFT)
                if  proj_factor != 1:
                    if currently_in_k_space:
                        # currently in k-space: Convert to position-space
                        E_out = ifft2_phys_spatial(E_out, ax)
                        currently_in_k_space = False
                    
                    E_out = self.grid.stretch_y(E_out, 0, proj_factor)
                
        elif side == Side.RIGHT:
            # reflection on the right side
            E_out = E_in * self.r_R
            
            if self.mirror_tilted:   
                # we need to calculate mirror tilt
                if currently_in_k_space:
                    # input in k-space: Convert to position-space
                    # to prepare for mirror tilt
                    E_out = ifft2_phys_spatial(E_out, ax)
                    currently_in_k_space = False
            
                # apply tilt masks
                if self.rot_around_y != 0:
                    E_out *= self.tilt_mask_x_R 
                if self.rot_around_x != 0:
                    E_out *= self.tilt_mask_y_R  
                    
            # if activated, change size of outgoing light field in y-direction 
            # according to mirror tilt and incident angle
            if self.project_according_to_angles:
                proj_factor = self.get_projection_factor_y2(Side.RIGHT)
                if  proj_factor != 1:
                    if currently_in_k_space:
                        # currently in k-space: Convert to position-space
                        E_out = ifft2_phys_spatial(E_out, ax)
                        currently_in_k_space = False
                    
                    E_out = self.grid.stretch_y(E_out, 0, proj_factor)
        
        if currently_in_k_space:
             # result is currently in k-space
             if not k_space_out:
                 # but output should be in position-space
                 E_out = ifft2_phys_spatial(E_out, self.grid.axis_tot)
        
        else:
             # result is currently in position-space
             if k_space_out:
                 # but output should be in k-space
                 E_out = fft2_phys_spatial(E_out, self.grid.axis_tot)     

        return E_out
    
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        """ 
        Propagates the input field E_in in LTR or RTL z-direction 
        by simply applying the t transmission factor (for "normal" mirror)
        or by imitating the reflection behavior 
        (if .transm_behaves_like_refl_left or .transm_behaves_like_refl_right)
        """
        
        currently_in_k_space = k_space_in
        ax = self.grid.axis_tot
        
        if (direction == Dir.LTR and self.LTR_transm_behaves_like_refl_left) or \
            (direction == Dir.RTL and self.RTL_transm_behaves_like_refl_left):
            # Transmission behaves like reflection from left side
            E_out = E_in * self.r_L
                            
            if self.mirror_tilted:   
                # we need to calculate mirror tilt
                if currently_in_k_space:
                    # currently in k-space: Convert to position-space
                    E_out = ifft2_phys_spatial(E_out, ax)
                    currently_in_k_space = False
            
                # apply tilt mask
                if self.rot_around_y != 0:
                    E_out *= self.tilt_mask_x_L 
                if self.rot_around_x != 0:
                    E_out *= self.tilt_mask_y_L  
                
            # if activated, change size of outgoing light field in y-direction 
            # according to mirror tilt and incident angle
            if self.project_according_to_angles:
                proj_factor = self.get_projection_factor_y2(Side.LEFT)
                if  proj_factor != 1:
                    if currently_in_k_space:
                        # currently in k-space: Convert to position-space
                        E_out = ifft2_phys_spatial(E_out, ax)
                        currently_in_k_space = False
                    
                    E_out = self.grid.stretch_y(E_out, 0, proj_factor)
                                        
                                               
        elif (direction == Dir.LTR and self.LTR_transm_behaves_like_refl_right) or \
            (direction == Dir.RTL and self.RTL_transm_behaves_like_refl_right):
            # Transmission behaves like reflection from right side
            E_out = E_in * self.r_R
                
            if self.mirror_tilted:   
                # we need to calculate mirror tilt
                if currently_in_k_space:
                    # currently in k-space: Convert to position-space
                    E_out = ifft2_phys_spatial(E_out, ax)
                    currently_in_k_space = False
                
                # apply tilt mask
                if self.rot_around_y != 0:
                    E_out *= self.tilt_mask_x_R 
                if self.rot_around_x != 0:
                    E_out *= self.tilt_mask_y_R  
                
            # if activated, change size of outgoing light field in y-direction 
            # according to mirror tilt and incident angle
            if self.project_according_to_angles:
                proj_factor = self.get_projection_factor_y2(Side.RIGHT)
                if  proj_factor != 1:
                    if currently_in_k_space:
                        # currently in k-space: Convert to position-space
                        E_out = ifft2_phys_spatial(E_out, ax)
                        currently_in_k_space = False
                        
                    E_out = self.grid.stretch_y(E_out, 0, proj_factor)
                             
        elif (direction == Dir.LTR and self.LTR_transm_behaves_neutral) or \
            (direction == Dir.RTL and self.RTL_transm_behaves_neutral):
            # Transmission in this direction behaves neutral (does nothing)
            E_out = E_in
            
        else:
            # normal transmission (never affected by rotation)
            if direction == Dir.LTR:
                E_out = E_in * self.t_LTR
            else:
                E_out = E_in * self.t_RTL
            
                                        
        if currently_in_k_space:
             # result is currently in k-space
             if not k_space_out:
                 # but output should be in position-space
                 E_out = ifft2_phys_spatial(E_out, self.grid.axis_tot)
        
        else:
             # result is currently in position-space
             if k_space_out:
                 # but output should be in k-space
                 E_out = fft2_phys_spatial(E_out, self.grid.axis_tot)

        return E_out
    

###############################################################################
# clsSplitMirror
# represents a partially reflective mirror with different refletivities
# in the upper and the lower half
############################################################################### 
class clsSplitMirror(clsOptComponent2port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__sym_phase = True # symmetric phase behavior
        self.__R1 = 1 # power reflection top side from left and right
        self.__T1 = 0 # power transmission top side in either direction
        self.__r1 = -1 # complex reflection coefficient top from left and right
        self.__t1 = 0 # complex transmission coefficient top from either side
        self.__R2 = 1 # power reflection bottom side from left and right
        self.__T2 = 0 # power transmission bottom side in either direction
        self.__r2 = -1 # complex reflection coefficient bottom from left and right
        self.__t2 = 0 # complex transmission coefficient bottom from either side
        self.__rot_around_x_deg = 0 # rotation around x-axis in degrees
        self.__rot_around_y_deg = 0 # rotation around y-axis in degrees
        
        self.__tilt_mask_x_L = None
        self.__tilt_mask_x_R = None
        self.__tilt_mask_y_L = None
        self.__tilt_mask_y_R = None
         
        self.__RT_matrices = None
        self._R_L_mat_available = False
        self._R_R_mat_available = False
        self._T_mat_available = False
        
        self.__left_side_relevant = True
        self.__right_side_relevant = True
        self.par_RT_calc = True # parallel calcuation of R and T matrices 
    
    @property
    def _RT_matrices(self):
        """ 
        0,0 ... R_L_mat_tot
        0,1 ... T_mat_tot
        1,1 ... R_R_mat_tot
        """
        
        delete_old_instance = False
        create_new_instance = False
        if self.__RT_matrices is None:
            # no current instance: create a new instance
            create_new_instance = True
            
        else:
            # there is already an instance
            if self.cavity.use_swap_files_in_bmatrix_class:
                # it should be set to file caching
                if not self.__RT_matrices.file_caching:
                    # but it is not!
                    delete_old_instance = True
                    create_new_instance = True
                
                elif self.__RT_matrices.tmp_dir != self.cavity.tmp_folder:
                    # the current instance is set to file caching (as it should)
                    # but it refers to a wrong tmp folder!
                    delete_old_instance = True
                    create_new_instance = True
                
            else:
                # the current instance should *not* be set to file caching
                if self.__RT_matrices.file_caching:
                    # but it is!
                    delete_old_instance = True
                    create_new_instance = True
            
        if delete_old_instance:
            self._R_L_mat_available = False
            self._R_R_mat_available = False
            self._T_mat_available = False
            self.__RT_matrices = None
            gc.collect()
        
        if create_new_instance:
            self.__RT_matrices = clsBlockMatrix(2, 
                                            self.cavity.use_swap_files_in_bmatrix_class,
                                            self.cavity.tmp_folder)
        
        return self.__RT_matrices
    
    def clear_mem_cache(self):   
        super().clear_mem_cache()
        if (not self.__tilt_mask_x_L is None) or (not self.__tilt_mask_x_R is None):
            self.__tilt_mask_x_L = None
            self.__tilt_mask_x_R = None
            self.cavity.progress.push_print(
                "deleting phase mask (x-shift) of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        
        if (not self.__tilt_mask_y_L is None) or (not self.__tilt_mask_y_R is None):
            self.__tilt_mask_y_L = None
            self.__tilt_mask_y_R = None
            self.cavity.progress.push_print(
                "deleting phase mask (y-shift) of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
        
        if self._R_L_mat_available or self._R_R_mat_available:
            self._R_L_mat_available = False
            self._RT_matrices.set_block(0, 0, 0)
            
            self._R_R_mat_available = False
            self._RT_matrices.set_block(1, 1, 0)
            
            self.cavity.progress.push_print(
                "deleting reflection matrices R_L and R_R of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
            
        if self._T_mat_available:
            self._T_mat_available = False
            self._RT_matrices.set_block(0, 1, 0)
            
            self.cavity.progress.push_print(
                "deleting transmission matrix of '"+self.name+"' from memory cache") 
            self.cavity.progress.pop()
            
        gc.collect()
    
    @property
    def left_side_relevant(self):
        """ If False, the reflection on the left side may be calculated 
        incorrectly (e.g. for tilted mirrors) for sake of speed """
        return self.__left_side_relevant
    
    @left_side_relevant.setter
    def left_side_relevant(self, r: bool):
        if not self.cavity is None:
            self.cavity.clear()       
        self._R_L_mat_available = False
        self._RT_matrices.set_block(0, 0, 0)
        self.__left_side_relevant = r
    
    @property
    def right_side_relevant(self):
        """ If False, the reflection on the right side may be calculated 
        incorrectly (e.g. for tilted mirrors) for sake of speed """
        return self.__right_side_relevant
    
    @right_side_relevant.setter
    def right_side_relevant(self, r: bool):
        if not self.cavity is None:
            self.cavity.clear()       
        self._R_R_mat_available = False
        self._RT_matrices.set_block(1, 1, 0)
        self.__right_side_relevant = r
    
    @property
    def rot_around_x_deg(self):
        """ 
        angle in degree by which the mirror is rotated around the x axis
        positive values manes: Left reflection moves towards positive y direction
        ("up"), and the right reflection moves towards negative y values ("down")
        """
        return self.__rot_around_x_deg
     
    @rot_around_x_deg.setter
    def rot_around_x_deg(self, alpha):
        if abs(alpha)>=45:
            print("Error. Tilt angle must be < 45°")
        else:
            if not self.cavity is None:
                self.cavity.clear()       
            self.__rot_around_x_deg = alpha
            self.__tilt_mask_x_L = None
            self.__tilt_mask_x_R = None
            self._R_L_mat_available = False
            self._RT_matrices.set_block(0, 0, 0)
            self._R_R_mat_available = False
            self._RT_matrices.set_block(1, 1, 0)
   
    @property
    def rot_around_y_deg(self):
        """ 
        angle in degree by which the mirror is rotated arund the y axis
        positive values manes: Left reflection moves towards negative x direction
        ("left" if seen from the right side, or "right" if seen from the left side), 
        and the right reflection moves towards positive x values ("right"
        if seen from the right side, or "left" if seen from the left side)
        """
        return self.__rot_around_y_deg
    
    @rot_around_y_deg.setter
    def rot_around_y_deg(self, alpha):
        if abs(alpha)>=45:
            print("Error. Tilt angle must be < 45°")
        else:
            if not self.cavity is None:
                self.cavity.clear()       
            self.__rot_around_y_deg = alpha
            self.__tilt_mask_y_L = None
            self.__tilt_mask_y_R = None
            self._R_L_mat_available = False
            self._RT_matrices.set_block(0, 0, 0)
            self._R_R_mat_available = False
            self._RT_matrices.set_block(1, 1, 0)

    @property
    def symmetric(self):
        return (self.__rot_around_x_deg == 0 and self.__rot_around_y_deg == 0)
    
    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
    
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space
    
    @property
    def k_space_in_dont_care(self):    
        # we don't care if the input is in k-space or position space
        # only if the mirror is not tilted
        return False

    @property
    def k_space_out_dont_care(self):    
        # we don't care if the output is in k-space or position space
        # only if the mirrir is not tilted
        return False

    @property
    def sym_phase(self):
        return self.__sym_phase
    
    def __calc_tilt_masks_x(self):
        # shift in x-direction (depends on y rotation)
        if self.__rot_around_y_deg == 0:
            self.__tilt_mask_x_L = 1
            self.__tilt_mask_x_R = 1
        else:
            ax = self.grid.axis_tot
            x, _ = np.meshgrid(ax, ax)
            k0 = 2 * np.pi / self._lambda # wave number
            alpha = self.__rot_around_y_deg / 180 * math.pi # tilt angle in radians
            # positive y-rotation causes left reflection to move in negative x direction
            if self.__left_side_relevant:
                self.__tilt_mask_x_L =  np.exp(1j * k0 * x * math.tan(-alpha * 2))
            # positive y-rotation causes right reflection to move in positive x direction
            if self.__right_side_relevant:
                self.__tilt_mask_x_R =  np.exp(1j * k0 * x * math.tan(alpha * 2))
    
    def __calc_tilt_masks_y(self):
        # shift in y-direction (depends on x rotation)
        if self.__rot_around_x_deg == 0:
            self.__tilt_mask_y_L = 1
            self.__tilt_mask_y_R = 1
        else:
            ax = self.grid.axis_tot
            _, y = np.meshgrid(ax, ax)
            k0 = 2 * np.pi / self._lambda # wave number
            alpha = self.__rot_around_x_deg / 180 * math.pi # tilt angle in radians
            # positive x-rotation causes left reflection to move in positive y direction
            if self.__left_side_relevant:
                self.__tilt_mask_y_L =  np.exp(1j * k0 * y * math.tan(alpha * 2))
            # positive x-rotation causes right reflection to move in negative y direction
            if self.__right_side_relevant:
                self.__tilt_mask_y_R =  np.exp(1j * k0 * y * math.tan(-alpha * 2))
    
    @property
    def tilt_mask_x_L(self):
        """ returns the tilt-mask in x-direction on the left side """
        if self.__tilt_mask_x_L is None and self.__left_side_relevant:
            self.__calc_tilt_masks_x()
        return self.__tilt_mask_x_L
    
    @property
    def tilt_mask_x_R(self):
        """ returns the tilt-mask in x-direction on the right side """
        if self.__tilt_mask_x_R is None and self.__right_side_relevant:
            self.__calc_tilt_masks_x()
        return self.__tilt_mask_x_R
    
    @property
    def tilt_mask_y_L(self):
        """ returns the tilt-mask in y-direction on the left side """
        if self.__tilt_mask_y_L is None and self.__left_side_relevant:
            self.__calc_tilt_masks_y()
        return self.__tilt_mask_y_L
    
    @property
    def tilt_mask_y_R(self):
        """ returns the tilt-mask in y-direction on the right side """
        if self.__tilt_mask_y_R is None and self.__right_side_relevant:
            self.__calc_tilt_masks_y()
        return self.__tilt_mask_y_R
    
    @property
    def R1(self):
        """ power reflection top hald (left and right) """
        return self.__R1
    
    @R1.setter
    def R1(self, R1_new):
        if R1_new>1:
            print("Warning: R1 must be <= 1. Setting R1 to 1.")
            R1_new = 1
        elif R1_new<0:
            print("Warning: R1 must be >= 0. Setting R1 to 0.")
            R1_new = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__R1 = R1_new
        self.__T1 = 1 - R1_new
        self.__set_rt()
        
    @property
    def R2(self):
        """ power reflection bottom half (left and right) """
        return self.__R2
    
    @R2.setter
    def R2(self, R2_new):
        if R2_new>1:
            print("Warning: R2 must be <= 1. Setting R2 to 1.")
            R2_new = 1
        elif R2_new<0:
            print("Warning: R2 must be >= 0. Setting R2 to 0.")
            R2_new = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__R2 = R2_new
        self.__T2 = 1 - R2_new
        self.__set_rt()    
    
    @property
    def T1(self):
        """ power transmission top half (LTR and RTL) """
        return self.__T1
        
    @T1.setter
    def T1(self, T1_new):
        if T1_new>1:
            print("Warning: T1 must be <= 1. Setting T1 to 1.")
            T1_new = 1
        elif T1_new<0:
            print("Warning: T1 must be >= 0. Setting T1 to 0.")
            T1_new = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__T1 = T1_new
        self.__R1 = 1 - T1_new
        self.__set_rt()
 
    @property
    def T2(self):
        """ power transmission bottom half (LTR and RTL) """
        return self.__T2
        
    @T2.setter
    def T2(self, T2_new):
        if T2_new>1:
            print("Warning: T2 must be <= 1. Setting T2 to 1.")
            T2_new = 1
        elif T2_new<0:
            print("Warning: T2 must be >= 0. Setting T2 to 0.")
            T2_new = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__T2 = T2_new
        self.__R2 = 1 - T2_new
        self.__set_rt()   
 
    def set_T1_non_energy_conserving(self, T1):
        """" 
        allows to set an unphysical (non-energy-conerving) power transmissivity
        e.g. to simulate perfectly reflecting mirrors with R=1 which would
        cause the transfer matrix to become singular
        """
        if not self.cavity is None:
            self.cavity.clear()       
        self.__T1 = T1
        self.__set_rt()
        
    def set_T2_non_energy_conserving(self, T2):
        """" 
        allows to set an unphysical (non-energy-conerving) power transmissivity
        e.g. to simulate perfectly reflecting mirrors with R=1 which would
        cause the transfer matrix to become singular
        """
        if not self.cavity is None:
            self.cavity.clear()       
        self.__T2 = T2
        self.__set_rt()    
    
    @property
    def r1_L(self):
        return self.__r1 
            
    @property
    def r1_R(self):
        if self.__sym_phase:
            return self.__r1
        else:
            return -self.__r1
    
    @property
    def r2_L(self):
        return self.__r2 
            
    @property
    def r2_R(self):
        if self.__sym_phase:
            return self.__r2
        else:
            return -self.__r2
    
    @property
    def t1_LTR(self):
        return self.__t1 
    
    @property
    def t1_RTL(self):
        return self.__t1 

    @property
    def t2_LTR(self):
        return self.__t2 
    
    @property
    def t2_RTL(self):
        return self.__t2
        
    def set_phys_behavior(self, sym_phase: bool):
        """ 
        If sym_phase == True (default), then the mirror behaves
        physically insofar, as if it has the same reflection coefficient 
        on the left and right side, both will get the same phase. 
        If sym_phase == False, the mirror follows the convention that r_L, t_LTR 
        and t_RTL are real positive numbers, and r_R is a real negative number 
        """
        if not self.cavity is None:
            self.cavity.clear()       
        self.__sym_phase = sym_phase
        self.R1 =  self.__R1
        self.R2 =  self.__R2
        
    def __set_rt(self):
        if self.__sym_phase:
            self.__r1 = -self.__R1 - 1j * math.sqrt(self.__R1) * math.sqrt(1 - self.__R1)
            R1 = 1-self.__T1
            self.__t1 = 1-R1 - 1j * math.sqrt(R1) * math.sqrt(1 - R1)
            
            self.__r2 = -self.__R2 - 1j * math.sqrt(self.__R2) * math.sqrt(1 - self.__R2)
            R2 = 1-self.__T2
            self.__t2 = 1-R2 - 1j * math.sqrt(R2) * math.sqrt(1 - R2)
            
        else:
            self.__r1 = math.sqrt(self.__R1)
            self.__t1 = math.sqrt(self.__T1)
            self.__r2 = math.sqrt(self.__R2)
            self.__t2 = math.sqrt(self.__T2)

        self._R_L_mat_available = False
        self._RT_matrices.set_block(0, 0, 0)
        self._R_R_mat_available = False
        self._RT_matrices.set_block(1, 1, 0)
        self._T_mat_available = False
        self._RT_matrices.set_block(0, 1, 0)
    
    def __apply_r1r2(self, X, side: Side):
        """" 
        applies r1 to upper half and r2 to lower half of light field X
        X must be total resolution and in position space
        """
        if side == Side.LEFT:
            r1 = self.r1_L
            r2 = self.r2_L
        else:
            r1 = self.r1_R
            r2 = self.r2_R
    
        lines = X.shape[0]
        lines_h = lines//2
        if lines%2 == 0:
            # even number of lines
            X[:lines_h,:] *= r2
            X[lines_h:,:] *= r1
        else:
            rm = (r1+r2)/2
            X[:lines_h,:] *= r2
            X[lines_h,:] *= rm
            X[lines_h+1,:] *= r1
            
    def __apply_t1t2(self, X):
        """" 
        applies t1 to upper half and t2 to lower half of light field X
        X must be total resolution and in position space
        """
        
        t1 = self.__t1
        t2 = self.__t2
        
        lines = X.shape[0]
        lines_h = lines//2
        if lines%2 == 0:
            # even number of lines
            X[:lines_h,:] *= t2
            X[lines_h:,:] *= t1
        else:
            tm = (t1+t2)/2
            X[:lines_h,:] *= t2
            X[lines_h,:] *= tm
            X[lines_h+1,:] *= t1
            
    def _calc_T_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        self.__apply_t1t2(psi) 
        return self.grid.arr_to_vec(psi, False, False) 
    
    def calc_T_mat_tot(self):
        """
        Calulates the transmission matrix  
        """
        # simulate propagation for all modes in parallel    
        
        self.cavity.progress.push_print("calculating transmission matrix")
        NN = self.grid.res_tot**2
        
        if self.par_RT_calc:
            #########################################
            # Parallel version
            #########################################
            keep_clsBlockMatrix_alive(True)
            result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_T_mat_tot_helper)(i) for i in range(NN))
            keep_clsBlockMatrix_alive(False)
            T_mat_tot = np.column_stack(result)
            
            
        else:
            ##########################################
            # non-parallel version
            ##########################################
            T_mat_tot = np.empty((NN,NN), dtype=self.cavity.dtype_c)        
            i = 0
            self.cavity.progress.tic_reset(NN, True, "processing...")
            for nx, ny in self.grid.mode_numbers_tot:            
                psi = self.grid.fourier_basis_func(nx, ny, True, False)
                self.__apply_t1t2(psi)
                T_mat_tot[:,i] = self.grid.arr_to_vec(psi, False, False)
                i += 1
                self.cavity.progress.tic()
        
        self._RT_matrices.set_block(0, 1, T_mat_tot)
        self._T_mat_available = True
        T_mat_tot = None
        
        gc.collect()
        self.cavity.progress.pop()
        
    
    @property    
    def T_LTR_mat_tot(self):
        if not self._T_mat_available:
            self.calc_T_mat_tot()
        return self._RT_matrices.get_block(0, 1)

    @property    
    def T_RTL_mat_tot(self):
        if not self._T_mat_available:
            self.calc_T_mat_tot()
        return self._RT_matrices.get_block(0, 1) 
    
    def _calc_R_L_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        self.__apply_r1r2(psi, Side.LEFT) 
        if self.__left_side_relevant:
            if self.__rot_around_y_deg != 0:
                psi *= self.__tilt_mask_x_L 
            if self.__rot_around_x_deg != 0:
                psi *= self.__tilt_mask_y_L 
        return self.grid.arr_to_vec(psi, False, False) 
    
    def calc_R_L_mat_tot(self):
        """
        Calulates the left reflection matrix with mirror tilt
        """
        # simulate propagation for all modes in parallel    
        self.cavity.progress.push_print("calculating left reflection matrix R_L")
        if self.__tilt_mask_x_L is None or self.__tilt_mask_Y_L is None:
            if self.__rot_around_y_deg != 0:
                self.__calc_tilt_masks_x()
            if self.__rot_around_x_deg != 0:
                self.__calc_tilt_masks_y()
        NN = self.grid.res_tot**2
        
        if self.par_RT_calc:
            #########################################
            # Parallel version
            #########################################
            keep_clsBlockMatrix_alive(True)
            result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_R_L_mat_tot_helper)(i) for i in range(NN))
            keep_clsBlockMatrix_alive(False)
            R_L_mat_tot = np.column_stack(result)
        else:
            ##########################################
            # non-parallel version
            ##########################################
            R_L_mat_tot = np.empty((NN,NN), dtype=self.cavity.dtype_c)        
            i = 0
            self.cavity.progress.tic_reset(NN, True, "processing...")
            for nx, ny in self.grid.mode_numbers_tot:            
                psi = self.grid.fourier_basis_func(nx, ny, True, False)
                self.__apply_r1r2(psi, Side.LEFT) 
                if self.__right_side_relevant:
                    if self.__rot_around_y_deg != 0:
                        psi *= self.__tilt_mask_x_L 
                    if self.__rot_around_x_deg != 0:
                        psi *= self.__tilt_mask_y_L 
                R_L_mat_tot[:,i] = self.grid.arr_to_vec(psi, False, False)
                i += 1
                self.cavity.progress.tic()
        
        self._RT_matrices.set_block(0, 0, R_L_mat_tot)
        self._R_L_mat_available = True
        R_L_mat_tot = None
        
        self.cavity.progress.pop()
        gc.collect()
    
    def _calc_R_R_mat_tot_helper(self, i):
        gc.set_threshold(1, 1, 1)
        nx = self.grid.mode_numbers_tot[i,0]
        ny = self.grid.mode_numbers_tot[i,1]
        psi = self.grid.fourier_basis_func(nx, ny, True, False)
        self.__apply_r1r2(psi, Side.RIGHT) 
        if self.__right_side_relevant:
            if self.__rot_around_y_deg != 0:
                psi *= self.__tilt_mask_x_R 
            if self.__rot_around_x_deg != 0:
                psi *= self.__tilt_mask_y_R 
        return self.grid.arr_to_vec(psi, False, False) 
    
    def calc_R_R_mat_tot(self):
        """
        Calulates the left reflection matrix with mirror tilt
        """
        # simulate propagation for all modes in parallel        
        
        self.cavity.progress.push_print("calculating right reflection matrix R_R")
        
        
        if self.__tilt_mask_x_R is None or self.__tilt_mask_y_R is None:
            if self.__rot_around_y_deg != 0:
                self.__calc_tilt_masks_x()
            if self.__rot_around_x_deg != 0:
                self.__calc_tilt_masks_y()
            
        NN = self.grid.res_tot**2
        
        if self.par_RT_calc:
            #########################################
            # Parallel version
            #########################################
            keep_clsBlockMatrix_alive(True)
            result = Parallel(n_jobs=self.cavity.mp_pool_processes)(delayed(self._calc_R_R_mat_tot_helper)(i) for i in range(NN))
            keep_clsBlockMatrix_alive(False)
            R_R_mat_tot = np.column_stack(result)
        
        else:
            ##########################################
            # non-parallel version
            ##########################################
            R_R_mat_tot = np.empty((NN,NN), dtype=self.cavity.dtype_c)        
            i = 0
            self.cavity.progress.tic_reset(NN, True, "processing...")
            for nx, ny in self.grid.mode_numbers_tot:            
                psi = self.grid.fourier_basis_func(nx, ny, True, False)
                self.__apply_r1r2(psi, Side.RIGHT) 
                if self.__right_side_relevant:
                    if self.__rot_around_y_deg != 0:
                        psi *= self.__tilt_mask_x_R 
                    if self.__rot_around_x_deg != 0:
                        psi *= self.__tilt_mask_y_R 
                R_R_mat_tot[:,i] = self.grid.arr_to_vec(psi, False, False)
                i += 1
                self.cavity.progress.tic()
        
        self._RT_matrices.set_block(1, 1, R_R_mat_tot)
        self._R_R_mat_available = True
        R_R_mat_tot = None
        
        self.cavity.progress.pop()
        gc.collect()
    
    @property    
    def R_L_mat_tot(self):
        if not self._R_L_mat_available:
            self.calc_R_L_mat_tot()                    
        return self._RT_matrices.get_block(0, 0)
    
    @property    
    def R_R_mat_tot(self):
        if not self._R_R_mat_available:
            self.calc_R_R_mat_tot()
        return self._RT_matrices.get_block(1, 1)
      
    def prop(self, E_in, k_space_in, k_space_out, direction: Dir):
        """ 
        Propagates the input field E_in in positive z-direction (L->R) 
        by simply applying the t_LTR transmission factor
        """
        
        if k_space_in:
             # input in k-space
             # convert to position space
             E_in = ifft2_phys_spatial(E_in, self.grid.axis_tot)
        
        self.__apply_t1t2(E_in)
            
        if k_space_out:
            # but output should be in k-space
            E_in = fft2_phys_spatial(E_in, self.grid.axis_tot)

        return E_in
    
###############################################################################
# clsTaskManager
############################################################################### 
class clsTaskManager(ABC):
    def __init__(self, simulations: int, steps_per_simulation: int, folder):
        """" 
        simulations ... number of simulations to be performed
        steps_per_simulation ... number of steps per simulation
        """
        self.simulations = simulations
        self.steps_per_simulation = steps_per_simulation
        self.folder = folder
        self.sleep_time = 10
 
    @property
    def simulations(self):
        return self.__simulations
    
    @simulations.setter
    def simulations(self, simulations):
        if simulations<0:
            simulations = 1
        self.__simulations = simulations
    
    @property
    def steps_per_simulation(self):
        return self.__steps_per_simulation
    
    @steps_per_simulation.setter
    def steps_per_simulation(self, steps_per_simulation):
        if steps_per_simulation<0:
            steps_per_simulation = 1
        self.__steps_per_simulation = steps_per_simulation

    def get_next_task(self):
        for sim in range(0, self.simulations):
            for step in range (0, self.steps_per_simulation):
                grabbed, processing = self.grab_task(sim, step)
                if grabbed:
                    return sim, step
                if processing:
                    break
        
        print(f"Could not grab a task. Will sleep for {self.sleep_time} seconds.")
        time.sleep(self.sleep_time)
        return -1, -1
    
    def delete_all_files(self):
        for sim in range(0, self.simulations):
            for step in range (0, self.steps_per_simulation):
                filename_proc = str(sim)+"_"+str(step)+"_a.tmp"
                filename_done = str(sim)+"_"+str(step)+"_b.tmp"
                if self.folder != "":
                    filename_proc = self.folder + os.sep + filename_proc
                    filename_done = self.folder + os.sep + filename_done
                    
                if os.path.exists(filename_done):
                    os.remove(filename_done)
                if os.path.exists(filename_proc):
                    os.remove(filename_proc)
        
    def grab_task(self, sim:int, step:int):
        dummy = True
        
        sleep_time = np.random.rand()        
        time.sleep(sleep_time)
        
        if sim>=self.simulations:
            return False, False
        
        if step>=self.steps_per_simulation:
            return False, False
        
        filename_proc = str(sim)+"_"+str(step)+"_a.tmp"
        filename_done = str(sim)+"_"+str(step)+"_b.tmp"
        if self.folder != "":
            filename_proc = self.folder + os.sep + filename_proc
            filename_done = self.folder + os.sep + filename_done
            
        if os.path.exists(filename_done):
            # task already done
            grabbed = False; 
            processing = False;
            
        elif os.path.exists(filename_proc):
            # task currently processed by other thread
            grabbed = False; 
            processing = True;
            
        else:
            # task available; grab task
            with open(filename_proc, 'wb') as f:
                pickle.dump(dummy, f)
            grabbed = True;
            processing = True;
            
        return grabbed, processing

    def end_task(self, sim:int, step:int):
        dummy = True
        if sim>=self.simulations:
            return
        if step>=self.steps_per_simulation:
            return 
        
        filename_done = str(sim)+"_"+str(step)+"_b.tmp"
        if self.folder != "":
            filename_done = self.folder + os.sep + filename_done
        with open(filename_done, 'wb') as f:
                pickle.dump(dummy, f)
            
###############################################################################
# clsCavity
# represents a superclass for a cavity (one or two path)
############################################################################### 
class clsCavity(ABC):
    def __init__(self, name, full_precision = True):      
        gc.set_threshold(1, 1, 1)
        if full_precision:
            self.dtype_c = np.complex128 # complex data type
            self.dtype_r = np.float64 # complex data type
        else:
            self.dtype_c = np.complex64 # complex data type
            self.dtype_r = np.float32 # complex data type
        self.__R_L_mat_tot = None # whole cavity  reflection block matrix from left
        self.__R_R_mat_tot = None # whole cavity reflection block matrix from right
        self.__T_LTR_mat_tot = None # whole cavity left-to-right transmission block matrix
        self.__T_RTL_mat_tot = None # whole cavity right-to-left transmission block matrix
        self.__R_L_mat_fov = None # whole cavity reflection matrix from left
        self.__R_R_mat_fov = None # whole cavity reflection matrix from right
        self.__T_LTR_mat_fov = None # whole cavity left-to-right transmission matrix
        self.__T_RTL_mat_fov = None # whole cavity right-to-left transmission matrix
        self.__name = name
        self.M_bmat_tot = None # whole cavity transfer matrix (total resolution)
        self.S_bmat_tot = None # whole cavity scattering matrix (total resolution)
        self.progress = clsProgressPrinter()
        self.folder = ""
        self.tmp_folder = ""
        self.UID = 0
        self.additional_steps = 0
        self.pre_steps = 0
        self.__allow_temp_mem_caching = True # allow "spontneous" memory-caching
        self.__allow_temp_file_caching = False # allow "spontneous" file-caching
        self.__grid = clsGrid(self)
        self.__lambda_ref_nm = 633
        self.__lambda_ref = 633 / 1000000000
        self.__lambda_nm = 633
        self.__lambda = 633 / 1000000000   
        self.__components = []   
        self.__file_cache_min_calc_time = 3
        self.__mp_pool_processes = 10
        self.sep_char = ","
        self.single_step_mode = False
        
    def __del__(self):
        self.close_mp_pool()
       
    @abstractmethod
    def clear_results(self):
        pass
    
    @property
    @abstractmethod
    def use_bmatrix_class(self):
        """ Is clsBlockMatrix to be used? """
        pass
    
    @property
    @abstractmethod
    def use_swap_files_in_bmatrix_class(self):
        """ are temoprary files to usd in clsBlockMatrix to save RAM memory? """        
        pass
    
    @property
    def R_L_mat_tot(self):
        """ returns the reflection matrix from left for the whole cavity """
        if self.__R_L_mat_tot is None:
            self.calc_R_L_mat_tot()
        
        return self.__R_L_mat_tot
    
    @R_L_mat_tot.setter
    def R_L_mat_tot(self, R):
        self.__R_L_mat_tot = R
    
    @abstractmethod
    def calc_R_L_mat_tot(self):
        pass
    
    @property
    def R_R_mat_tot(self):
        """ returns the reflection matrix from right for the whole cavity """
        if self.__R_R_mat_tot is None:
            self.calc_R_R_mat_tot()
        
        return self.__R_R_mat_tot

    @R_R_mat_tot.setter
    def R_R_mat_tot(self, R):
        self.__R_R_mat_tot = R

    @abstractmethod
    def calc_R_R_mat_tot(self):
        pass

    @property
    def T_LTR_mat_tot(self):
        """ 
        returns the left-to-right transmission matrix for the whole cavity 
        (total resolution)
        """
        if self.__T_LTR_mat_tot is None:
            self.calc_T_LTR_mat_tot()
        
        return self.__T_LTR_mat_tot
    
    @T_LTR_mat_tot.setter
    def T_LTR_mat_tot(self, T):
        self.__T_LTR_mat_tot = T
 
    @abstractmethod
    def calc_T_LTR_mat_tot(self):
        pass   
 
    @property
    def T_RTL_mat_tot(self):
        """ 
        returns the left-to-right transmission matrix for the whole cavity 
        (total resolution)
        """
        if self.__T_RTL_mat_tot is None:
            self.calc_T_RTL_mat_tot()
        
        return self.__T_RTL_mat_tot    
    
    @T_RTL_mat_tot.setter
    def T_RTL_mat_tot(self, T):
        self.__T_RTL_mat_tot = T
 
    @abstractmethod
    def calc_T_RTL_mat_tot(self):
        pass      
 
    @property
    def R_L_mat_fov(self):
        """ returns the reflection matrix from left for the whole cavity """
        if self.__R_L_mat_fov is None:
            self.convert_R_L_mat_tot_to_fov()
        
        return self.__R_L_mat_fov
    
    @R_L_mat_fov.setter
    def R_L_mat_fov(self, R):
        self.__R_L_mat_fov = R
        
    @abstractmethod
    def convert_R_L_mat_tot_to_fov(self):
        pass
    
    @property
    def R_R_mat_fov(self):
        """ returns the reflection matrix from left for the whole cavity """
        if self.__R_R_mat_fov is None:
            self.convert_R_R_mat_tot_to_fov()
        
        return self.__R_R_mat_fov
    
    @R_R_mat_fov.setter
    def R_R_mat_fov(self, R):
        self.__R_R_mat_fov = R
    
    @abstractmethod
    def convert_R_R_mat_tot_to_fov(self):
        pass
    
    @property
    def T_LTR_mat_fov(self):
        """ 
        returns the left-to-right transmission matrix for the whole cavity 
        (FOV resolution)
        """
        if self.__T_LTR_mat_fov is None:
            self.convert_T_LTR_mat_tot_to_fov()
        
        return self.__T_LTR_mat_fov    
    
    @T_LTR_mat_fov.setter
    def T_LTR_mat_fov(self, T):
        self.__T_LTR_mat_fov = T
        
    @abstractmethod
    def convert_T_LTR_mat_tot_to_fov(self):
        pass
    
    @property
    def T_RTL_mat_fov(self):
        """ 
        returns the right-to-left transmission matrix for the whole cavity 
        (FOV resolution)
        """
        if self.__T_RTL_mat_fov is None:
            self.convert_T_RTL_mat_tot_to_fov()
        
        return self.__T_RTL_mat_fov       
    
    @T_RTL_mat_fov.setter
    def T_RTL_mat_fov(self, T):
        self.__T_RTL_mat_fov = T
    
    @abstractmethod
    def convert_T_RTL_mat_tot_to_fov(self):
        pass
    
    @property
    def components(self):
        return self.__components

    def get_last_component(self):
        if len(self.__components) == 0:
            return None
        else:
            return self.__components[-1]

    @property
    def allow_temp_mem_caching(self):
        return self.__allow_temp_mem_caching 
    
    @allow_temp_mem_caching.setter
    def allow_temp_mem_caching(self, x):
        self.__allow_temp_mem_caching = x
        if x:
            self.__allow_temp_file_caching = False
            
    @property
    def allow_temp_file_caching(self):
        return self.__allow_temp_file_caching 
    
    @allow_temp_file_caching.setter
    def allow_temp_file_caching(self, x):
        self.__allow_temp_file_caching = x
        if x:
            self.__allow_temp_mem_caching = False
            
    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name):
        self.__name = name
        
    @property 
    def file_cache_min_calc_time(self):
        """ 
        minimum time a calulation needs to take before a matrix is file-cached
        """
        return self.__file_cache_min_calc_time
    
    @file_cache_min_calc_time.setter
    def file_cache_min_calc_time(self, t):
        if t<0:
            t=0
        self.__file_cache_min_calc_time = t
    
    @property
    def grid(self) -> clsGrid:
        return self.__grid
    
    @property
    def Lambda_nm(self):
        """ returns exact wavelength in nm"""
        return self.__lambda_nm
    
    @Lambda_nm.setter    
    def Lambda_nm(self, Lambda_nm: float):
        """ sets reference wavelength in nm"""
        self.__lambda_nm = Lambda_nm
        self.__lambda = Lambda_nm / 1000000000
        for component in self.__components:
            component.Lambda_nm = Lambda_nm
        self.clear()
        
    @property
    def Lambda(self):
        """ returns exact wavelength in m """
        return self.__lambda
    
    @Lambda.setter    
    def Lambda(self, Lambda: float):
        """ sets exact wavelength in nm"""
        self.__lambda = Lambda
        self.__lambda_nm = Lambda * 1000000000   
        for component in self.__components:
            component.Lambda = Lambda
        self.clear()
        
    @property
    def Lambda_ref_nm(self):
        """ returns reference wavelength in nm"""
        return self.__lambda_ref_nm
    
    @Lambda_ref_nm.setter    
    def Lambda_ref_nm(self, x: float):
        """ sets reference wavelength in nm"""
        self.__lambda_ref_nm = x
        self.__lambda_ref = x / 1000000000
        
    @property
    def Lambda_ref(self):
        """ returns regference wavelength in m """
        return self.__lambda_ref
    
    @Lambda_ref.setter    
    def Lambda_ref(self, x: float):
        """ sets exact wavelength in nm"""
        self.__lambda_ref = x
        self.__lambda_ref_nm = x * 1000000000   
        
    def clear(self):
        self.__R_L_mat_tot = None # whole cavity  reflection block matrix from left
        self.__R_R_mat_tot = None # whole cavity reflection block matrix from right
        self.__T_LTR_mat_tot = None # whole cavity left-to-right transmission block matrix
        self.__T_RTL_mat_tot = None # whole cavity right-to-left transmission block matrix
        self.__R_L_mat_fov = None # whole cavity reflection matrix from left
        self.__R_R_mat_fov = None # whole cavity reflection matrix from right
        self.__T_LTR_mat_fov = None # whole cavity left-to-right transmission matrix
        self.__T_RTL_mat_fov = None # whole cavity right-to-left transmission matrix
        self.M_bmat_tot = None # whole cavity transfer matrix (total resolution)
        self.S_bmat_tot = None # whole cavity scattering matrix (total resolution)
        self.clear_results()
              
    @property
    def component_count(self):
        """ returns number of optical components """
        return len(self.__components)
    
    def get_component(self, i: int):
        """ returns the i-th optical component (starting with i=0) """
        if i<0:
            return None
        elif i>=len(self.__components):
            return None
        else:
            return self.__components[i]

    @property
    def total_steps(self):
        """ returns number of total steps required for calculation """
        return len(self.components) + self.additional_steps + self.pre_steps  

    @property
    def mp_pool_processes(self):
        """" number of parallel processes in the multiprocessing pool """
        return self.__mp_pool_processes
    
    @mp_pool_processes.setter
    def mp_pool_processes(self, processes: int):
        """" number of parallel processes in the multiprocessing pool """
        self.close_mp_pool()
        if processes < 1:
            self.__mp_pool_processes = 1
        else:
            self.__mp_pool_processes = processes
        
    def close_mp_pool(self):
        """ closes the multiprocessing pool """
        #clsPoolSingleton.reset_pool() 
        pass
      
    def write_to_file(self, file_name, data):
        """appends data to file. data must be a list. """
        if self.folder != "":
            file_name = self.folder + os.sep + file_name
        with portalocker.Lock(file_name, 'a') as f:
            # Write data to the file
            first = True;
            for item in data:
                if first:
                    f.write(str(item))
                    first = False
                else:
                    f.write(self.sep_char+" "+str(item))
            f.write('\n') 
 
    def _full_file_name(self, name, idx=-1, file_extension = "pkl", start_with_underscore = False):
        """ returns full filename for file"""
        if idx>=0:
            filename = str(self.UID)+"_"+str(idx)+"_"+name+"."+file_extension
        else:
            filename = str(self.UID)+"_"+name+"."+file_extension
        if start_with_underscore:
            filename = "_" + filename
        if self.folder != "":
            filename = self.folder + os.sep + filename
        return filename
    
    def save_mat(self, name, m, idx=-1, msg=""):
        """ saves the matrix m with filenmae based on name and idx """
        if msg == "":
            self.progress.push_print("saving "+name)
        else:
            self.progress.push_print(msg)
        filename = self._full_file_name(name,idx)
        if isinstance(m, clsBlockMatrix):
            m.keep_tmp_files = True
        joblib.dump(m, filename)
        self.progress.pop()            
            
    def load_mat(self, name, idx=-1, msg=""):
        if msg == "":
            self.progress.push_print("loading "+name)
        else:
            self.progress.push_print(msg)
        
        filename = self._full_file_name(name,idx)        
        m = joblib.load(filename)
        if isinstance(m, clsBlockMatrix):
            m.keep_tmp_files = True
        self.progress.pop()
        return m  
    
    def file_exists(self, name, idx=-1):
        filename = self._full_file_name(name,idx)
        return os.path.exists(filename)
    
    def delete_mat(self, name, idx=-1, msg=""):        
        """ deletes the matrix with filename based on name and idx """                        
        filename = self._full_file_name(name,idx)
        
        if os.path.exists(filename):
            # delete temporary cached matrix files from tmp folder, if neccessary
            if msg == "":
                self.progress.push_print("deleting "+name)
            else:
                self.progress.push_print(msg)
            if self.use_bmatrix_class and self.use_swap_files_in_bmatrix_class:
                m = joblib.load(filename)
                if isinstance(m, clsBlockMatrix):
                    m.keep_tmp_files = False
                    m.clear()
                    del m
                else:
                    m = None
        
            # now delete matrix file itself               
            os.remove(filename)
            self.progress.pop()
        
    def additional_step(self, step: int):
        """ 
        Additional steps after calulating M_bmat (e.g. calculating R_R, 
        T_LTR, etc.) to be implemented here (in a sub-class) 
        """
        pass
    
    def pre_step(self, step: int):
        """ 
        Preparing steps before calulating M_bmat (e.g. calculating big
        transfer matrices of components) 
        """
        pass
        
    def activate_component_file_cache(self, idx_from=0, idx_to=999)    :
        """
        activates file caching of the individual transfer matrices 
        for all or some components
        """
        idx_to += 1
        if idx_from<0:
            idx_from = 0        
        if idx_to > self.component_count:
            idx_to = self.component_count
        
        for idx in range(idx_from, idx_to):
            component = self.components[idx]
            component.file_cache_M_bmat = True

    def delete_cached_component_files(self):
        """
        deletes all cached individual component transfer matrices 
        """
        for component in self.components:
            component.delete_M_bmat_tot()
    
    def save_M_bmat(self):
        self.save_mat("M_bmat", self.M_bmat_tot, msg="saving transfer block matrix of cavity")   
                    
    def load_M_bmat(self):
        self.M_bmat_tot = self.load_mat("M_bmat", msg="loading transfer block matrix of cavity")    
        
    def M_bmat_file_exists(self):
        return self.file_exists("M_bmat")
        
    def _get_M_bmat_tot_considering_caching(self, step, idx_from, idx_to):
        """ Get M matrix from component considering all caching settings"""
        #keep_clsBlockMatrix_alive(True)
        #self.progress.push_print("sleeping")
        #time.sleep(30)
        #self.progress.pop()
        component = self.components[step]
        predefined_file_caching = component.file_cache_M_bmat
        predefined_mem_caching = component.mem_cache_M_bmat
        
        delete_cache_file = False
        delete_cache_mem = False
        
        # determine, whether the compnent was used before
        component_was_used_before = False
        if step>idx_from:
            component_was_used_before = component in self.components[idx_from:step]
        
        # determine, if component will be used again
        component_will_be_used_again = False
        component_will_be_used_again = component in self.components[step+1:]
        
        if predefined_mem_caching:
            self.progress.push_print("memory caching active for this component")
            self.progress.pop()
        
        if predefined_file_caching:
            self.progress.push_print("file caching active for this component")
            self.progress.pop()
        
        if (not predefined_mem_caching) and (not self.allow_temp_mem_caching):
            # neither pre-defined mem-caching, nor temporary mem-caching
            if not component_will_be_used_again:
                # component will not be re-used
                # delete everything from component's memory after use 
                delete_cache_mem = True
        
        if (not predefined_mem_caching) and (not predefined_file_caching):
            # neiter pre-defined memory-caching 
            # nor pre-defined file-caching for the current component
            
            if self.allow_temp_mem_caching:
                # temporary mem-caching allowed
                if component_will_be_used_again:
                    # component will be used again ->
                    # active memory caching flag 
                    self.progress.push_print("component will be used again: activating memory caching for component") 
                    component.mem_cache_M_bmat = True 
                    self.progress.pop()
                    
                elif component_was_used_before:
                    # component will not be used again, but was used before
                    # delete from memory after loaded and used
                    self.progress.push_print("component was used before and will not be used again: preparing memory cache to be deleted")
                    delete_cache_mem = True
                    self.progress.pop()
                    
                else:
                    delete_cache_mem = True
                    
            else:
                # no memory caching
                delete_cache_mem = True
                
            if self.allow_temp_file_caching:
                # temporary file-caching allowed
                if component_will_be_used_again:
                    # component will be used again ->
                    # active file caching flag 
                    self.progress.push_print("component will be used again: activating file caching for component") 
                    component.file_cache_M_bmat = True 
                    self.progress.pop()
                    
                elif component_was_used_before:
                    # component will not be used again, but was used before
                    # delete file after loaded and used
                    self.progress.push_print("component was used before and will not be used again:")
                    self.progress.print("- activating cached file to be loaded")
                    self.progress.print("- preparing cache file to be deleted after use")
                    component.file_cache_M_bmat = True
                    delete_cache_file = True
                    self.progress.pop()
                
        # ** 1 **
        M2 = component.M_bmat_tot           
                        
        component.file_cache_M_bmat = predefined_file_caching
        component.mem_cache_M_bmat = predefined_mem_caching
        
        if delete_cache_file:                
            cache_file_delete_component = component #.delete_M_bmat_tot()  
        else:
            cache_file_delete_component = None
            
        if delete_cache_mem:
            component.clear_mem_cache()
            
            
        #keep_clsBlockMatrix_alive(False)
        
        return M2, cache_file_delete_component
    
    def calc_M_bmat_tot(self, idx_from=0, idx_to=999):
        """ 
        calculates the transfer block matrix for the whole 
        or a part of the cavity (total resolution)
        incl idx_from, incl idx_to 
        """
        self.single_step_mode = False
        
        idx_to += 1
        if idx_from<0:
            idx_from = 0        
        if idx_to > self.component_count:
            idx_to = self.component_count
        
        component1 = self.get_component(idx_from)
        component2 = self.get_component(idx_to-1)
        
        if idx_from == 0 and idx_to == self.component_count:
            self.progress.push_print("calculating cavity's transfer matrix M")
        else:
            self.progress.push_print("calculating cavity's transfer matrix M " + \
                                     "between "+component1.name+" and "+component2.name)
        
        for step in range(idx_from, idx_to):
            component = self.components[step]
            self.progress.print("")
            self.progress.push_print("processing component "+str(step)+": "+component.name)
            
            
            # I don't understand why this dirty hack is neccessary, but it is
            # otherwise the temporary files of clsBlockMatrix 
            # self.M_bmat_tot get deleted
            flag = False
            if isinstance(self.M_bmat_tot, clsBlockMatrix):
                if not self.M_bmat_tot.keep_tmp_files:
                    self.M_bmat_tot.keep_tmp_files = True
                    flag = True                
            
            M2, del_component = self._get_M_bmat_tot_considering_caching(step, idx_from, idx_to)                        
            
            # Undoing the above step
            if flag:
                self.M_bmat_tot.keep_tmp_files = False
            
            self.progress.push_print("processing transfer matrix M")
            flag = False
            if self.M_bmat_tot is None:
                if isinstance(M2, clsBlockMatrix):
                    self.M_bmat_tot = clsBlockMatrix(M2.dim, self.use_swap_files_in_bmatrix_class, self.tmp_folder) 
                    self.M_bmat_tot.clone(M2)
                else:
                    self.M_bmat_tot = M2    
            
            else:            

                if isinstance(self.M_bmat_tot, clsBlockMatrix):
                    tmp = self.M_bmat_tot                    
                
                self.M_bmat_tot = bmat_mul(self.M_bmat_tot, M2, self.progress, "performing matrix multiplication")   
                
                if isinstance(self.M_bmat_tot, clsBlockMatrix):
                    tmp.keep_tmp_files = False
                    tmp.clear()
                    tmp = None
                    
                    
                    
            
            if not del_component is None:
                del_component.delete_M_bmat_tot()
            
            self.progress.pop()
            self.progress.pop()
        self.progress.pop()
        self.progress.print("")
 
    
    def single_step(self, step: int, reverse_mul = False):
        """ 
        performing only one single step (one component) in calculating
        the transfer block matrix for the whole cavity (total resolution)
        Loading and saving the result before and after
        and doing the additional steps
        """
        
        self.single_step_mode = True
        
        if self.total_steps == 0:
            return False
        
        if step<0:
            step = 0
        
        if step>=self.total_steps:
            return False
            
        
        self.progress.push_print("UID: "+str(self.UID)+", single step "+str(step))
        
        if step < self.pre_steps:
            # processing pre-steps
            self.progress.push_print("processing pre-step "+str(step))
            self.pre_step(step)
            
        elif step < self.component_count + self.pre_steps:     
            # processing main steps
            s1 = step - self.pre_steps
            component = self.components[s1]
            self.progress.push_print("processing component "+str(s1)+": "+component.name)
            
            #if s1>0:
            if self.M_bmat_file_exists():
                #  not the first component
                self.load_M_bmat()                
                
            M2, del_component = self._get_M_bmat_tot_considering_caching(s1, 0, self.component_count)
                       
            #print("self.M_bmat_tot.file_names")
            #if self.M_bmat_tot is None:
            #    print ("None")
            #else:
            #    print(self.M_bmat_tot.file_names)
            #print("self.M2.file_names")
            #print(M2.file_names)
                         
            self.progress.push_print("processing transfer matrix M")
            if self.M_bmat_tot is None:
                if isinstance(M2, clsBlockMatrix):
                    self.M_bmat_tot = clsBlockMatrix(M2.dim, self.use_swap_files_in_bmatrix_class, self.tmp_folder) 
                    self.M_bmat_tot.clone(M2)
                else:
                    self.M_bmat_tot = M2    # schasi
            else:     
                if isinstance(self.M_bmat_tot, clsBlockMatrix):
                    tmp = self.M_bmat_tot
                    
                if reverse_mul:
                    self.M_bmat_tot = bmat_mul(M2, self.M_bmat_tot, self.progress, "performing matrix multiplication") 
                else:
                    self.M_bmat_tot = bmat_mul(self.M_bmat_tot, M2, self.progress, "performing matrix multiplication")   
                
                if isinstance(self.M_bmat_tot, clsBlockMatrix):
                    tmp.keep_tmp_files = False
                    tmp.clear()
                    tmp = None
            
            if not del_component is None:
                del_component.delete_M_bmat_tot()
            
            self.progress.pop()
            self.save_M_bmat()
              
            
        else:
            # process additional steps
            s2 = step - self.component_count - self.pre_steps
            self.progress.push_print("processing additional step "+str(s2))
            self.load_M_bmat()
            self.additional_step(s2)
        
        self.progress.pop()
        self.progress.pop()
        
        return True 
    
    def resonance_data_simple_cavity(self, R_left, R_right, length_opt, 
                                     sym_phase = True):
        """ 
        wavelength,  wavenumber and longitudinal mode number
        and the period in wavelength
        for a simple resonator with optical length length_opt 
        """
        twopi = 2 * math.pi 
        
        if sym_phase:
            l_mode = math.atan(math.sqrt((1 - R_left) / R_left)) / twopi
            l_mode += math.atan(math.sqrt((1 - R_right) / R_right)) / twopi
            l_mode += 2*length_opt / self.Lambda      
        else:
            l_mode = 2*length_opt / self.Lambda - 1/2  
        
        l_mode = round(l_mode)
        
        
        
        if sym_phase:
            k_c = 2*math.pi*l_mode 
            k_c -= math.atan(math.sqrt((1-R_left)/R_left))
            k_c -= math.atan(math.sqrt((1-R_right)/R_right))
            k_c /= (2*length_opt)
    
            lambda_c = twopi / k_c
            
            lambda_period = 2*length_opt / l_mode - 2*length_opt / (l_mode + 1)
            
        else:
            lambda_c = 4*length_opt / (2*l_mode+1)
            lambda_c2 = 4*length_opt / (2*(l_mode+1)+1)
            k_c = twopi / lambda_c
            lambda_period = lambda_c - lambda_c2
        
        
        
        return lambda_c, k_c, l_mode, lambda_period
    
    def resonance_data_8f_cavity(self, R_left, R_center, R_right, f,
                                 sym_phase = True):
        """
        returns the length correction for the length of the second
        sub-cavity in a 8f cavity due to the phase-behavior of the mirrors
        mirr1_id, mirr2_ix are the indices of the first and the central mirror
        R_right is the reflection of the rightmost mirror
        """
        lambda_c, k_c, n, lambda_period = \
            self.resonance_data_simple_cavity(R_left, R_center, 4*f, sym_phase)
        if lambda_c == -1:
            return 0
        
        if sym_phase:
            l_corr = math.atan(math.sqrt(1 - R_left) / R_left) 
            l_corr -= math.atan(math.sqrt(1 - R_center) / R_center)
            l_corr -= math.atan(math.sqrt(1 - R_right) / math.sqrt(R_right))
            l_corr /= (2*k_c)
        else:
            l_corr = 0
        
        return lambda_c, k_c, n, lambda_period, l_corr

###############################################################################
# clsCavity1path
# represents a cavity with one path
############################################################################### 
class clsCavity1path(clsCavity):
    """ Represents a cavity with 1 optical path """
    def __init__(self, name, full_precision = True):
        super().__init__(name, full_precision)
        self.__incident_field_left = None
        self.__incident_field_right = None
        self.__output_field_left = None
        self.__output_field_right = None
        self.__bulk_field_LTR = None
        self.__bulk_field_RTL = None
        self.__bulk_field_pos = None
        self.__use_bmatrix_class = True
        self.__use_swap_files_in_bmatrix_class = True
    
    def clear_results(self):
        self.__output_field_left = None
        self.__output_field_right = None
        self.__bulk_field_LTR = None
        self.__bulk_field_RTL = None
        self.__bulk_field_pos = None
         
    @property
    def use_bmatrix_class(self):
        """ Is clsBlockMatrix to be used? """
        return self.__use_bmatrix_class
    
    @use_bmatrix_class.setter
    def use_bmatrix_class(self, x):
        """ Is clsBlockMatrix to be used? """
        self.__use_bmatrix_class = x
        if not x:
            self.__use_swap_files_in_bmatrix_class = False
    
    @property
    def use_swap_files_in_bmatrix_class(self):
        """ are temoprary files to usd in clsBlockMatrix to save RAM memory? """        
        return self.__use_swap_files_in_bmatrix_class
    
    @use_swap_files_in_bmatrix_class.setter
    def use_swap_files_in_bmatrix_class(self, x):
        """ Is clsBlockMatrix to be used? """
        if self.__use_bmatrix_class:
            self.__use_swap_files_in_bmatrix_class = x
        else:
            self.__use_swap_files_in_bmatrix_class = False
    
    @property
    def bulk_field_LTR(self):
        """ returns the left-to-right bulk field """
        return self.__bulk_field_LTR
    
    @property
    def bulk_field_RTL(self):
        """ returns the right-to-left bulk field """
        return self.__bulk_field_RTL
    
    @property
    def bulk_field(self):
        """ returns the total bulk field (sum of RTL and LTR) """
        field = clsLightField(self.grid)
        if not self.__bulk_field_LTR is None:
            field.add(self.__bulk_field_LTR)
        if not self.__bulk_field_RTL is None:
            field.add(self.__bulk_field_RTL) 
        return field
    
    @property
    def bulk_field_pos(self):
        return self.__bulk_field_pos
    
    def calc_bulk_field_from_left(self, idx: int):
        """ 
        Calulates the bulk field on the right side
        of the component identified with idx
        based on the left incident and output field
        """
        self.progress.push_print("calculating bulk field right of component "
                                 +str(idx)+", coming from the left side")
        if self.component_count == 0:
            self.progress.print("There are no components! End task.")
            self.__bulk_field_LTR = None
            self.__bulk_field_RTL = None
            self.__bulk_field_pos = None
            return
        
        if idx >= self.component_count-1:
            # right from rightmost component
            # "bulk field" is the field on the right side
            self.progress.print("'Right of index "+str(idx)+"' "+ 
                                "is outside the bulk on the right side. "+
                                "Calculating field on the right side of cavity.")
            self.__bulk_field_LTR = self.output_field_right
            self.__bulk_field_RTL = self.incident_field_right
            if self.__bulk_field_LTR is None:
                self.__bulk_field_LTR = clsLightField(self.grid)
            if self.__bulk_field_RTL is None:
                self.__bulk_field_RTL = clsLightField(self.grid)
            self.__bulk_field_pos = self.component_count-1
            return
        
        if idx < 0:
            # right from component with idx -1 or smaller ->
            # "bulk field" is the field on the left side
            self.progress.print("'Right of index "+str(idx)+ +"' "+ 
                                "is outside the bulk on the left side. "+
                                "Calculating field on the left side of cavity.")
            self.__bulk_field_LTR = self.incident_field_left
            self.__bulk_field_RTL = self.output_field_left
            if self.__bulk_field_LTR is None:
                self.__bulk_field_LTR = clsLightField(self.grid)
            if self.__bulk_field_RTL is None:
                self.__bulk_field_RTL = clsLightField(self.grid)
            self.__bulk_field_pos = -1
            return
    
        for i in range(idx, -1,-1):
            component = self.components[i]
            self.progress.push_print("processing component "+str(i)+": "+component.name)
            if i==idx:
                invM = component.inv_M_bmat_tot
            else:
                invM = bmat_mul(invM, component.inv_M_bmat_tot)        
            self.progress.pop()
    
        # calculate bulk field right-to-left    
        self.progress.push_print("calculating right-to-left bulk field.")
        self.__bulk_field_RTL = clsLightField(self.grid)
        if not self.output_field_left is None:
            if isinstance(invM, clsBlockMatrix):
                self.__bulk_field_RTL.add(
                    self.output_field_left.apply_TR_mat(invM.get_block(0,0)))
            else:
                self.__bulk_field_RTL.add(
                    self.output_field_left.apply_TR_mat(invM[0][0]))
        
        if not self.incident_field_left is None:
            if isinstance(invM, clsBlockMatrix):
                self.__bulk_field_RTL.add(
                    self.incident_field_left.apply_TR_mat(invM.get_block(0,1)))
            else:
                self.__bulk_field_RTL.add(
                    self.incident_field_left.apply_TR_mat(invM[0][1]))
        self.progress.pop()
        
        # calculate bulk field left-to-right    
        self.progress.push_print("calculating left-to-right bulk field.")
        self.__bulk_field_LTR = clsLightField(self.grid)
        if not self.output_field_left is None:
            if isinstance(invM, clsBlockMatrix):
                self.__bulk_field_LTR.add(
                    self.output_field_left.apply_TR_mat(invM.get_block(1,0)))
            else:
                self.__bulk_field_LTR.add(
                    self.output_field_left.apply_TR_mat(invM[1][0]))
        
        if not self.incident_field_left is None:
            if isinstance(invM, clsBlockMatrix):
                self.__bulk_field_LTR.add(
                    self.incident_field_left.apply_TR_mat(invM.get_block(1,1)))
            else:
                self.__bulk_field_LTR.add(
                    self.incident_field_left.apply_TR_mat(invM[1][1]))
            
        self.__bulk_field_pos = idx
        self.progress.pop()
        
        self.progress.pop()
    
        
    def calc_bulk_field_from_right(self, idx: int):
        """ 
        Calulates the bulk field right of component identified with idx
        based on the right incident and output field
        """
        self.progress.push_print("calculating bulk field right of component "
                                 +str(idx)+", coming from the right side")
        if self.component_count == 0:
            self.progress.print("There are no components! End task.")
            self.__bulk_field_LTR = None
            self.__bulk_field_RTL = None
            self.__bulk_field_pos = None
            return
        
        if idx >= self.component_count-1:
            # right from rightmost component
            # "bulk field" is the field on the right side
            self.progress.print("'Right of index "+str(idx)+"' "+ 
                                "is outside the bulk on the right side. "+
                                "Calculating field on the right side of cavity.")
            self.__bulk_field_LTR = self.output_field_right
            self.__bulk_field_RTL = self.incident_field_right
            if self.__bulk_field_LTR is None:
                self.__bulk_field_LTR = clsLightField(self.grid)
            if self.__bulk_field_RTL is None:
                self.__bulk_field_RTL = clsLightField(self.grid)
            self.__bulk_field_pos = self.component_count-1
            return
        
        if idx < 0:
            # right from component with idx -1 or smaller ->
            # "bulk field" is the field on the left side
            self.progress.print("'Right of index "+str(idx)+ "' "+ 
                                "is outside the bulk on the left side. "+
                                "Calculating field on the left side of cavity.")
            self.__bulk_field_LTR = self.incident_field_left
            self.__bulk_field_RTL = self.output_field_left
            if self.__bulk_field_LTR is None:
                self.__bulk_field_LTR = clsLightField(self.grid)
            if self.__bulk_field_RTL is None:
                self.__bulk_field_RTL = clsLightField(self.grid)
            self.__bulk_field_pos = -1
            return
    
        for i in range(idx+1, self.component_count):
            component = self.components[i]
            self.progress.push_print("processing component "+str(i)+": "+component.name)
            if i==idx+1:
                M = component.M_bmat_tot
            else:
                M = bmat_mul(M, component.M_bmat_tot)        
            self.progress.pop()
    
        # calculate bulk field right-to-left    
        self.progress.push_print("calculating right-to-left bulk field.")
        self.__bulk_field_RTL = clsLightField(self.grid)
        if not self.incident_field_right is None:
            if isinstance(M, clsBlockMatrix):
                self.__bulk_field_RTL.add(
                    self.incident_field_right.apply_TR_mat(M.get_block(0,0)))
            else:
                self.__bulk_field_RTL.add(
                    self.incident_field_right.apply_TR_mat(M[0][0]))
        
        if not self.output_field_right is None:
            if isinstance(M, clsBlockMatrix):
                self.__bulk_field_RTL.add(
                    self.output_field_right.apply_TR_mat(M.get_block(0,1)))
            else:
                self.__bulk_field_RTL.add(
                    self.output_field_right.apply_TR_mat(M[0][1]))
        self.progress.pop()
        
        # calculate bulk field left-to-right    
        self.progress.push_print("calculating left-to-right bulk field.")
        self.__bulk_field_LTR = clsLightField(self.grid)
        if not self.incident_field_right is None:
            if isinstance(M, clsBlockMatrix):
                self.__bulk_field_LTR.add(
                    self.incident_field_right.apply_TR_mat(M.get_block(1,0)))
            else:
                self.__bulk_field_LTR.add(
                    self.incident_field_right.apply_TR_mat(M[1][0]))
        
        if not self.output_field_right is None:
            if isinstance(M, clsBlockMatrix):
                self.__bulk_field_LTR.add(
                    self.output_field_right.apply_TR_mat(M.get_block(1,1)))
            else:
                self.__bulk_field_LTR.add(
                    self.output_field_right.apply_TR_mat(M[1][1]))
            
        self.__bulk_field_pos = idx
        self.progress.pop()
        
        self.progress.pop()
    
    @property
    def incident_field_left(self):
        return self.__incident_field_left
    
    @incident_field_left.setter
    def incident_field_left(self, field):
        self.__incident_field_left = field
        self.__output_field_left = None
        self.__output_field_right = None
        self.__bulk_field_LTR = None
        self.__bulk_field_RTL = None
        self.__bulk_field_pos = None   
        
    @property
    def incident_field_right(self):
        return self.__incident_field_right
    
    @incident_field_right.setter
    def incident_field_right(self, field):
        self.__incident_field_right = field
        self.__output_field_left = None
        self.__output_field_right = None    
        self.__bulk_field_LTR = None
        self.__bulk_field_RTL = None
        self.__bulk_field_pos = None   
    
    @property
    def output_field_left(self):
        if self.__output_field_left is None:
            self.calc_output_field_left()
        return self.__output_field_left
    
    def calc_output_field_left(self):
        """ 
        calculates the left output field 
        based on the left and right input fields
        """
        self.__output_field_left = clsLightField(self.grid)
        self.__output_field_left.name = "Output Field Left"
        if not self.__incident_field_left is None:
            self.__output_field_left.add(
                self.incident_field_left.apply_TR_mat(self.R_L_mat_tot))
        
        if not self.__incident_field_right is None:
            self.__output_field_left.add(
                self.incident_field_right.apply_TR_mat(self.T_RTL_mat_tot))
                    
    @property
    def output_field_right(self):
        if self.__output_field_right is None:
            self.calc_output_field_right()
        return self.__output_field_right
    
    def calc_output_field_right(self):
        """ 
        calculates the right output field 
        based on the left and right input fields
        """
        self.__output_field_right = clsLightField(self.grid)
        self.__output_field_right.name = "Output Field Right"
        if not self.__incident_field_right is None:
            self.__output_field_right.add(
                self.incident_field_right.apply_TR_mat(self.R_R_mat_tot))
        
        if not self.__incident_field_left is None:
            self.__output_field_right.add(
                self.incident_field_left.apply_TR_mat(self.T_LTR_mat_tot))

    def save_R_L_mat_tot(self):
        self.save_mat("R_L_mat_tot", self.R_L_mat_tot)        
            
    def load_R_L_mat_tot(self):
        self.R_L_mat_tot = self.load_mat("R_L_mat_tot")        
            
    def save_R_L_mat_fov(self):
        self.save_mat("R_L_mat_fov", self.R_L_mat_fov)

    def load_R_L_mat_fov(self):
        self.R_L_mat_fov = self.load_mat("R_L_mat_fov")        
            
    def save_T_LTR_mat_tot(self):
        self.save_mat("T_LTR_mat_tot", self.T_LTR_mat_tot)
            
    def load_T_LTR_mat_tot(self):
        self.T_LTR_mat_tot = self.load_mat("T_LTR_mat_tot")        
            
    def save_T_LTR_mat_fov(self):
       self.save_mat("T_LTR_mat_fov", self.T_LTR_mat_fov)
            
    def load_T_LTR_mat_fov(self):
        self.T_LTR_mat_fov = self.load_mat("T_LTR_mat_fov")     
        
    def save_R_R_mat_tot(self):
        self.save_mat("R_R_mat_tot", self.R_R_mat_tot)        
            
    def load_R_R_mat_tot(self):
        self.R_R_mat_tot = self.load_mat("R_R_mat_tot")        
            
    def save_R_R_mat_fov(self):
        self.save_mat("R_R_mat_fov", self.R_R_mat_fov)
    
    def load_R_R_mat_fov(self):
        self.R_R_mat_fov = self.load_mat("R_R_mat_fov")        
            
    def save_T_RTL_mat_tot(self):
        self.save_mat("T_RTL_mat_tot", self.T_RTL_mat_tot)
            
    def load_T_RTL_mat_tot(self):
        self.T_RTL_mat_tot = self.load_mat("T_RTL_mat_tot")        
            
    def save_T_RTL_mat_fov(self):
       self.save_mat("T_RTL_mat_fov", self.T_RTL_mat_fov)
            
    def load_T_RTL_mat_fov(self):
        self.T_RTL_mat_fov = self.load_mat("T_RTL_mat_fov")            
    
    def calc_S_bmat_tot(self, idx_from=0, idx_to=999):
        """ 
        calculates the scattering block matrix for the whole cavity 
        (total resolution)
        """ 
        self.progress.push_print("calculating scattering block matrix S for the whole cavity")
        if self.use_bmatrix_class:
            self.S_bmat_tot = clsBlockMatrix(2, 
                                             self.use_swap_files_in_bmatrix_class,
                                             self.tmp_folder) 
            
            self.S_bmat_tot.set_block(0, 0, self.R_L_mat_tot)
            self.S_bmat_tot.set_block(0, 1, self.T_RTL_mat_tot)
            self.S_bmat_tot.set_block(1, 0, self.T_LTR_mat_tot)
            self.S_bmat_tot.set_block(1, 1, self.R_R_mat_tot)
        
        else:
            S11 = self.R_L_mat_tot
            S12 = self.T_RTL_mat_tot
            S21 = self.T_LTR_mat_tot
            S22 = self.R_R_mat_tot
            self.progress.pop()
            self.S_bmat_tot = [[S11, S12],[S21, S22]]

    def get_dist_phys(self, idx1=0, idx2=999):
        """ 
        returns the physical distance in meters between 
        the left side of the component identified by idx1
        and the right side of the component identified by idx2
        """
        
        idx2 += 1
        if idx1<0:
            idx1 = 0
        if idx2>self.component_count:
            idx2 = self.component_count
            
        dist = 0
        for i in range(idx1, idx2):
            component = self.components[i]
            dist += component.dist_phys
        
        return dist
    
    def get_dist_opt(self, idx1=0, idx2=999):
        """ 
        returns the optical distance in meters between 
        the left side of the component identified by idx1
        and the right side of the component identified by idx2
        """
        
        idx2 += 1
        if idx1<0:
            idx1 = 0
        if idx2>self.component_count:
            idx2 = self.component_count
            
        dist = 0
        for i in range(idx1, idx2):
            component = self.components[i]
            dist += component.dist_opt
        
        return dist
                
    def add_component(self, component: clsOptComponent):
        """ add an optical component (mirror, lens, or propagation, etc) """
        self.components.append(component)
        component._connect_to_cavity(self, self.grid, self.component_count-1)
        
    def prop(self, field: clsLightField, idx_from, idx_to, direction: Dir):
        """ 
        one-way-propagation of lightfield E_in 
        from left to right (direction = Dir.LTR)
        or frm right-to-left (direction = Dir.RTL)
        from optical component indicated by idx_from (incl) 
        to optical compenent indicated by idx_to (incl) 
        """
        
        
        prv_k_sp_out = field.k_space
        E_out = field.get_field_tot(prv_k_sp_out) # false
        if is_matrix(E_out):
            E_out = E_out.copy()
        
        if direction == Dir.LTR:
            # left-to-right propagation
            if idx_from > idx_to:
                idx_from, idx_to = idx_to, idx_from
                
            if idx_from<0:
                idx_from = 0
            
            if idx_to>=self.component_count:
                idx_to  = self.component_count-1
                
            idx_last = idx_to
            idx_to += 1
            idx_dir = 1
            
        else:
            # right-to-left propagation
            if idx_from < idx_to:
                idx_from, idx_to = idx_to, idx_from
                
            if idx_to<0:
                idx_to = 0
            
            if idx_from>=self.component_count:
                idx_from  = self.component_count-1
                
            idx_last = idx_to
            idx_to -= 1
            idx_dir = -1
            
        
        #prv_k_sp_out = field.k_space
        for idx in range(idx_from, idx_to, idx_dir):
            # determine if it is better to "interface" in k-space
            # or position space
            if idx == idx_last:
                nxt_k_sp_in_pref = False
                nxt_k_sp_in_dont_care = True
            else:
                if direction == Dir.LTR:
                    nxt_component = self.get_component(idx+1)
                else:
                    nxt_component = self.get_component(idx-1)
                nxt_k_sp_in_pref = nxt_component.k_space_in_prefer
                nxt_k_sp_in_dont_care = nxt_component.k_space_in_dont_care
            
            component = self.get_component(idx)
            # print("processing component",idx,":",component.name) 
            self.progress.print(f"processing component {idx}: {component.name}")
            # by default: position-space as output for current component
            k_space_out = False
            if component.k_space_out_dont_care:
                # current component output does not care
                if nxt_k_sp_in_pref:
                    # next component input prefers k-space:
                    # set output to k-space
                    k_space_out = True
            elif component.k_space_out_prefer:
                # current component output prefers k-space
                if nxt_k_sp_in_dont_care or nxt_k_sp_in_pref:
                    # next component input does not care or prefers k-space:
                    # set output to k-space
                    k_space_out = True
            else:
                # current component output prefers pos-space
                if nxt_k_sp_in_pref:
                    # but next component input prefers k-space:
                    # set output to k-space
                    k_space_out = True
                    
            E_out = component.prop(E_out, prv_k_sp_out, k_space_out, direction)
            prv_k_sp_out = k_space_out
        
        field_out = clsLightField(self.grid) 
        field_out.set_field(E_out, k_space_out)
        return field_out
 
    def get_T_mat_LTR(self, idx_from, idx_to):
        """
        Returns the transmission matrix for a 
        left-to-right trip 
        from optical component indicated by idx1 (incl) 
        to optical compenent indicated by idx2 (incl)         
        """
        if self.component_count == 0:
            # no components: return unity
            return 1
        
        if idx_from > idx_to:
                idx_from, idx_to = idx_to, idx_from
                
        if idx_from<0:
            idx_from = 0
        
        if idx_to>=self.component_count:
            idx_to  = self.component_count-1
            
        
        component1 = self.get_component(idx_from)
        name1 = component1.name
        component2 = self.get_component(idx_to)
        name2 = component2.name
        self.progress.push_print("Calculating LTR trip between left side of "+ \
                                 name1+" and right side of "+name2)
        
        T = 1
        for idx in range(idx_from, idx_to + 1):
            component = self.get_component(idx)
            self.progress.push_print("processing component "+str(idx)+": "+component.name)
            T = mat_mul(T, component.T_LTR_mat_tot)
            self.progress.pop()
        self.progress.pop()        
        
        return T   
    
    def get_T_mat_RTL(self, idx_from, idx_to):
        """
        Returns the transmission matrix for a 
        right-to-left trip 
        from optical component indicated by idx1 (incl) 
        to optical compenent indicated by idx2 (incl)         
        """
        if self.component_count == 0:
            # no components: return unity
            return 1
        
        if idx_from < idx_to:
                idx_from, idx_to = idx_to, idx_from
                
        if idx_to<0:
            idx_to = 0
        
        if idx_from>=self.component_count:
            idx_from  = self.component_count-1                            
            
        
        component1 = self.get_component(idx_from)
        name1 = component1.name
        component2 = self.get_component(idx_to)
        name2 = component2.name
        self.progress.push_print("Calculating RTL trip between right side of "+ \
                                 name1+" and left side of "+name2)
        
        T = 1
        for idx in range(idx_from, idx_to - 1, -1):
            component = self.get_component(idx)
            self.progress.push_print("processing component "+str(idx)+": "+component.name)
            T = mat_mul(T, component.T_RTL_mat_tot)
            self.progress.pop()
        self.progress.pop()        
        
        return T   
 
    def prop_mult_round_trips_LTR_RTL(self, no_of_trips, bulk_field_pos = 0, 
                                      print_progress = False,
                                      plot_left_out_progress = False,
                                      plot_right_out_progress = False,
                                      plot_bulk_LTR_progress = False,
                                      plot_bulk_RTL_progress = False,
                                      idx1 = 0, idx2 = 999,
                                      x_plot_shift = 0, y_plot_shift = 0):
        """
        Multiple LTR-RTL round-trips of field
        It is assumed that the left incident field enteres the cavity LTR 
        through the component with idx1 and the right incident field enteres 
        the cavity RTL through the component with idx2 . 
        Thecinside of the avity ends at the left side of the component with idx2.
        The effective left and right output fields are set.
        The bulk field on the right side of component with bulk_field_pos 
        is stored in .bulk_field_LTR and .bulk_field_RTL
        """
        self.__output_field_left = None
        self.__output_field_right = None
        self.__bulk_field_LTR = None
        self.__bulk_field_RTL = None
        
        if self.component_count == 0:
            # no components: do nothing
            return
        
        # idx1 supposed to be smaller or equal
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        if idx1 < 0:
            idx1 = 0
        if idx1 >= self.component_count:
            idx1 = self.component_count-1
        if idx2 < 0:
            idx2 = 0   
        if idx2 >= self.component_count:
            idx2 = self.component_count-1
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
            
        if bulk_field_pos < idx1-1:
            bulk_field_pos = idx1 - 1
        if bulk_field_pos > idx2:
            bulk_field_pos = idx2            
        self.__bulk_field_pos = bulk_field_pos

        component1 = self.get_component(idx1)
        name1 = component1.name
        component2 = self.get_component(idx2)
        name2 = component2.name
        
        if self.incident_field_left is None and self.incident_field_right is None:
            # no incident fields: Black output
            self.__output_field_left = clsLightField(self.grid)
            self.__output_field_right = clsLightField(self.grid)
            self.__bulk_field_LTR = clsLightField(self.grid)
            self.__bulk_field_RTL = clsLightField(self.grid)
            return
        
        self.progress.push_print("Calculating multiple LTR-RTL round-trips between "+ \
                                 name1+" and "+name2)
        silent_setting = self.progress.silent
        self.progress.silent = True
            
        # calculate left outer reflection at leftmost component
        if self.__incident_field_left is None:
            L_refl_at_component_1 = clsLightField(self.grid)
        else:
            L_refl_at_component_1 = self.reflect(
                self.__incident_field_left, idx1, Side.LEFT)
            
        # calculate right outer reflection at rightmost component
        if self.__incident_field_right is None:
            R_refl_at_component_2 = clsLightField(self.grid)
        else:
            R_refl_at_component_2 = self.reflect(
                self.__incident_field_right, idx2, Side.RIGHT)

        # left-to-right transmission through leftmost component
        if self.__incident_field_left is None:
            LTR_transm_throu_component_1 = clsLightField(self.grid)
        else:
            LTR_transm_throu_component_1 = self.prop(
                self.__incident_field_left, idx1, idx1, Dir.LTR) 
            
        # right-to-left transmission through rightmost component
        if self.__incident_field_right is None:
            RTL_transm_throu_component_2 = clsLightField(self.grid)
        else:
            RTL_transm_throu_component_2 = self.prop(
                self.__incident_field_right, idx2, idx2, Dir.RTL)
        
        if idx1 == idx2:
            # THERE IS ONLY ONE COMPONENT TO CONSIDER, NOT REALLY A
            # CAVITY FOR WHICH WE CAN CALCULATE ROUND-TRIPS
            self.__output_field_right = LTR_transm_throu_component_1
            self.__output_field_right.add(R_refl_at_component_2)
            
            self.__output_field_left = RTL_transm_throu_component_2
            self.__output_field_left.add(L_refl_at_component_1)
            
            if bulk_field_pos < idx1:
                # "bulk" field is on left side
                self.__bulk_field_LTR = self.__incident_field_left
                self.__bulk_field_RTL = self.__output_field_left
            
            elif bulk_field_pos >= idx2:
                # "bulk" field is on right side
                self.__bulk_field_LTR = self.__output_field_right
                self.__bulk_field_RTL = self.__incident_field_right
            
            return
            
        # WE HAVE A CAVITY. NOW LET'S START CALCULATING ROUND-TRIPS
        tic_start_time = time.perf_counter()
        last_tic_output_time = tic_start_time
        
        bulk_field = clsLightField(self.grid)
        
        for r in range(no_of_trips):
            final_round = (r == no_of_trips-1)
            
            calc_left_out = False
            calc_right_out = False
            calc_bulk = False
    
            if final_round:
                # in the final round calculate all output fields
                calc_left_out = True
                calc_right_out = True
                calc_bulk = True
                    
            # determine if it is time to print the progress (if required) 
            cur_time = time.perf_counter()
            diff_time = cur_time - last_tic_output_time 
            print_flag = (final_round or 
                          (diff_time>self.progress.max_time_betw_tic_outputs))            
            
            if print_flag:
                # calculate output fields in this round if required for 
                # progress plots
                if not calc_left_out:
                    calc_left_out = plot_left_out_progress 
                if not calc_right_out:
                    calc_right_out = plot_right_out_progress
                if not calc_bulk:
                    calc_bulk = (plot_bulk_LTR_progress or plot_bulk_RTL_progress)
            
            # ADD LEFT INCIDENT FIELD (after passing LTR through component 1) 
            # to current LTR bulk field
            bulk_field.add(LTR_transm_throu_component_1)
            
            # current bulk field is LTR bulk field if bulk_idx = idx1
            if calc_bulk:
                if bulk_field_pos == idx1:
                    self.__bulk_field_LTR = bulk_field
            
            # LTR TRIP INSIDE OF CAVITY
            if idx2-1 >= idx1+1:
                if calc_bulk and bulk_field_pos>idx1 and bulk_field_pos<idx2-1:
                    # The bulk field is to be calculated in this round and it 
                    # lies somewhere in the middle of the cavity:
                    # We need to split the LTR propagation into two parts
                    bulk_field = self.prop(bulk_field, 
                                           idx1+1, bulk_field_pos, Dir.LTR)
                    self.__bulk_field_LTR = bulk_field
                    bulk_field = self.prop(bulk_field, 
                                           bulk_field_pos+1, idx2-1, Dir.LTR)
                else:
                    # we can do the LTR propagation in a single pass
                    bulk_field = self.prop(bulk_field, idx1+1, idx2-1, Dir.LTR)
                
           
            #calculate right output field if required
            if calc_right_out:
                self.__output_field_right = self.prop(bulk_field, idx2, idx2, Dir.LTR)
                self.__output_field_right.add(R_refl_at_component_2)
            
            if calc_bulk:
                if bulk_field_pos == idx2-1:
                    # bulk field position is exactly before the rightmost 
                    # component, hence it is the current bulk field
                    self.__bulk_field_LTR = bulk_field
                
                elif bulk_field_pos >= idx2:
                    # bulk field position is right outside the cavity
                    # RTL bulk field is incident field
                    # LTR bulk field is right output field
                    self.__bulk_field_RTL = self.__incident_field_right
                    if calc_right_out:
                        self.__bulk_field_LTR = self.__output_field_right
                    else:
                        self.__bulk_field_LTR = self.prop(bulk_field, idx2, idx2, Dir.LTR)
                        self.__bulk_field_LTR.add(R_refl_at_component_2)
        
            # REFLECTION ON THE LEFT SIDE OF THE RIGHTMOST COMPONENT
            if idx2 > idx1:
                bulk_field = self.reflect(bulk_field, idx2, Side.LEFT)
            
            # ADD RIGHT INCIDENT FIELD (after passing RTL through component 2) 
            # to current RTL bulk field
            bulk_field.add(RTL_transm_throu_component_2)
            
            if calc_bulk:
                if bulk_field_pos == idx2-1:
                    # bulk field position is exactly before the rightmost 
                    # component, hence it is the current bulk field
                    self.__bulk_field_RTL = bulk_field
            
            # RTL TRIP INSIDE THE CAVITY
            if idx2-1 >= idx1+1:
                if calc_bulk and bulk_field_pos>idx1 and bulk_field_pos<idx2-1:
                    # The bulk field is to be calculated in this round and it 
                    # lies somewhere in the middle of the cavity:
                    # We need to split the RTL propagation into two parts
                    bulk_field = self.prop(bulk_field, 
                                           idx2-1, bulk_field_pos+1, Dir.RTL)
                    self.__bulk_field_RTL = bulk_field
                    bulk_field = self.prop(bulk_field, 
                                           bulk_field_pos, idx1+1,  Dir.RTL)
                else:
                    # we can do the LTR propagation in a single pass
                    bulk_field = self.prop(bulk_field, idx2-1, idx1+1,  Dir.RTL)
            
            #calculate left output field if required
            if calc_left_out:
                self.__output_field_left = self.prop(bulk_field, idx1, idx1, Dir.RTL)
                self.__output_field_left.add(L_refl_at_component_1)
                #self.__output_field_left.name = f"Left Output Field after Roundtrip {r+1}"
            
            if calc_bulk:
                if bulk_field_pos == idx1:
                    # bulk field position is right of leftmost component
                    # hence it is the current bulk field
                    self.__bulk_field_RTL = bulk_field
                
                elif bulk_field_pos < idx1:
                    # bulk field position is left outside the cavity
                    # LTR bulk field is incident field
                    # RTL bulk field is let output field
                    self.__bulk_field_LTR = self.__incident_field_left
                    if calc_left_out:
                        self.__bulk_field_RTL = self.__output_field_left
                    else:
                        self.__bulk_field_RTL = self.prop(bulk_field, idx1, idx1, Dir.RTL)
                        self.__bulk_field_RTL.add(L_refl_at_component_1)
                    
            
            # REFLECTION ON THE RIGHT SIDE OF THE LEFTMOST COMPONENT
            bulk_field = self.reflect(bulk_field, idx1, Side.RIGHT)
            
            # print or plot progress, if required
            if print_flag:
                last_tic_output_time = cur_time
                if print_progress:
                    self.progress.silent = silent_setting
                    self.progress.print(f"   roundtrip {r+1} of {no_of_trips}")
                    self.progress.silent = True
                    
                if plot_left_out_progress:
                    out = self.__output_field_left.clone()
                    out.name = f"Left Output Field after Roundtrip {r+1}"
                    out.shift(x_plot_shift, y_plot_shift)
                    out.plot_field(5)
                                    
                if plot_right_out_progress:
                    out = self.__output_field_right
                    out.name = f"Right Output Field after Roundtrip {r+1}"
                    out.shift(x_plot_shift, y_plot_shift)
                    out.plot_field(5)
                    
                if plot_bulk_LTR_progress:
                    out = self.__bulk_field_LTR
                    out.name = f"LTR Bulk Field after Roundtrip {r+1}"
                    out.shift(x_plot_shift, y_plot_shift)
                    out.plot_field(5)
                    
                if plot_bulk_RTL_progress:
                    out = self.__bulk_field_RTL
                    out.name = f"RTL Bulk Field after Roundtrip {r+1}"
                    out.shift(x_plot_shift, y_plot_shift)
                    out.plot_field(5)
            
        # print total time
        self.progress.silent = silent_setting 
        self.progress.pop()
        
 
    def reflect(self, field: clsLightField, idx, side: Side):
        """
        reflects the field at the component identified with idx
        on the side identified by Side
        """
        if self.component_count == 0:
            # no components: return black
            return field.apply_TR_mat(0)
        
        if idx<0 or idx >= self.component_count:
            # idx left or right of cavity
            return field.apply_TR_mat(0)
        
        component = self.get_component(idx)
        
        if (
                isinstance(component, clsMirror) or 
                isinstance(component, clsCurvedMirror)  or 
                isinstance(component, clsAmplitudeScaling)
        ):
            k_space_in = field.k_space
            k_space_out = False
            E_out = component.reflect(
                field.get_field_tot(k_space_in), k_space_in, k_space_out, side)
                
            out = clsLightField(self.grid)
            out.set_field(E_out, k_space_out)    
        else:
            if side == Side.LEFT:
                out = field.apply_TR_mat(component.R_L_mat_tot)
            else:
                out = field.apply_TR_mat(component.R_R_mat_tot)
            
        return out
        
    def prop_round_trip_LTR_RTL(self, field: clsLightField, idx1, idx2, incl_left_refl: bool):   
        """
        Single left-to-right, right-to-left roundtrip propagation
        - starting on the right side of the component with idx1, 
        - being transmitted left-to-right to the component with idx2, 
        - getting reflected at the left side of component with idx2, 
        - going back right-to-left to the component with idx1, and 
          (only if incl_left_refl == True):
        - getting reflected on the right side on the component with idx1
        """
        if self.component_count == 0:
            # no components: return black
            return field.apply_TR_mat(0)
        
        # idx1 supposed to be smaller
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        
        if idx1 < -1:
            idx1 = -1
        if idx2 >= self.component_count:
            idx2 = self.component_count-1
            
        if idx2 <= idx1:
            # no propagation: return input field
            return field
        
        if idx1 == -1 and incl_left_refl:
            # left reflection outside cavity -> result is black
            return field.apply_TR_mat(0)
        
        if idx1 == -1:
            name1 = "left cavity side"
            component1 = None
        else:
            component1 = self.get_component(idx1)
            name1 = component1.name
        component2 = self.get_component(idx2)
        name2 = component2.name
        self.progress.push_print("Calculating single LTR-RTL propagation between "+ \
                                 name1+" and "+name2)
        # left-to-right propagation
        field = self.prop(field, idx1+1, idx2-1, Dir.LTR)
        
        # right reflection
        self.progress.push_print("processing reflection at "+name2)
        
        
        field = self.reflect(field, idx2, Side.LEFT)
        
        #if isinstance(component2, clsMirror) or isinstance(component2, clsCurvedMirror):
        #    k_space_in = field.k_space
        #    k_space_out = False
        #    E_out = component2.reflect(field.get_field_tot(k_space_in), k_space_in, k_space_out, Side.LEFT)
        #    field.set_field(E_out, k_space_out)
        #else:
        #    field = field.apply_TR_mat(component2.R_L_mat_tot)
            
        self.progress.pop()
        
        # right-to-left propagation
        field = self.prop(field, idx2-1, idx1+1, Dir.RTL)
        
        if incl_left_refl:
            if not component1 is None:
                self.progress.push_print("processing reflection at "+name1)
                field = field.apply_TR_mat(component1.R_R_mat_tot)
                self.progress.pop()
        
        self.progress.pop()
        return field
 
    def get_round_trip_T_mat_LTR_RTL(self, idx1, idx2, incl_left_refl: bool):
        """
        Returns the roundtrip transmission matrix for a 
        left-to-right, right-to-left roundtrip 
        - starting on the right side of the component with idx1, 
        - being transmitted left-to-right to the component with idx2, 
        - getting reflected at the left side of component with idx2, 
        - going back right-to-left to the component with idx1, and 
          (only if incl_left_refl == True):
        - getting reflected on the right side on the component with idx1.
        """
        if self.component_count == 0:
            # no components: return unity
            return 1
        
        # idx1 supposed to be smaller
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        
        if idx1 < -1:
            idx1 = -1
        if idx2 >= self.component_count:
            idx2 = self.component_count-1
            
        if idx2 <= idx1:
            # no propagation: return unity
            return 1
        
        if idx1 == -1 and incl_left_refl:
            # left reflection outside cavity -> result is zero
            return 0
        
        if idx1 == -1:
            name1 = "left cavity side"
            component1 = None
        else:
            component1 = self.get_component(idx1)
            name1 = component1.name
        component2 = self.get_component(idx2)
        name2 = component2.name
        self.progress.push_print("Calculating single LTR-RTL roundtrip between "+ \
                                 name1+" and "+name2)
        
        # left-to-right propagation from idx+1 to idx2-1
        self.progress.push_print("left-to-right propagation:")
        T = 1
        for idx in range(idx1+1,idx2):
            component = self.get_component(idx)
            self.progress.push_print("processing component "+str(idx)+": "+component.name)
            T = mat_mul(T, component.T_LTR_mat_tot)
            self.progress.pop()
        self.progress.pop()
        
        # reflection on right component
        self.progress.push_print("calculating reflection at "+name2)
        T = mat_mul(T, component2.R_L_mat_tot)
        self.progress.pop()
        
        # right-to-left propagation from idx2-1 to idx1+1
        self.progress.push_print("right-to-left propagation:")
        for idx in range(idx2-1,idx1,-1):
            component = self.get_component(idx)
            self.progress.push_print("processing component "+str(idx)+": "+component.name)
            T = mat_mul(T, component.T_RTL_mat_tot)
            self.progress.pop()
        self.progress.pop()
        
        # reflection on left component
        if not component1 is None:
            if incl_left_refl:
                self.progress.push_print("calculating reflection at "+name1)
                T = mat_mul(T, component1.R_R_mat_tot)
                self.progress.pop()
        
        self.progress.pop()
        return T
    
    def get_inf_round_trip_R_L_mat(self, idx1, idx2):
        """ 
        returns the left reflection matrix for a single cavity reflecting
        between component with idx1 and component with idx2 
        """
        if self.component_count == 0:
            # no components: return unity
            return 0
        
        # idx1 supposed to be smaller
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        
        if idx1 < 0:
            idx1 = 0
        if idx2 >= self.component_count:
            idx2 = self.component_count-1
            
        if idx2 <= idx1:
            # no propagation: return unity
            return 0
        
        component1 = self.get_component(idx1)
        component2 = self.get_component(idx2)
        
        self.progress.push_print("calculating left reflection matrix " + \
                                 "after infinite roundtrips between "+ \
                                 component1.name+" and "+component2.name)
        
        # single round-trip matrix
        T_RT = self.get_round_trip_T_mat_LTR_RTL(idx1, idx2, False)
        # complete single round-trip matrix (including left reflection)
        R_T_RT = mat_mul(component1.R_R_mat_tot, T_RT)
        
        self.progress.push_print("calculating infinite roundtrip T-matrix")
        R =  mat_inv(mat_minus(1, R_T_RT))
        self.progress.pop()
        
        self.progress.push_print("finalizing calculation")
        R = mat_mul(T_RT, R)
        R = mat_mul3(component1.T_RTL_mat_tot, R, component1.T_LTR_mat_tot)
        R = mat_plus(component1.R_L_mat_tot, R)
        self.progress.pop()
        
        self.progress.pop()
        return R
        
    def get_round_trip_T_mat_RTL_LTR(self, idx1, idx2, incl_right_refl: bool):
        """
        Returns the roundtrip transmission matrix for a 
        right-to-left, left-to-right roundtrip 
        - starting on the left side of the component with idx2, 
        - being transmitted right-to-left to the component with idx1, 
        - getting reflected at the right side of component with idx1, 
        - going back left-to-right to the component with idx2, and 
          (only if incl_right_refl == True):
        - getting reflected on the left side on the component with idx2.
        """
        if self.component_count == 0:
            # no components: return unity
            return 1
        
        # idx1 supposed to be smaller
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        
        if idx1 < 0:
            idx1 = 0
        if idx2 > self.component_count:
            idx2 = self.component_count
            
        if idx2 <= idx1:
            # no propagation: return unity
            return 1
        
        if idx2 == self.component_count and incl_right_refl:
            # right reflection outside cavity -> result is zero
            return 0
        
        component1 = self.get_component(idx1)
        name1 = component1.name
        if idx2 == self.component_count:
            component2 = None
            name2 = "right side of cavity"
        else:
            component2 = self.get_component(idx2)
            name2 = component2.name
            
        self.progress.push_print("Calculating single RTL-LTR roundtrip between "+ \
                                 name2+" and "+name1)
        
        T = 1
        # right-to-left propagation from idx2-1 to idx1+1
        self.progress.push_print("right-to-left propagation:")
        for idx in range(idx2-1,idx1,-1):
            component = self.get_component(idx)
            self.progress.push_print("processing component "+str(idx)+": "+component.name)
            T = mat_mul(T, component.T_LTR_mat_tot)
            self.progress.pop()
        self.progress.pop()
        
        # reflection on left component
        self.progress.push_print("calculating reflection at "+name1)
        T = mat_mul(T, component1.R_R_mat_tot)
        self.progress.pop()
        
        # left-to-right propagation from idx+1 to idx2-1
        self.progress.push_print("left-to-right propagation:")
        for idx in range(idx1+1,idx2):
            component = self.get_component(idx)
            self.progress.push_print("processing component "+str(idx)+": "+component.name)
            T = mat_mul(T, component.T_LTR_mat_tot)
            self.progress.pop()
        self.progress.pop()
        
        # reflection on right component
        if not component2 is None:
            if incl_right_refl:
                self.progress.push_print("calculating reflection at "+component2.name)
                T = mat_mul(T, component2.R_L_mat_tot)
                self.progress.pop()
        
        self.progress.pop()
        return T
    
    def get_inf_round_trip_R_R_mat(self, idx1, idx2):
        """ 
        returns the right reflection matrix for a single cavity reflecting
        between component with idx1 and component with idx2 
        """
        if self.component_count == 0:
            # no components: return unity
            return 0
        
        # idx1 supposed to be smaller
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        
        if idx1 < 0:
            idx1 = 0
        if idx2 >= self.component_count:
            idx2 = self.component_count-1
            
        if idx2 <= idx1:
            # no propagation: return unity
            return 0
        
        component1 = self.get_component(idx1)
        component2 = self.get_component(idx2)
        
        self.progress.push_print("calculating right reflection matrix " + \
                                 "after infinite roundtrips between "+ \
                                 component2.name+" and "+component1.name)
        
        # single round-trip matrix
        T_RT = self.get_round_trip_T_mat_RTL_LTR(idx1, idx2, False)
        # complete single round-trip matrix (including right reflection)
        R_T_RT = mat_mul(component2.R_L_mat_tot, T_RT)
        
        self.progress.push_print("calculating infinite roundtrip T-matrix")
        R =  mat_inv(mat_minus(1, R_T_RT))
        self.progress.pop()
        
        self.progress.push_print("finalizing calculation")
        R = mat_mul(T_RT, R)
        R = mat_mul3(component2.T_LTR_mat_tot, R, component2.T_RTL_mat_tot)
        R = mat_plus(component2.R_L_mat_tot, R)
        self.progress.pop()
        
        self.progress.pop()
        return R
    
    def calc_R_L_mat_tot(self):
        """ 
        calculates the left reflection matrix for the whole cavity 
        (which is the top-left entry of the cavity's scattering block matrix)
        """ 
        self.progress.push_print("calculating cavity's reflection matrix R_L from cavity's transfer matrix M")
        if self.M_bmat_tot is None:
            self.calc_M_bmat_tot()
                
        M = self.M_bmat_tot
        if isinstance(M, clsBlockMatrix): 
            M01 = partial(M.get_block, 0, 1)
            M11 = partial(M.get_block, 1, 1)
            self.R_L_mat_tot = mat_div_bm(M01, M11, 
                                          self.progress, 
                                          "performing matrix division")
        else:
            self.R_L_mat_tot = mat_div(M[0][1],M[1][1])
        self.progress.pop()
        
    def calc_R_R_mat_tot(self):
        """ 
        calculates the right reflection matrix for the whole cavity 
        (which is the bottom-right entry of the cavity's scattering block matrix)
        """ 
        self.progress.push_print("calculating cavity's reflection matrix R_R from cavity's transfer matrix M")
        if self.M_bmat_tot is None:
            self.calc_M_bmat_tot()
        M = self.M_bmat_tot
        
        if isinstance(M, clsBlockMatrix):
            M11 = partial(M.get_block, 1, 1)
            M10 = partial(M.get_block, 1, 0)        
            self.R_R_mat_tot = -mat_inv_X_mul_Y_bm(M11, M10)
        else:
            self.R_R_mat_tot = -mat_inv_X_mul_Y(M[1][1], M[1][0])
        self.progress.pop()
                    
    def calc_T_LTR_mat_tot(self):
        """ 
        calculates the left-to-right transmission matrix for the whole cavity 
        (which is the bottom-left entry of the cavity's scattering block matrix)
        """        
        
        self.progress.push_print("calculating cavity's transmission matrix T_LTR from cavity's transfer matrix M")
        if self.M_bmat_tot is None:
            self.calc_M_bmat_tot()
        M = self.M_bmat_tot
        if isinstance(M, clsBlockMatrix):
            self.T_LTR_mat_tot = mat_inv(M.get_block(1,1))
        else:
            self.T_LTR_mat_tot = mat_inv(M[1][1])
        self.progress.pop()
        
    def calc_T_RTL_mat_tot(self):
        """ 
        calculates the right-to-left transmission matrix for the whole cavity 
        (which is the top-right entry of the cavity's scattering block matrix)
        """        
        self.progress.push_print("calculating cavity's transmission matrix T_RTL from cavity's transfer matrix M")
        if self.M_bmat_tot is None:
            self.calc_M_bmat_tot()
        M = self.M_bmat_tot
        if isinstance(M, clsBlockMatrix):
            #self.T_RTL_mat_tot = mat_minus(M.get_block(0,0), 
            #                               mat_mul(
            #                                   mat_div(
            #                                       M.get_block(0,1), 
            #                                       M.get_block(1,1), True), 
            #                                   M.get_block(1,0)))
            
            self.T_RTL_mat_tot = None
            gc.collect()
            M01 = partial(M.get_block, 0, 1)
            M11 = partial(M.get_block, 1, 1)
            tmp1 = mat_div_bm(M01, M11)
            #tmp1 =  mat_div(M.get_block(0,1), M.get_block(1,1)) 
            gc.collect()
            tmp2 = mat_mul(tmp1, M.get_block(1,0))
            del tmp1
            gc.collect()
            self.T_RTL_mat_tot = mat_minus(M.get_block(0,0), tmp2)
            del tmp2
            gc.collect()
            
        else:
            self.T_RTL_mat_tot = mat_minus(M[0][0], mat_mul(mat_div(M[0][1], M[1][1]), M[1][0]))
        
        self.progress.pop()

    def convert_R_L_mat_tot_to_fov(self):
        R_L_mat_tot = self.R_L_mat_tot
        self.progress.push_print("converting R_L_mat_tot to R_L_mat_fov")        
        self.R_L_mat_fov = self.grid.convert_TR_mat_tot_to_fov(R_L_mat_tot)
        self.progress.pop()
        
    def convert_R_R_mat_tot_to_fov(self):
        R_R_mat_tot = self.R_R_mat_tot
        self.progress.push_print("converting R_R_mat_tot to R_R_mat_fov")        
        self.R_R_mat_fov = self.grid.convert_TR_mat_tot_to_fov(R_R_mat_tot)
        self.progress.pop()
            
    def convert_T_LTR_mat_tot_to_fov(self):
        T_LTR_mat_tot = self.T_LTR_mat_tot
        self.progress.push_print("converting T_LTR_mat_tot to T_LTR_mat_fov")
        self.T_LTR_mat_fov = self.grid.convert_TR_mat_tot_to_fov(T_LTR_mat_tot)
        self.progress.pop()            
                    
    def convert_T_RTL_mat_tot_to_fov(self):
        T_RTL_mat_tot = self.T_RTL_mat_tot
        self.progress.push_print("converting T_RTL_mat_tot to T_RTL_mat_fov")
        self.T_RTL_mat_fov = self.grid.convert_TR_mat_tot_to_fov(T_RTL_mat_tot)
        self.progress.pop()                    
    
    
    

###############################################################################
# clsOptComponent4port
# represents any 4-port optical component (e.g. beamsplitter mirror)
###############################################################################      
class clsOptComponent4port(clsOptComponent):
    """ Represents an optical component with 4 ports (e.g. a beasmplitter) """
    def __init__(self, name, cavity):
        super().__init__(name, cavity) 
        
    #@property
    #def Lambda(self):
    #    """ returns exact wavelength in m """
    #    return super().Lambda
    
    #@Lambda.setter    
    #def Lambda(self, Lambda: float):
    #    """ sets wavelength in nm """
    #    super().Lambda = Lambda
    
    @abstractmethod
    def get_R_bmat_tot(self, side: Side):     
        """ 
        side: Side.LEFT or Side.RIGHT
        
        returns the left or right 2x2 reflection block matrix for all paths
        (total resolution) 
        """
        pass
    
    @abstractmethod    
    def get_T_bmat_tot(self, direction: Dir):    
        """ 
        direction: Dir.LTR or Dir.RTL
        
        returns the left-to-right or right-to-left 2x2 transmission 
        block matrix for all paths (total resolution) 
        """
        pass
    
    @abstractmethod
    def get_R_mat_tot(self, side: Side, path1: Path, path2: Path):     
        """ 
        side: side.LEFT or Side.RIGHT
        path1: Path.A or Path B
        path2: Path.A or Path B
        
        returns the left or right reflection matrix for 
        reflection in side from path1 to path2 (total resolution) 
        """
        pass
     
    @abstractmethod    
    def get_T_mat_tot(self, direction: Dir, path1: Path, path2: Path):    
        """ 
        direction: Dir.LTR or Dir.RTL
        path1: Path.A or Path B
        path2: Path.A or Path B
        
        returns the left-to-right or right-to-left transmission 
        from path1 to path2 (total resolution) 
        """
        pass

    
    @abstractmethod
    def get_dist_phys(self):
        """ 
        returns the physical distance for path A and path B
        """
        pass
    
    @abstractmethod
    def get_dist_opt(self):
        """ 
        returns the physical distance for path A and path B
        """
        pass
    
    def prop(self, in_A: clsLightField, in_B: clsLightField, direction: Dir):
        """
        Propagates the total input fields E_A and A_B into the direction
        indicated by paramter direction                
        ----------
        in_A : input light field in path A 
        in_B : input light field in path B 
        direction : Dir
            diretction of propagation (Dir.LTR or Dir.RTL)
        Returns
        -------
        Light fields after propagation 
        """
        # transfer from A to A
        T = self.get_T_mat_tot(direction, Path.A, Path.A)
        out_AA = in_A.apply_TR_mat(T)
        # transfer from A to B
        T = self.get_T_mat_tot(direction, Path.A, Path.B)
        out_AB = in_A.apply_TR_mat(T)
        # transfer from B to A
        T = self.get_T_mat_tot(direction, Path.B, Path.A)
        out_BA = in_B.apply_TR_mat(T)
        # transfer from B to B
        T = self.get_T_mat_tot(direction, Path.B, Path.B)
        out_BB = in_B.apply_TR_mat(T)
        
        out_AA.add(out_BA) # merge A->A and B->A to A_out
        out_BB.add(out_AB) # merge B->B and A->B to B_out
        
        return out_AA, out_BB

    @property
    def S_bmat_tot(self):
        """ 
        returns the 4x4 block scatering matrix (total resolution)
        as list [[S11, S12, S13, S14], ..., [S41, S42, S43, S44]] 
        """
        A = self.get_R_bmat_tot(Side.LEFT)
        B = self.get_T_bmat_tot(Dir.RTL)
        C = self.get_T_bmat_tot(Dir.LTR)
        D = self.get_R_bmat_tot(Side.RIGHT)
        
        if isinstance(A, clsBlockMatrix):
            S = clsBlockMatrix(4, self.cavity.use_swap_files_in_bmatrix_class,
                               self.cavity.tmp_folder)
            S.set_quadrant(1, A, False)
            S.set_quadrant(2, B, False)
            S.set_quadrant(3, C, False)
            S.set_quadrant(4, D, False)
            
            return S
        
        else:
            return bmat4_from_quadrants(A, B, C, D)
        
    def _S_to_M(self, S, p = None, msg = ""):  
        
        if not p is None:
            if not msg == "":
                p.tic_reset(5+3*8+2, False, msg)
        
        if isinstance(S, clsBlockMatrix):
            S11 = S.get_quadrant(1)
            S12 = S.get_quadrant(2)
            S21 = S.get_quadrant(3)
            S22 = S.get_quadrant(4)
            
            M = clsBlockMatrix(4, self.cavity.use_swap_files_in_bmatrix_class,
                               self.cavity.tmp_folder)
            
            S21_inv = bmat2_inv(S21, p) # inv: 5 steps       
            S11_S21_inv = bmat_mul(S11, S21_inv, p) # mul: 8 steps
            
            M.set_quadrant(1, bmat2_minus(S12, bmat_mul(S11_S21_inv, S22, p))) # mul: 8 steps
            if not p is None:
                p.tic()
                
            M.set_quadrant(2, S11_S21_inv)        
            M.set_quadrant(3, bmat_flip_sign(bmat_mul(S21_inv, S22, p))) # mul: 8 steps
            if not p is None:
                p.tic()
                
            M.set_quadrant(4, S21_inv)
            
            return M
        
        
        else:
                
            S11, S12, S21, S22 = get_bmat4_quadrants(S)
            
            S21_inv = bmat2_inv(S21, p) # inv: 5 steps       
            S11_S21_inv = bmat_mul(S11, S21_inv, p) # mul: 8 steps
            
            A = bmat2_minus(S12, bmat_mul(S11_S21_inv, S22, p)) # mul: 8 steps
            if not p is None:
                p.tic()
                
            B = S11_S21_inv        
            C = bmat_flip_sign(bmat_mul(S21_inv, S22, p)) # mul: 8 steps
            if not p is None:
                p.tic()
                
            D = S21_inv
            
            return bmat4_from_quadrants(A, B, C, D)
    
    @property
    def M_bmat_tot(self):
        """ returns the 4x4 block transfer matrix (total resolution)
        as list [[M11, M12, M13, M14], ..., [M41, M42, M43, M44]] """
        if self._M_bmat_tot_mem_cached:            
            # already cached in memory
            return self._M_bmat_tot
        
        if self.file_cache_M_bmat:
            # file caching active! Check, if file exists
            filename = self.cavity._full_file_name("M_mat",self.idx)
            if os.path.exists(filename):
                # file exists! Load from file
                self.load_M_bmat_tot()
                return self._M_bmat_tot
        
        self.cavity.progress.push_print("calculating transfer matrix M")        
        A = self.get_R_bmat_tot(Side.LEFT)
        B = self.get_T_bmat_tot(Dir.RTL)
        C = self.get_T_bmat_tot(Dir.LTR)
        D = self.get_R_bmat_tot(Side.RIGHT)
        
        if isinstance(A, clsBlockMatrix):           
            
            self.cavity.progress.push_print("constructing scattering matrix S")  
            clone_and_save = False
            #elapsed_time = self.cavity.progress.pop()
            if self.file_cache_M_bmat:
                #if elapsed_time > self.cavity.file_cache_min_calc_time:
                #    # matrix must be cloned for saving
                clone_and_save = True
            
            S = clsBlockMatrix(4, self.cavity.use_swap_files_in_bmatrix_class,
                               self.cavity.tmp_folder)
            #print("clone_and_save", clone_and_save)
            S.set_quadrant(1, A, clone_and_save)
            S.set_quadrant(2, B, clone_and_save)
            S.set_quadrant(3, C, clone_and_save)
            S.set_quadrant(4, D, clone_and_save)
            self.cavity.progress.pop()
            
            M = self._S_to_M(S, self.cavity.progress, 
                             "converting S-matrix to M-matrix") 
            
            if self.mem_cache_M_bmat:
                self._M_bmat_tot = M
                self._M_bmat_tot_mem_cached = True    
            
            if clone_and_save:
                self._M_bmat_tot = M
                self.save_M_bmat_tot()
                self._M_bmat_tot = None
                self._inv_M_bmat_tot = None
            
           
        
        else:
            M = self._S_to_M(bmat4_from_quadrants(A, B, C, D))
            
            if self.mem_cache_M_bmat:
                self._M_bmat_tot = M
                self._M_bmat_tot_mem_cached = True                
            
            elapsed_time = self.cavity.progress.pop()
            
            if self.file_cache_M_bmat:
                if elapsed_time > self.cavity.file_cache_min_calc_time:
                    self._M_bmat_tot = M
                    self.save_M_bmat_tot()
                    self._M_bmat_tot = None
                    self._inv_M_bmat_tot = None
    
        
        self.cavity.progress.pop()
        return M 
    
    def calc_inv_M_bmat_tot(self):
        """ calculates the inverse transfer block matrix """
        return bmat4_inv(self.M_bmat_tot, self.cavity.progress, "inverting M")
    
###############################################################################
# clsBeamSplitterMirror
# represents a beam splitter mirror as 4-port-component
############################################################################### 
class clsBeamSplitterMirror(clsOptComponent4port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__mirror2port = clsMirror(name, None)
        
    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
    
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space
    
    @property
    def k_space_in_dont_care(self):    
        return True

    @property
    def k_space_out_dont_care(self):    
        return True

    def set_phys_behavior(self, sym_phase: bool):
        """ 
        If sym_phase == True (default), then the mirror behaves
        physically insofar, as if it has the same reflection coefficient 
        on the left and right side, both will get the same phase. 
        If sym_phase == False, the mirror follows the convention that r_L, t_LTR 
        and t_RTL are real positive numbers, and r_R is a real negative number 
        """
        self.__mirror2port.set_phys_behavior(sym_phase)
        
    
    @property
    def R(self):
        """ power reflection on either side """
        self.__mirror2port.R
    
    @R.setter
    def R(self, R_new):
        if not self.cavity is None:
            self.cavity.clear()       
        self.__mirror2port.R = R_new 
    
    
    @property
    def T(self):
        """ power transmissivity on either side """
        self.__mirror2port.T
    
    @T.setter
    def T(self, T_new):
        if not self.cavity is None:
            self.cavity.clear()       
        self.__mirror2port.T = T_new 
    
    def get_R_bmat_tot(self, side: Side): 
        if side == Side.LEFT:
            # left reflection 2x2 block matrix 
            # top left quadrant of scattering 4x4 block matrix
            r_L = self.__mirror2port.r_L
            if self.cavity.use_bmatrix_class:
                R = clsBlockMatrix(2, 
                                   self.cavity.use_swap_files_in_bmatrix_class,
                                   self.cavity.tmp_folder)
                R.set_block(0, 1, r_L)
                R.set_block(1, 0, r_L)                
                return R
            
            else:
                return [[0,r_L],[r_L,0]]
        else:
            # right reflection 2x2 block matrix 
            # bottom right quadrant of scattering 4x4 block matrix
            r_R = self.__mirror2port.r_R
            if self.cavity.use_bmatrix_class:
                R = clsBlockMatrix(2, 
                                   self.cavity.use_swap_files_in_bmatrix_class,
                                   self.cavity.tmp_folder)
                R.set_block(0, 1, r_R)
                R.set_block(1, 0, r_R)
                return R
            
            else:
                return [[0,r_R],[r_R,0]]
            
            
    def get_T_bmat_tot(self, direction: Dir):
        if direction == Dir.LTR:
            # left-to-right transmission 2x2 block matrix
            # bottom left quadrant of scattering 4x4 block matrix
            t_LTR = self.__mirror2port.t_LTR
            if self.cavity.use_bmatrix_class:
                T = clsBlockMatrix(2, 
                                   self.cavity.use_swap_files_in_bmatrix_class,
                                   self.cavity.tmp_folder)
                T.set_block(0, 0, t_LTR)
                T.set_block(1, 1, t_LTR)
                return T
            
            else:
                return [[t_LTR,0],[0,t_LTR]]
        else:
            # right-to-left transmission 2x2 block matrix
            # top right quadrant of scattering 4x4 block matrix
            t_RTL = self.__mirror2port.t_RTL
            if self.cavity.use_bmatrix_class:
                T = clsBlockMatrix(2, 
                                   self.cavity.use_swap_files_in_bmatrix_class,
                                   self.cavity.tmp_folder)
                T.set_block(0, 0, t_RTL)
                T.set_block(1, 1, t_RTL)
                return T
            
            else:
                return [[t_RTL,0],[0,t_RTL]]
    
    def get_R_mat_tot(self, side: Side, path1: Path, path2: Path):  
        if path1 == path2:
            # only reflection between distinct paths
            return 0 
        
        if side == Side.LEFT:
            return self.__mirror2port.r_L
        else:
            return self.__mirror2port.r_R

    
    def get_T_mat_tot(self, direction: Dir, path1: Path, path2: Path): 
        if path1 != path2:
            # only transmission between identical paths
            return 0 
        
        if direction == Dir.LTR:
            return self.__mirror2port.t_LTR
        else:
            return self.__mirror2port.t_RTL

    def get_dist_phys(self):
        return 0, 0
        
    def get_dist_opt(self):
        return 0,0
        
###############################################################################
# clsTransmissionMixer
# mixes transmissions between paths A and B
############################################################################### 
class clsTransmissionMixer(clsOptComponent4port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__T_same = 1 # power transmission A to A, B to B
        self.__T_mix = 0 # power transmission A to B, B to A
        self.__t_same = 1 # amplitude transmission A to A
        self.__t_mix = 0 # amplitude transmission A to B
        self.__sym_phase = True
        self.__refl_behavior_for_path_mixing = True
        self.__set_transmission_amplitudes()

    @property
    def T_same(self):    
        """" power transmission between same paths """
        return self.__T_same     
    
    @T_same.setter
    def T_same(self, T):    
        """" power transmission between same paths """
        if T>1:
            print("Warning: T must be <= 1. Setting T to 1.")
            T = 1
        elif T<0:
            print("Warning: T must be >= 0. Setting T to 0.")
            T = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__T_same = T
        self.__T_mix = 1-T        
        self.__set_transmission_amplitudes()
        
    @property
    def T_mix(self):    
        """" power transmission between mixed paths """
        return self.__T_mix 

    @T_mix.setter
    def T_mix(self, T):    
        """" power transmission between mixed paths """
        if T>1:
            print("Warning: T must be <= 1. Setting T to 1.")
            T = 1
        elif T<0:
            print("Warning: T must be >= 0. Setting T to 0.")
            T = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__T_mix = T
        self.__T_same = 1-T        
        self.__set_transmission_amplitudes()

    @property
    def sym_phase(self):    
        """" If True, left/right symmetric phase behavior """
        return self.__sym_phase 
    
    @sym_phase.setter
    def sym_phase(self, x):
        if not self.cavity is None:
            self.cavity.clear()       
        self.__sym_phase = x
        self.__set_transmission_amplitudes()
    
    @property
    def refl_behavior_for_path_mixing(self):    
        """" 
        If True, phase behavior like a reflection when 
        transmitting between paths
        """
        return self.__refl_behavior_for_path_mixing 
    
    @refl_behavior_for_path_mixing.setter 
    def refl_behavior_for_path_mixing(self, x):
        if not self.cavity is None:
            self.cavity.clear()       
        self.__refl_behavior_for_path_mixing = x
        self.__set_transmission_amplitudes()
            
    def __set_transmission_amplitudes(self):
        """"set the amplitudes transmission factors from power transmission"""
        if self.sym_phase:
            # symmetric phase -> physical bhavior
            if self.refl_behavior_for_path_mixing:
                # mix-path transmission is like reflecting
                R = self.__T_mix 
                r = -R - 1j * math.sqrt(R) * math.sqrt(1 - R)
                t = 1 + r
                self.__t_same = t
                self.__t_mix = r
            else:
                # same-path transmission is like reflecting
                R = self.__T_same 
                r = -R - 1j * math.sqrt(R) * math.sqrt(1 - R)
                t = 1 + r
                self.__t_same = r
                self.__t_mix = t
        else:
            # simple phase bhavior with real-valued transmission amlitudes
            if self.refl_behavior_for_path_mixing:
                # mix-path transmission is like reflecting
                R = self.__T_mix 
                r = math.sqrt(R)
                t = math.sqrt(1-R)
                self.__t_same = t
                self.__t_mix = r
            else:
                # same-path transmission is like reflecting
                R = self.__T_same 
                r = math.sqrt(R)
                t = math.sqrt(1-R)
                self.__t_same = t
                self.__t_mix = r
                            
        
    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
    
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space
    
    @property
    def k_space_in_dont_care(self):    
        return True

    @property
    def k_space_out_dont_care(self):    
        return True

    def get_R_bmat_tot(self, side: Side):         
        # No reflection
        if self.cavity.use_bmatrix_class:
            R = clsBlockMatrix(2, 
                               self.cavity.use_swap_files_in_bmatrix_class,
                               self.cavity.tmp_folder)
            return R
        else:
            return [[0,0],[0,0]]
    
    def get_T_bmat_tot(self, direction: Dir):        
        # left-to-right or right-to-left transmission 2x2 block matrix
        if self.cavity.use_bmatrix_class:
            T = clsBlockMatrix(2, 
                               self.cavity.use_swap_files_in_bmatrix_class,
                               self.cavity.tmp_folder)
            T.set_block(0, 0, self.__t_same)
            T.set_block(0, 1, self.__t_mix)
            T.set_block(1, 0, self.__t_mix)
            T.set_block(1, 1, self.__t_same)     
            return T
        
        else:
            return [[self.__t_same,self.__t_mix],[self.__t_mix,self.__t_same]]
        
    
    def get_R_mat_tot(self, side: Side, path1: Path, path2: Path):
        # No reflection from any side
        return 0
    
    def get_T_mat_tot(self, direction: Dir, path1: Path, path2: Path):
        # left-to-right-transmission matrix 
        # (which is just a scalar in this case)
        if path1 == path2:
            return self.__t_same
        else:
            return self.__t_mix

    def get_dist_phys(self):
        return 0, 0

    def get_dist_opt(self):
        return 0,0
    
###############################################################################
# clsReflectionMixer
# mixes reflections between paths A and B
############################################################################### 
class clsReflectionMixer(clsOptComponent4port):
    def __init__(self, name, cavity):
        super().__init__(name, cavity)
        self.__R_same = 1 # power relection A to A, B to B
        self.__R_mix = 0 # power transmission A to B, B to A
        self.__r_same = 1 # amplitude transmission A to A
        self.__r_mix = 0 # amplitude transmission A to B
        self.__sym_phase = True
        self.__refl_behavior_for_path_mixing = True
        self.__set_reflection_amplitudes()

    @property
    def r_same(self):    
        return self.__r_same 
    
    @property
    def r_mix(self):    
        return self.__r_mix 

    @property
    def R_same(self):    
        """" power reflection between mixed paths """
        return self.__R_same     
    
    @R_same.setter
    def R_same(self, R):    
        """" power reflection between same paths """
        if R>1:
            print("Warning: R must be <= 1. Setting R to 1.")
            R = 1
        elif R<0:
            print("Warning: R must be >= 0. Setting R to 0.")
            R = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__R_same = R
        self.__R_mix = 1-R        
        self.__set_reflection_amplitudes()
        
    @property
    def R_mix(self):    
        """" power reflection between mixed paths """
        return self.__R_mix 

    @R_mix.setter
    def R_mix(self, R):    
        """" power reflection between mixed paths """
        if R>1:
            print("Warning: R must be <= 1. Setting R to 1.")
            R = 1
        elif R<0:
            print("Warning: R must be >= 0. Setting R to 0.")
            R = 0
        if not self.cavity is None:
            self.cavity.clear()       
        self.__R_mix = R
        self.__R_same = 1-R        
        self.__set_reflection_amplitudes()

    @property
    def sym_phase(self):    
        """" If True, left/right symmetric phase behavior """
        return self.__sym_phase 
    
    @sym_phase.setter
    def sym_phase(self, x):
        if not self.cavity is None:
            self.cavity.clear()       
        self.__sym_phase = x
        self.__set_reflection_amplitudes()
    
    @property
    def refl_behavior_for_path_mixing(self):    
        """" 
        If True, mix-path reflection is like reflecting 
        If False, same-path transmission is like reflecting
        """
        return self.__refl_behavior_for_path_mixing 
    
    @refl_behavior_for_path_mixing.setter 
    def refl_behavior_for_path_mixing(self, x):
        if not self.cavity is None:
            self.cavity.clear()       
        self.__refl_behavior_for_path_mixing = x
        self.__set_reflection_amplitudes()
            
    def __set_reflection_amplitudes(self):
        """"set the amplitudes reflection factors from power reflection"""
        if self.sym_phase:
            # symmetric phase -> physical bhavior
            if self.refl_behavior_for_path_mixing:
                # mix-path reflection is like reflecting
                R = self.__R_mix 
                r = -R - 1j * math.sqrt(R) * math.sqrt(1 - R)
                t = 1 + r
                self.__r_same = t
                self.__r_mix = r
            else:
                # same-path transmission is like reflecting
                R = self.__R_same 
                r = -R - 1j * math.sqrt(R) * math.sqrt(1 - R)
                t = 1 + r
                self.__r_same = r
                self.__r_mix = t
        else:
            # simple phase bhavior with real-valued transmission amlitudes
            if self.refl_behavior_for_path_mixing:
                # mix-path transmission is like reflecting
                R = self.__R_mix 
                r = math.sqrt(R)
                t = math.sqrt(1-R)
                self.__r_same = t
                self.__r_mix = r
            else:
                # same-path transmission is like reflecting
                R = self.__R_same 
                r = math.sqrt(R)
                t = math.sqrt(1-R)
                self.__r_same = t
                self.__r_mix = r
                            
        
    @property
    def k_space_in_prefer(self):    
        return False # preferred input: position-space
    
    @property
    def k_space_out_prefer(self):    
        return False  # preferred output: position-space
    
    @property
    def k_space_in_dont_care(self):    
        return True

    @property
    def k_space_out_dont_care(self):    
        return True

    def get_R_bmat_tot(self, side: Side):         
        # reflection block matrix
        if not self.sym_phase and side==Side.RIGHT:
            if self.cavity.use_bmatrix_class:
                R = clsBlockMatrix(2, 
                                   self.cavity.use_swap_files_in_bmatrix_class,
                                   self.cavity.tmp_folder)
                R.set_block(0, 0, -self.__r_same)
                R.set_block(0, 1, -self.__r_mix)
                R.set_block(1, 0, -self.__r_mix)
                R.set_block(1, 1, -self.__r_same)
                
                return R
            
            else:
                return [[-self.__r_same, -self.__r_mix],[-self.__r_mix, -self.__r_same]]
        
        else:
            if self.cavity.use_bmatrix_class:
                R = clsBlockMatrix(2, 
                                   self.cavity.use_swap_files_in_bmatrix_class,
                                   self.cavity.tmp_folder)
                R.set_block(0, 0, self.__r_same)
                R.set_block(0, 1, self.__r_mix)
                R.set_block(1, 0, self.__r_mix)
                R.set_block(1, 1, self.__r_same)
                
                return R
            
            else:
                return [[self.__r_same,self.__r_mix],[self.__r_mix,self.__r_same]]
        
    def get_T_bmat_tot(self, direction: Dir):        
        # no transmission at all
        if self.cavity.use_bmatrix_class:
            T = clsBlockMatrix(2, 
                               self.cavity.use_swap_files_in_bmatrix_class,
                               self.cavity.tmp_folder)
            return T
            
        else:
            return [[0,0],[0,0]]
            
    
    def get_R_mat_tot(self, side: Side, path1: Path, path2: Path):
        # reflection
        if not self.sym_phase and side==Side.RIGHT:
            f = -1
        else:
            f = 1
        
        if path1 == path2:
            return f * self.__r_same
        else:
            return f * self.__r_mix
        
    
    def get_T_mat_tot(self, direction: Dir, path1: Path, path2: Path):
        # no transnmission in any direction
        return 0
        
    def get_dist_phys(self):
        return 0, 0
    
    def get_dist_opt(self):
        return 0,0
                
###############################################################################
# clsOptComponentAdapter
# merges one or two 2-port-components into a 4-port-component
############################################################################### 
class clsOptComponentAdapter(clsOptComponent4port):
    def __init__(self, name, cavity):
        self.__component_A = None
        self.__component_B = None
        super().__init__(name, cavity)

    @property
    def name(self):
        if super().name != "":
            # return explicitly defined name
            return super().name
        
        # return name composed of sub-component's names
        name = ""
        if not self.__component_A is None:
            name += self.__component_A.name
            name += " in path A"
            if not self.__component_B is None:
                name += ", "
        
        if not self.__component_B is None:
            name += self.__component_B.name
            name += " in path B"
    
        return name

    @property
    def Lambda_nm(self):
        """ returns wavelength in m """
        return super().Lambda_nm

    @Lambda_nm.setter    
    def Lambda_nm(self, lambda_nm: float):
        """ sets wavelength in nm """
        super().Lambda_nm = lambda_nm
        if not self.__component_A is None:
            self.__component_A.Lambda_nm = lambda_nm
        if not self.__component_B is None:
            self.__component_B.Lambda_nm = lambda_nm
    
    @property
    def Lambda(self):
        """ returns wavelength in m """
        return super().Lambda
    
    @Lambda.setter    
    def Lambda(self, Lambda: float):
        """ sets wavelength in nm """
        #super().Lambda = Lambda
        self._lambda = Lambda
        self._lambda_nm = Lambda * 1000000000
        self.clear_mem_cache()
        if not self.__component_A is None:
            self.__component_A.Lambda = Lambda
        if not self.__component_B is None:
            self.__component_B.Lambda = Lambda

    @property   
    def component_A(self):
        """ returns optical 2-port-component from path A """
        return self.__component_A

    @property   
    def component_B(self):
        """ returns optical 2-port-component from path B """
        return self.__component_B

    def __sub_component_index(self, path: Path):
        # calculates index for sub-components in path A or B
        if self.idx == -1:
            if path == Path.A:
                return -2
            else:
                return -3
        if path == Path.A:
            return self.idx+1000
        else:
            return self.idx+2000
                
    def connect_component(self, component: clsOptComponent2port, path: Path):
        """
        connects a 2-port-component to path Path.A or path.B, respectively
        """
        if not self.cavity is None:
            self.cavity.clear()       
        if path == Path.A:
            self.__component_A = component
        else:
            self.__component_B = component
        if not component is None:
            idx = self.__sub_component_index(path)
            component._connect_to_cavity(self.cavity, self.grid, idx) 
            
    def _connect_to_cavity(self, cavity, grid, idx):
        # extension of default behavior:
        # also connects sub-components to cavity    
        super()._connect_to_cavity(cavity, grid, idx)
        if not self.component_A is None:
            idx = self.__sub_component_index(Path.A)
            self.component_A._connect_to_cavity(self.cavity, self.grid, idx)
        if not self.component_B is None:
            idx = self.__sub_component_index(Path.B)
            self.component_B._connect_to_cavity(self.cavity, self.grid, idx)
        
    def clear_mem_cache(self):
        super().clear_mem_cache()
        if not self.component_A is None:
            self.component_A.clear_mem_cache()
        if not self.component_B is None:
            self.component_B.clear_mem_cache()
    
    @property
    def k_space_in_prefer(self):   
        if self.component_A is None and self.component_B is None:    
            return False # preferred input: position-space
        
        if self.component_A is None:
            return self.component_B.k_space_in_prefer
        else:
            return self.component_A.k_space_in_prefer
    
    @property
    def k_space_out_prefer(self):    
        if self.component_A is None and self.component_B is None:    
            return False # preferred output: position-space
        
        if self.component_A is None:
            return self.component_B.k_space_out_prefer
        else:
            return self.component_A.k_space_out_prefer
    
    @property
    def k_space_in_dont_care(self):    
        if self.component_A is None and self.component_B is None:    
            return True # No component: No preference
        
        if self.component_A is None:
            return self.component_B.k_space_in_dont_care
        else:
            return self.component_A.k_space_in_dont_care
    
    @property
    def k_space_out_dont_care(self):    
        if self.component_A is None and self.component_B is None:    
            return True # No component: No preference
        
        if self.component_A is None:
            return self.component_B.k_space_out_dont_care
        else:
            return self.component_A.k_space_out_dont_care

    def get_R_bmat_tot(self, side: Side): 
        # default values: no reflection
        R_A = 0
        R_B = 0
        
        if side == Side.LEFT:
            # left reflection 2x2 block matrix 
            # top left quadrant of scattering 4x4 block matrix
            if not self.component_A is None:
                R_A = self.component_A.R_L_mat_tot
            if not self.component_B is None:
                R_B = self.component_B.R_L_mat_tot
        else:
            # right reflection 2x2 block matrix 
            # bottom right quadrant of scattering 4x4 block matrix
            if not self.component_A is None:
                R_A = self.component_A.R_R_mat_tot
            if not self.component_B is None:
                R_B = self.component_B.R_R_mat_tot
        
        
        if self.cavity.use_bmatrix_class:
            R = clsBlockMatrix(2, 
                               self.cavity.use_swap_files_in_bmatrix_class,
                               self.cavity.tmp_folder)
        
            R.set_block(0, 0, R_A)
            R.set_block(1, 1, R_B)
            
            return R
            
        else:
            return [[R_A, 0],[0, R_B]]
            
            
    def get_T_bmat_tot(self, direction: Dir):
        # default values: transmission without changes
        T_A = 1
        T_B = 1
        
        if direction == Dir.LTR:
            # left-to-right transmission 2x2 block matrix
            # bottom left quadrant of scattering 4x4 block matrix
            if not self.component_A is None:
                T_A = self.component_A.T_LTR_mat_tot
            if not self.component_B is None:
                T_B = self.component_B.T_LTR_mat_tot
        else:
            # right-to-left transmission 2x2 block matrix
            # top right quadrant of scattering 4x4 block matrix
            if not self.component_A is None:
                T_A = self.component_A.T_RTL_mat_tot
            if not self.component_B is None:
                T_B = self.component_B.T_RTL_mat_tot
        
        if self.cavity.use_bmatrix_class:
            T = clsBlockMatrix(2, 
                               self.cavity.use_swap_files_in_bmatrix_class,
                               self.cavity.tmp_folder)
        
            T.set_block(0, 0, T_A)
            T.set_block(1, 1, T_B)
            
            return T
            
        else:
            return [[T_A,0],[0,T_B]]

    def get_R_mat_tot(self, side: Side, path1: Path, path2: Path):  
        if path1 != path2:
            # only reflection between identical paths
            return 0 
        
        R = 0 # default value: no reflection
        if path1 == Path.A:
            # reflection from path A to path A
            if not self.component_A is None:
                # component A exists
                if side == Side.LEFT:
                    # left reflection from path A to path A
                    R = self.component_A.R_L_mat_tot
                else:
                    # right reflection from path A to path A
                    R = self.component_A.R_R_mat_tot
        else:
            # reflection from path B to path B
           if not self.component_B is None:
               # component B exists
               if side == Side.LEFT:
                   # left reflection from path B to path B
                   R = self.component_B.R_L_mat_tot
               else:
                   # right reflection from path B to path B
                   R = self.component_B.R_R_mat_tot

        return R
    
    def get_T_mat_tot(self, direction: Dir, path1: Path, path2: Path):  
        if path1 != path2:
            # only transmission between identical paths
            return 0 
        
        T = 1 # default value: neutral transmission
        if path1 == Path.A:
            # transmission in path A 
            if not self.component_A is None:
                # component A exists
                if direction == Dir.LTR:
                    # left-to-right transmission in path A
                    T = self.component_A.T_LTR_mat_tot
                else:
                    # right-to-left transmission in path A
                    T = self.component_A.T_RTL_mat_tot
        else:
            # transmission in path B
           if not self.component_B is None:
               # component B exists
               if direction == Dir.LTR:
                   # left-to-right transmission in path B
                   T = self.component_B.T_LTR_mat_tot
               else:
                   # right-to-left transmission in path A
                   T = self.component_B.T_RTL_mat_tot
        
        return T

    def get_dist_phys(self):
        dist_A, dist_B = 0, 0
        if not self.component_A is None:
            dist_A = self.component_A.dist_phys
        if not self.component_B is None:
            dist_B = self.component_B.dist_phys
        return dist_A, dist_B
    
    def get_dist_opt(self):
        dist_A, dist_B = 0, 0
        if not self.component_A is None:
            dist_A = self.component_A.dist_opt
        if not self.component_B is None:
            dist_B = self.component_B.dist_opt
        return dist_A, dist_B

    
###############################################################################
# clsCavity2path
# represents a cavity with two paths (e.g. a ring cavity)
############################################################################### 
class clsCavity2path(clsCavity):
    """ Represents a cavity with 2 optical paths """
    def __init__(self, name, full_precision = True):
        super().__init__(name, full_precision)   
        self.__incident_field_left_A = None
        self.__incident_field_left_B = None
        self.__incident_field_right_A = None
        self.__incident_field_right_B = None
        self.__output_field_left_A = None
        self.__output_field_left_B = None
        self.__output_field_right_A = None
        self.__output_field_right_B = None
        self.__bulk_field_LTR_A = None
        self.__bulk_field_LTR_B = None
        self.__bulk_field_RTL_A = None
        self.__bulk_field_RTL_B = None
        self.__bulk_field_pos = None    
        self.__use_bmatrix_class = True
        self.__use_swap_files_in_bmatrix_class = True
        
    @property
    def use_bmatrix_class(self):
        """ Is clsBlockMatrix to be used? """
        return self.__use_bmatrix_class
    
    @use_bmatrix_class.setter
    def use_bmatrix_class(self, x):
        self.__use_bmatrix_class = x
        
    @property
    def use_swap_files_in_bmatrix_class(self):
        """ are temporary files to usd in clsBlockMatrix to save RAM memory? """        
        return self.__use_swap_files_in_bmatrix_class 
    
    @use_swap_files_in_bmatrix_class.setter
    def use_swap_files_in_bmatrix_class(self, x):
        self.__use_swap_files_in_bmatrix_class = x
    
    def clear_results(self):
        self.__output_field_left_A = None
        self.__output_field_left_B = None
        self.__output_field_right_A = None
        self.__output_field_right_B = None
        self.__bulk_field_LTR_A = None
        self.__bulk_field_LTR_B = None
        self.__bulk_field_RTL_A = None
        self.__bulk_field_RTL_B = None
        self.__bulk_field_pos = None  
    
    def get_bulk_field_LTR(self, path: Path):
        """
        Returns the left-to-right bulk field in path A or B
        """
        if path == Path.A:
            return self.__bulk_field_LTR_A
        else:
            return self.__bulk_field_LTR_B
    
    def get_bulk_field_RTL(self, path: Path):
        """
        Returns the right-to-left bulk field in path A or B
        """
        if path == Path.A:
            return self.__bulk_field_RTL_A
        else:
            return self.__bulk_field_RTL_B

    def get_bulk_field(self, path: Path):
        """
        Returns the total bulk field in path A or B
        (sum of LTR and RTL fields)
        """
        field = clsLightField(self.grid)
        if not self.get_bulk_field_LTR(path) is None:
            field.add(self.get_bulk_field_LTR(path))
        if not self.get_bulk_field_RTL(path) is None:
            field.add(self.get_bulk_field_RTL(path)) 
        return field
    
    @property
    def bulk_field_pos(self):
        return self.__bulk_field_pos    

    
    def set_incident_field(self, side: Side, path: Path, field: clsLightField):
        """
        Sets the left or right incident field in path A or B
        """

        if side == Side.LEFT:
            # left side
            if path == Path.A:
                self.__incident_field_left_A = field
            else:
                self.__incident_field_left_B = field
        else:
            # right side
            if path == Path.A:
                self.__incident_field_right_A = field
            else:
                self.__incident_field_right_B = field
        self.__output_field_left_A = None
        self.__output_field_left_B = None
        self.__output_field_right_A = None
        self.__output_field_right_B = None
        self.__bulk_field_LTR_A = None
        self.__bulk_field_LTR_B = None
        self.__bulk_field_RTL_A = None
        self.__bulk_field_RTL_B = None
        self.__bulk_field_pos = None   
        
    def get_incident_field(self, side: Side, path: Path):
        """
        Returns the left or right incident field in path A or B
        """
        if side == Side.LEFT:
            # left side
            if path == Path.A:
                return self.__incident_field_left_A 
            else:
                return self.__incident_field_left_B
        else:
            # right side
            if path == Path.A:
                return self.__incident_field_right_A 
            else:
                return self.__incident_field_right_B 
    
    def get_output_field(self, side: Side, path: Path):
        """
        returns the left or right output field in path A or B
        """
        if side == Side.LEFT:
            # left side
            if path == Path.A:
                if self.__output_field_left_A is None:
                    self._calc_output_field(Side.LEFT, Path.A)
                return self.__output_field_left_A
            else:
                if self.__output_field_left_B is None:
                    self._calc_output_field(Side.LEFT, Path.B)
                return  self.__output_field_left_B
        else:
            # right side
            if path == Path.A:
                if self.__output_field_right_A is None:
                    self._calc_output_field(Side.RIGHT, Path.A)
                return self.__output_field_right_A
            else:
                if self.__output_field_right_B is None:
                    self._calc_output_field(Side.RIGHT, Path.B)
                return self.__output_field_right_B
    
    
    def _calc_output_field(self, side: Side, path: Path):
        """ 
        calculates the left or right output field in path A or B,
        based on the left and right input fields in path A and B
        """
        if side == Side.LEFT:
            # left output field
            if path == Path.A:
                # left output field, path A
                self.__output_field_left_A = clsLightField(self.grid)
                self.__output_field_left_A.name = "Output Field Left (Path A)"
                
                if not (self.__incident_field_left_A is None and 
                        self.__incident_field_left_B is None):
                    X = self.R_L_mat_tot
                
                if isinstance(X, clsBlockMatrix):
                    if not self.__incident_field_left_A is None:
                        self.__output_field_left_A.add(
                            self.__incident_field_left_A.apply_TR_mat(X.get_block(0,0)))
    
                    if not self.__incident_field_left_B is None:
                        self.__output_field_left_A.add(
                            self.__incident_field_left_B.apply_TR_mat(X.get_block(0,1)))
                    
                    if not (self.__incident_field_right_A is None and 
                            self.__incident_field_right_B is None):
                        X = self.T_RTL_mat_tot
                    
                    if not self.__incident_field_right_A is None:
                        self.__output_field_left_A.add(
                            self.__incident_field_right_A.apply_TR_mat(X.get_block(0,0)))
    
                    if not self.__incident_field_right_B is None:
                        self.__output_field_left_A.add(
                            self.__incident_field_right_B.apply_TR_mat(X.get_block(0,1)))
                
                else:
                    if not self.__incident_field_left_A is None:
                        self.__output_field_left_A.add(
                            self.__incident_field_left_A.apply_TR_mat(X[0][0]))
    
                    if not self.__incident_field_left_B is None:
                        self.__output_field_left_A.add(
                            self.__incident_field_left_B.apply_TR_mat(X[0][1]))
                    
                    if not (self.__incident_field_right_A is None and 
                            self.__incident_field_right_B is None):
                        X = self.T_RTL_mat_tot
                    
                    if not self.__incident_field_right_A is None:
                        self.__output_field_left_A.add(
                            self.__incident_field_right_A.apply_TR_mat(X[0][0]))
    
                    if not self.__incident_field_right_B is None:
                        self.__output_field_left_A.add(
                            self.__incident_field_right_B.apply_TR_mat(X[0][1]))
            else:
                # left output field, path B
                self.__output_field_left_B = clsLightField(self.grid)
                self.__output_field_left_B.name = "Output Field Left (Path B)"
                
                if not (self.__incident_field_left_A is None and 
                        self.__incident_field_left_B is None):
                    X = self.R_L_mat_tot
                
                if isinstance(X, clsBlockMatrix):
                    if not self.__incident_field_left_A is None:
                        self.__output_field_left_B.add(
                            self.__incident_field_left_A.apply_TR_mat(X.get_block(1,0)))
    
                    if not self.__incident_field_left_B is None:
                        self.__output_field_left_B.add(
                            self.__incident_field_left_B.apply_TR_mat(X.get_block(1,1)))
                    
                    if not (self.__incident_field_right_A is None and 
                            self.__incident_field_right_B is None):
                        X = self.T_RTL_mat_tot
                    
                    if not self.__incident_field_right_A is None:
                        self.__output_field_left_B.add(
                            self.__incident_field_right_A.apply_TR_mat(X.get_block(1,0)))
    
                    if not self.__incident_field_right_B is None:
                        self.__output_field_left_B.add(
                            self.__incident_field_right_B.apply_TR_mat(X.get_block(1,1)))
                
                else:
                    if not self.__incident_field_left_A is None:
                        self.__output_field_left_B.add(
                            self.__incident_field_left_A.apply_TR_mat(X[1][0]))
    
                    if not self.__incident_field_left_B is None:
                        self.__output_field_left_B.add(
                            self.__incident_field_left_B.apply_TR_mat(X[1][1]))
                    
                    if not (self.__incident_field_right_A is None and 
                            self.__incident_field_right_B is None):
                        X = self.T_RTL_mat_tot
                    
                    if not self.__incident_field_right_A is None:
                        self.__output_field_left_B.add(
                            self.__incident_field_right_A.apply_TR_mat(X[1][0]))
    
                    if not self.__incident_field_right_B is None:
                        self.__output_field_left_B.add(
                            self.__incident_field_right_B.apply_TR_mat(X[1][1]))
                    
        else:
            # right output field
            if path == Path.A:
                # right output field, path A
                self.__output_field_right_A = clsLightField(self.grid)
                self.__output_field_right_A.name = "Output Field Right (Path A)"
                
                if not (self.__incident_field_left_A is None and 
                        self.__incident_field_left_B is None):
                    X = self.T_LTR_mat_tot
                
                if isinstance(X, clsBlockMatrix):
                    if not self.__incident_field_left_A is None:
                        self.__output_field_right_A.add(
                            self.__incident_field_left_A.apply_TR_mat(X.get_block(0,0)))
    
                    if not self.__incident_field_left_B is None:
                        self.__output_field_right_A.add(
                            self.__incident_field_left_B.apply_TR_mat(X.get_block(0,1)))
                    
                    if not (self.__incident_field_right_A is None and 
                            self.__incident_field_right_B is None):
                        X = self.R_R_mat_tot
                    
                    if not self.__incident_field_right_A is None:
                        self.__output_field_right_A.add(
                            self.__incident_field_right_A.apply_TR_mat(X.get_block(0,0)))
    
                    if not self.__incident_field_right_B is None:
                        self.__output_field_right_A.add(
                            self.__incident_field_right_B.apply_TR_mat(X.get_block(0,1)))
                        
                else:
                    if not self.__incident_field_left_A is None:
                        self.__output_field_right_A.add(
                            self.__incident_field_left_A.apply_TR_mat(X[0][0]))
    
                    if not self.__incident_field_left_B is None:
                        self.__output_field_right_A.add(
                            self.__incident_field_left_B.apply_TR_mat(X[0][1]))
                    
                    if not (self.__incident_field_right_A is None and 
                            self.__incident_field_right_B is None):
                        X = self.R_R_mat_tot
                    
                    if not self.__incident_field_right_A is None:
                        self.__output_field_right_A.add(
                            self.__incident_field_right_A.apply_TR_mat(X[0][0]))
    
                    if not self.__incident_field_right_B is None:
                        self.__output_field_right_A.add(
                            self.__incident_field_right_B.apply_TR_mat(X[0][1]))
            else:
                # right output field, path B
                self.__output_field_right_B = clsLightField(self.grid)
                self.__output_field_right_B.name = "Output Field Right (Path B)"
                
                if not (self.__incident_field_left_A is None and 
                        self.__incident_field_left_B is None):
                    X = self.T_LTR_mat_tot
                
                if isinstance(X, clsBlockMatrix):
                    if not self.__incident_field_left_A is None:
                        self.__output_field_right_B.add(
                            self.__incident_field_left_A.apply_TR_mat(X.get_block(1,0)))
    
                    if not self.__incident_field_left_B is None:
                        self.__output_field_right_B.add(
                            self.__incident_field_left_B.apply_TR_mat(X.get_block(1,1)))
                    
                    if not (self.__incident_field_right_A is None and 
                            self.__incident_field_right_B is None):
                        X = self.R_R_mat_tot
                    
                    if not self.__incident_field_right_A is None:
                        self.__output_field_right_B.add(
                            self.__incident_field_right_A.apply_TR_mat(X.get_block(1,0)))
    
                    if not self.__incident_field_right_B is None:
                        self.__output_field_right_B.add(
                            self.__incident_field_right_B.apply_TR_mat(X.get_block(1,1))) 
                
                else:
                    if not self.__incident_field_left_A is None:
                        self.__output_field_right_B.add(
                            self.__incident_field_left_A.apply_TR_mat(X[1][0]))
    
                    if not self.__incident_field_left_B is None:
                        self.__output_field_right_B.add(
                            self.__incident_field_left_B.apply_TR_mat(X[1][1]))
                    
                    if not (self.__incident_field_right_A is None and 
                            self.__incident_field_right_B is None):
                        X = self.R_R_mat_tot
                    
                    if not self.__incident_field_right_A is None:
                        self.__output_field_right_B.add(
                            self.__incident_field_right_A.apply_TR_mat(X[1][0]))
    
                    if not self.__incident_field_right_B is None:
                        self.__output_field_right_B.add(
                            self.__incident_field_right_B.apply_TR_mat(X[1][1]))                
        
    def calc_bulk_field_from_left(self, idx: int):
        """ 
        Calulates the bulk field right of component identified with idx
        based on the left incident and left output field
        """
        self.progress.push_print("calculating bulk field right of component "
                                 +str(idx)+", coming from the left side")
        if self.component_count == 0:
            self.progress.print("There are no components! End task.")
            self.__bulk_field_LTR_A = None
            self.__bulk_field_LTR_B = None
            self.__bulk_field_RTL_A = None
            self.__bulk_field_RTL_B = None
            self.__bulk_field_pos = None
            return
        
        if idx >= self.component_count-1:
            # right from rightmost component
            # "bulk field" is the field on the right side
            self.progress.print("'Right of index "+str(idx)+"' "+ 
                                "is outside the bulk on the right side. "+
                                "Calculating field on the right side of cavity.")
            self.__bulk_field_LTR_A = self.get_output_field(Side.RIGHT, Path.A)
            self.__bulk_field_LTR_B = self.get_output_field(Side.RIGHT, Path.B)
            self.__bulk_field_RTL_A = self.get_incident_field(Side.RIGHT, Path.A)
            self.__bulk_field_RTL_B = self.get_incident_field(Side.RIGHT, Path.B)
            if self.__bulk_field_LTR_A is None:
                self.__bulk_field_LTR_A = clsLightField(self.grid)
            if self.__bulk_field_LTR_B is None:
                self.__bulk_field_LTR_B = clsLightField(self.grid)
            if self.__bulk_field_RTL_A is None:
                self.__bulk_field_RTL_A = clsLightField(self.grid)
            if self.__bulk_field_RTL_B is None:
                self.__bulk_field_RTL_B = clsLightField(self.grid)
            self.__bulk_field_pos = self.component_count-1
            return
        
        if idx < 0:
            # right from component with idx -1 or smaller ->
            # "bulk field" is the field on the left side
            self.progress.print("'Right of index "+str(idx)+ "' "+ 
                                "is outside the bulk on the left side. "+
                                "Calculating field on the left side of cavity.")
            self.__bulk_field_LTR_A = self.get_incident_field(Side.LEFT, Path.A)
            self.__bulk_field_LTR_B = self.get_incident_field(Side.LEFT, Path.B)
            self.__bulk_field_RTL_A = self.get_output_field(Side.LEFT, Path.A)
            self.__bulk_field_RTL_B = self.get_output_field(Side.LEFT, Path.B)
            
            if self.__bulk_field_LTR_A is None:
                self.__bulk_field_LTR_A = clsLightField(self.grid)
            if self.__bulk_field_LTR_B is None:
                self.__bulk_field_LTR_B = clsLightField(self.grid)
            if self.__bulk_field_RTL_A is None:
                self.__bulk_field_RTL_A = clsLightField(self.grid)
            if self.__bulk_field_RTL_B is None:
                self.__bulk_field_RTL_B = clsLightField(self.grid)
            self.__bulk_field_pos = -1
            return
    
        for i in range(idx, -1,-1):
            component = self.components[i]
            self.progress.push_print("processing component "+str(i)+": "+component.name)
            if i==idx:
                invM = component.inv_M_bmat_tot
            else:
                invM = bmat_mul(invM, component.inv_M_bmat_tot)        
            self.progress.pop()
    
        # calculate bulk field right-to-left in path A
        self.progress.push_print("calculating right-to-left bulk field in path A")   
        self.__bulk_field_RTL_A = clsLightField(self.grid)
        
        if isinstance(invM, clsBlockMatrix):
            if not self.get_output_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_output_field(Side.LEFT, Path.A).apply_TR_mat(invM.get_block(0,0)))
            
            if not self.get_output_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_output_field(Side.LEFT, Path.B).apply_TR_mat(invM.get_block(0,1)))
            
            if not self.get_incident_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_incident_field(Side.LEFT, Path.A).apply_TR_mat(invM.get_block(0,2)))
            
            if not self.get_incident_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_incident_field(Side.LEFT, Path.B).apply_TR_mat(invM.get_block(0,3)))    
            self.progress.pop()
            
            # calculate bulk field right-to-left in path B
            self.progress.push_print("calculating right-to-left bulk field in path B")
            self.__bulk_field_RTL_B = clsLightField(self.grid)
            if not self.get_output_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_output_field(Side.LEFT, Path.A).apply_TR_mat(invM.get_block(1,0)))
            
            if not self.get_output_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_output_field(Side.LEFT, Path.B).apply_TR_mat(invM.get_block(1,1)))
            
            if not self.get_incident_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_incident_field(Side.LEFT, Path.A).apply_TR_mat(invM.get_block(1,2)))
            
            if not self.get_incident_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_incident_field(Side.LEFT, Path.B).apply_TR_mat(invM.get_block(1,3)))
            
            self.progress.pop()
            
            # calculate bulk field left-to-right in path A
            self.progress.push_print("calculating left-to-right bulk field in path A")   
            self.__bulk_field_LTR_A = clsLightField(self.grid)
            
            if not self.get_output_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_output_field(Side.LEFT, Path.A).apply_TR_mat(invM.get_block(2,0)))
            
            if not self.get_output_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_output_field(Side.LEFT, Path.B).apply_TR_mat(invM.get_block(2,1)))
            
            if not self.get_incident_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_incident_field(Side.LEFT, Path.A).apply_TR_mat(invM.get_block(2,2)))
            
            if not self.get_incident_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_incident_field(Side.LEFT, Path.B).apply_TR_mat(invM.get_block(2,3)))    
            self.progress.pop()
            
            # calculate bulk field left-to-right in path B
            self.progress.push_print("calculating left-to-right bulk field in path B")   
            self.__bulk_field_LTR_B = clsLightField(self.grid)
            
            if not self.get_output_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_output_field(Side.LEFT, Path.A).apply_TR_mat(invM.get_block(3,0)))
            
            if not self.get_output_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_output_field(Side.LEFT, Path.B).apply_TR_mat(invM.get_block(3,1)))
            
            if not self.get_incident_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_incident_field(Side.LEFT, Path.A).apply_TR_mat(invM.get_block(3,2)))
            
            if not self.get_incident_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_incident_field(Side.LEFT, Path.B).apply_TR_mat(invM.get_block(3,3)))    
            self.progress.pop()
        
        else:
        
            if not self.get_output_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_output_field(Side.LEFT, Path.A).apply_TR_mat(invM[0][0]))
            
            if not self.get_output_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_output_field(Side.LEFT, Path.B).apply_TR_mat(invM[0][1]))
            
            if not self.get_incident_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_incident_field(Side.LEFT, Path.A).apply_TR_mat(invM[0][2]))
            
            if not self.get_incident_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_incident_field(Side.LEFT, Path.B).apply_TR_mat(invM[0][3]))    
            self.progress.pop()
            
            # calculate bulk field right-to-left in path B
            self.progress.push_print("calculating right-to-left bulk field in path B")
            self.__bulk_field_RTL_B = clsLightField(self.grid)
            if not self.get_output_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_output_field(Side.LEFT, Path.A).apply_TR_mat(invM[1][0]))
            
            if not self.get_output_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_output_field(Side.LEFT, Path.B).apply_TR_mat(invM[1][1]))
            
            if not self.get_incident_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_incident_field(Side.LEFT, Path.A).apply_TR_mat(invM[1][2]))
            
            if not self.get_incident_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_incident_field(Side.LEFT, Path.B).apply_TR_mat(invM[1][3]))
            
            self.progress.pop()
            
            # calculate bulk field left-to-right in path A
            self.progress.push_print("calculating left-to-right bulk field in path A")   
            self.__bulk_field_LTR_A = clsLightField(self.grid)
            
            if not self.get_output_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_output_field(Side.LEFT, Path.A).apply_TR_mat(invM[2][0]))
            
            if not self.get_output_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_output_field(Side.LEFT, Path.B).apply_TR_mat(invM[2][1]))
            
            if not self.get_incident_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_incident_field(Side.LEFT, Path.A).apply_TR_mat(invM[2][2]))
            
            if not self.get_incident_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_incident_field(Side.LEFT, Path.B).apply_TR_mat(invM[2][3]))    
            self.progress.pop()
            
            # calculate bulk field left-to-right in path B
            self.progress.push_print("calculating left-to-right bulk field in path B")   
            self.__bulk_field_LTR_B = clsLightField(self.grid)
            
            if not self.get_output_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_output_field(Side.LEFT, Path.A).apply_TR_mat(invM[3][0]))
            
            if not self.get_output_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_output_field(Side.LEFT, Path.B).apply_TR_mat(invM[3][1]))
            
            if not self.get_incident_field(Side.LEFT, Path.A) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_incident_field(Side.LEFT, Path.A).apply_TR_mat(invM[3][2]))
            
            if not self.get_incident_field(Side.LEFT, Path.B) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_incident_field(Side.LEFT, Path.B).apply_TR_mat(invM[3][3]))    
            self.progress.pop()
        
            
        self.__bulk_field_pos = idx
        self.progress.pop()
        
        self.progress.pop()

    def calc_bulk_field_from_right(self, idx: int):
        """ 
        Calulates the bulk field right of component identified with idx
        based on the right incident and right output field
        """
        self.progress.push_print("calculating bulk field right of component "
                                 +str(idx)+", coming from the right side")
        if self.component_count == 0:
            self.progress.print("There are no components! End task.")
            self.__bulk_field_LTR_A = None
            self.__bulk_field_LTR_B = None
            self.__bulk_field_RTL_A = None
            self.__bulk_field_RTL_B = None
            self.__bulk_field_pos = None
            return

        if idx >= self.component_count-1:
            # right from rightmost component
            # "bulk field" is the field on the right side
            self.progress.print("'Right of index "+str(idx)+"' "+ 
                                "is outside the bulk on the right side. "+
                                "Calculating field on the right side of cavity.")
            self.__bulk_field_LTR_A = self.get_output_field(Side.RIGHT, Path.A)
            self.__bulk_field_LTR_B = self.get_output_field(Side.RIGHT, Path.B)
            self.__bulk_field_RTL_A = self.get_incident_field(Side.RIGHT, Path.A)
            self.__bulk_field_RTL_B = self.get_incident_field(Side.RIGHT, Path.B)
            if self.__bulk_field_LTR_A is None:
                self.__bulk_field_LTR_A = clsLightField(self.grid)
            if self.__bulk_field_LTR_B is None:
                self.__bulk_field_LTR_B = clsLightField(self.grid)
            if self.__bulk_field_RTL_A is None:
                self.__bulk_field_RTL_A = clsLightField(self.grid)
            if self.__bulk_field_RTL_B is None:
                self.__bulk_field_RTL_B = clsLightField(self.grid)
            self.__bulk_field_pos = self.component_count-1
            return
        
        if idx < 0:
            # right from component with idx -1 or smaller ->
            # "bulk field" is the field on the left side
            self.progress.print("'Right of index "+str(idx)+ "' "+ 
                                "is outside the bulk on the left side. "+
                                "Calculating field on the left side of cavity.")
            self.__bulk_field_LTR_A = self.get_incident_field(Side.LEFT, Path.A)
            self.__bulk_field_LTR_B = self.get_incident_field(Side.LEFT, Path.B)
            self.__bulk_field_RTL_A = self.get_output_field(Side.LEFT, Path.A)
            self.__bulk_field_RTL_B = self.get_output_field(Side.LEFT, Path.B)
            
            if self.__bulk_field_LTR_A is None:
                self.__bulk_field_LTR_A = clsLightField(self.grid)
            if self.__bulk_field_LTR_B is None:
                self.__bulk_field_LTR_B = clsLightField(self.grid)
            if self.__bulk_field_RTL_A is None:
                self.__bulk_field_RTL_A = clsLightField(self.grid)
            if self.__bulk_field_RTL_B is None:
                self.__bulk_field_RTL_B = clsLightField(self.grid)
            self.__bulk_field_pos = -1
            return
        
        for i in range(idx+1, self.component_count):
            component = self.components[i]
            self.progress.push_print("processing component "+str(i)+": "+component.name)
            if i==idx+1:
                M = component.M_bmat_tot
            else:
                M = bmat_mul(M, component.M_bmat_tot)        
            self.progress.pop()
            
        # calculate bulk field right-to-left in path A
        self.progress.push_print("calculating right-to-left bulk field in path A")   
        self.__bulk_field_RTL_A = clsLightField(self.grid)
        
        if isinstance(M, clsBlockMatrix):
            if not self.get_incident_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_incident_field(Side.RIGHT, Path.A).apply_TR_mat(M.get_block(0,0)))
            
            if not self.get_incident_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_incident_field(Side.RIGHT, Path.B).apply_TR_mat(M.get_block(0,1)))
            
            if not self.get_output_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_output_field(Side.RIGHT, Path.A).apply_TR_mat(M.get_block(0,2)))
            
            if not self.get_output_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_output_field(Side.RIGHT, Path.B).apply_TR_mat(M.get_block(0,3)))    
            self.progress.pop()
            
            # calculate bulk field right-to-left in path B
            self.progress.push_print("calculating right-to-left bulk field in path B")   
            self.__bulk_field_RTL_B = clsLightField(self.grid)
            
            if not self.get_incident_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_incident_field(Side.RIGHT, Path.A).apply_TR_mat(M.get_block(1,0)))
            
            if not self.get_incident_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_incident_field(Side.RIGHT, Path.B).apply_TR_mat(M.get_block(1,1)))
            
            if not self.get_output_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_output_field(Side.RIGHT, Path.A).apply_TR_mat(M.get_block(1,2)))
            
            if not self.get_output_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_output_field(Side.RIGHT, Path.B).apply_TR_mat(M.get_block(1,3)))    
            self.progress.pop()
            
            # calculate bulk field right-to-left in path A
            self.progress.push_print("calculating left-to-right bulk field in path A")   
            self.__bulk_field_LTR_A = clsLightField(self.grid)
            
            if not self.get_incident_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_incident_field(Side.RIGHT, Path.A).apply_TR_mat(M.get_block(2,0)))
            
            if not self.get_incident_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_incident_field(Side.RIGHT, Path.B).apply_TR_mat(M.get_block(2,1)))
            
            if not self.get_output_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_output_field(Side.RIGHT, Path.A).apply_TR_mat(M.get_block(2,2)))
            
            if not self.get_output_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_output_field(Side.RIGHT, Path.B).apply_TR_mat(M.get_block(2,3)))    
            self.progress.pop()
            
            # calculate bulk field right-to-left in path B
            self.progress.push_print("calculating left-to-right bulk field in path B")   
            self.__bulk_field_LTR_B = clsLightField(self.grid)
            
            if not self.get_incident_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_incident_field(Side.RIGHT, Path.A).apply_TR_mat(M.get_block(3,0)))
            
            if not self.get_incident_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_incident_field(Side.RIGHT, Path.B).apply_TR_mat(M.get_block(3,1)))
            
            if not self.get_output_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_output_field(Side.RIGHT, Path.A).apply_TR_mat(M.get_block(3,2)))
            
            if not self.get_output_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_output_field(Side.RIGHT, Path.B).apply_TR_mat(M.get_block(3,3)))    
            self.progress.pop()
        
        else:
            if not self.get_incident_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_incident_field(Side.RIGHT, Path.A).apply_TR_mat(M[0][0]))
            
            if not self.get_incident_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_incident_field(Side.RIGHT, Path.B).apply_TR_mat(M[0][1]))
            
            if not self.get_output_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_output_field(Side.RIGHT, Path.A).apply_TR_mat(M[0][2]))
            
            if not self.get_output_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_RTL_A.add(
                    self.get_output_field(Side.RIGHT, Path.B).apply_TR_mat(M[0][3]))    
            self.progress.pop()
            
            # calculate bulk field right-to-left in path B
            self.progress.push_print("calculating right-to-left bulk field in path B")   
            self.__bulk_field_RTL_B = clsLightField(self.grid)
            
            if not self.get_incident_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_incident_field(Side.RIGHT, Path.A).apply_TR_mat(M[1][0]))
            
            if not self.get_incident_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_incident_field(Side.RIGHT, Path.B).apply_TR_mat(M[1][1]))
            
            if not self.get_output_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_output_field(Side.RIGHT, Path.A).apply_TR_mat(M[1][2]))
            
            if not self.get_output_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_RTL_B.add(
                    self.get_output_field(Side.RIGHT, Path.B).apply_TR_mat(M[1][3]))    
            self.progress.pop()
            
            # calculate bulk field right-to-left in path A
            self.progress.push_print("calculating left-to-right bulk field in path A")   
            self.__bulk_field_LTR_A = clsLightField(self.grid)
            
            if not self.get_incident_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_incident_field(Side.RIGHT, Path.A).apply_TR_mat(M[2][0]))
            
            if not self.get_incident_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_incident_field(Side.RIGHT, Path.B).apply_TR_mat(M[2][1]))
            
            if not self.get_output_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_output_field(Side.RIGHT, Path.A).apply_TR_mat(M[2][2]))
            
            if not self.get_output_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_LTR_A.add(
                    self.get_output_field(Side.RIGHT, Path.B).apply_TR_mat(M[2][3]))    
            self.progress.pop()
            
            # calculate bulk field right-to-left in path B
            self.progress.push_print("calculating left-to-right bulk field in path B")   
            self.__bulk_field_LTR_B = clsLightField(self.grid)
            
            if not self.get_incident_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_incident_field(Side.RIGHT, Path.A).apply_TR_mat(M[3][0]))
            
            if not self.get_incident_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_incident_field(Side.RIGHT, Path.B).apply_TR_mat(M[3][1]))
            
            if not self.get_output_field(Side.RIGHT, Path.A) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_output_field(Side.RIGHT, Path.A).apply_TR_mat(M[3][2]))
            
            if not self.get_output_field(Side.RIGHT, Path.B) is None:
                self.__bulk_field_LTR_B.add(
                    self.get_output_field(Side.RIGHT, Path.B).apply_TR_mat(M[3][3]))    
            self.progress.pop()

        self.__bulk_field_pos = idx
        self.progress.pop()
        
        self.progress.pop()
        
    def add_2port_component(self, A: clsOptComponent2port, B: clsOptComponent2port ):
        """ 
        adds an optical 2-port component (mirror, lens, propagation, etc) 
        to Path.A and/or Path.B
        
        A ... 2-port component to be added to path A (None if no component)
        B ... 2-port component to be added to path B (None if no component)
        """
        if A is None and B is None:
            return
        
        # Look if exactly these components already have been added
        found = False
        idx = 0
        for c in self.components:
            if isinstance(c, clsOptComponentAdapter):
                if c.component_A is A and c.component_B is B:
                    found = True
                    break
            idx += 1
            
        if found:
            print("re-using component",idx,"as component",self.component_count)
            self.components.append(c)
            return
            
        adapter = clsOptComponentAdapter("", self)
        self.components.append(adapter)
        adapter._connect_to_cavity(self, self.grid, self.component_count-1)
        if not A is None:
            adapter.connect_component(A, Path.A)
        if not B is None:
            adapter.connect_component(B, Path.B)                
            
        
    def add_4port_component(self, component: clsOptComponent4port):
        """
        adds an optical 4-port component (beam splitter, etc)
        """
        self.components.append(component)
        component._connect_to_cavity(self, self.grid, self.component_count-1)
 
    def calc_R_L_mat_tot(self):
        """ 
        calculates the left reflection 2x2 block matrix for the whole cavity 
        (which is the top-left entry of the cavity's scattering 4x4 block matrix)
        """ 
        self.progress.push_print("calculating cavity's reflection 2x2 block matrix R_L from cavity's transfer 4x4 block matrix M")
        if self.M_bmat_tot is None:
            self.calc_M_bmat_tot()
            
        if isinstance(self.M_bmat_tot, clsBlockMatrix):
            B = self.M_bmat_tot.get_quadrant(2)
            D = self.M_bmat_tot.get_quadrant(4)
            
            p = self.progress
            p.tic_reset(5+8)
            tmp = bmat2_inv(D, p) # inv: 5 steps
            self.R_L_mat_tot = bmat_mul(B, tmp, p) # mult: 8 steps
        
        else:
            A, B, C, D = get_bmat4_quadrants(self.M_bmat_tot)
            self.R_L_mat_tot = bmat_mul(B,bmat2_inv(D))
            
        self.progress.pop()
        
    def calc_R_R_mat_tot(self):
        """ 
        calculates the right reflection 2x2 block matrix for the whole cavity 
        (which is the bottom-right entry of the cavity's scattering 4x4 block matrix)
        """ 
        self.progress.push_print("calculating cavity's reflection 2x2 block matrix R_R from cavity's transfer 4x4 block matrix M")
        if self.M_bmat_tot is None:
            self.calc_M_bmat_tot()
            
        if isinstance(self.M_bmat_tot, clsBlockMatrix):
            C = self.M_bmat_tot.get_quadrant(3) 
            D = self.M_bmat_tot.get_quadrant(4)
            
            p = self.progress
            p.tic_reset(5+8)
            tmp = bmat2_inv(D, p) # inv: 5 steps
            self.R_R_mat_tot = bmat_flip_sign(bmat_mul(tmp, C, p)) # mult: 8 steps
            
        else:
            A, B, C, D = get_bmat4_quadrants(self.M_bmat_tot)
            self.R_R_mat_tot = bmat_flip_sign(bmat_mul(bmat2_inv(D), C))
        
        self.progress.pop()
        
    def calc_T_LTR_mat_tot(self):
        """ 
        calculates the left-to-right 2x2 transmission block matrix for the whole cavity 
        (which is the bottom-left entry of the cavity's 4x4 scattering block matrix)
        """        
        
        self.progress.push_print("calculating cavity's transmission 2x2 block matrix T_LTR from cavity's transfer 4x4 block matrix M")
        if self.M_bmat_tot is None:
            self.calc_M_bmat_tot()
            
        if isinstance(self.M_bmat_tot, clsBlockMatrix):
            D = self.M_bmat_tot.get_quadrant(4)
            
            p = self.progress
            p.tic_reset(5)
            self.T_LTR_mat_tot = bmat2_inv(D, p) # inv: 5 steps
        
        else:
            A, B, C, D = get_bmat4_quadrants(self.M_bmat_tot)
            self.T_LTR_mat_tot = bmat2_inv(D)
            
        self.progress.pop()
        
    def calc_T_RTL_mat_tot(self):
        """ 
        calculates the right-to-left 2x2 transmission block matrix for the whole cavity 
        (which is the top-right entry of the cavity's 4x4 scattering block matrix)
        """        
        self.progress.push_print("calculating cavity's transmission 2x2 block matrix T_RTL from cavity's transfer 4x4 block matrix M")
        if self.M_bmat_tot is None:
            self.calc_M_bmat_tot()
            
        if isinstance(self.M_bmat_tot, clsBlockMatrix):
            A = self.M_bmat_tot.get_quadrant(1)
            B = self.M_bmat_tot.get_quadrant(2)
            C = self.M_bmat_tot.get_quadrant(3) 
            D = self.M_bmat_tot.get_quadrant(4)
            
            p = self.progress
            p.tic_reset(5+2*8+1)
            
            inv_D = bmat2_inv(D, p) # inv: 5 steps
            B_inv_D = bmat_mul(B, inv_D, p) # mult: 8 steps
            tmp = bmat_mul(B_inv_D, C) # mult: 8 steps
            
            self.T_RTL_mat_tot = bmat2_minus(A, tmp)
            p.tic()
            
        else:
            A, B, C, D = get_bmat4_quadrants(self.M_bmat_tot)
            self.T_RTL_mat_tot = bmat2_minus(A, bmat_mul(bmat_mul(B, bmat2_inv(D)), C))
        
        self.progress.pop()

    def calc_S_bmat_tot(self, idx_from=0, idx_to=999):
        """ 
        calculates the scattering 4x4 block matrix for the whole cavity 
        (total resolution)
        """ 
        self.progress.push_print("calculating scattering 4x4 block matrix S for the whole cavity")
        A = self.R_L_bmat_tot
        B = self.T_RTL_bmat_tot
        C = self.T_LTR_bmat_tot
        D = self.R_R_bmat_tot
        
        if isinstance(A, clsBlockMatrix):
            self.S_bmat_tot =clsBlockMatrix(4, 
                            self.cavity.use_swap_files_in_bmatrix_class,
                            self.cavity.tmp_folder)
            
            self.S_bmat_tot.set_quadrant(1, A)
            self.S_bmat_tot.set_quadrant(2, B)
            self.S_bmat_tot.set_quadrant(3, C)
            self.S_bmat_tot.set_quadrant(4, D)
        
        else:
            self.S_bmat_tot = bmat4_from_quadrants(A, B, C, D)
            
        self.progress.pop()
        
    def convert_R_L_mat_tot_to_fov(self):
        R_L_mat_tot = self.R_L_mat_tot
        self.progress.push_print("converting R_L_mat_tot to R_L_mat_fov")        
        self.R_L_mat_fov = self.grid.convert_TR_bmat2_tot_to_fov(R_L_mat_tot)
        self.progress.pop()
        
    def convert_R_R_mat_tot_to_fov(self):
        R_R_mat_tot = self.R_R_mat_tot
        self.progress.push_print("converting R_R_mat_tot to R_R_mat_fov")        
        self.R_R_mat_fov = self.grid.convert_TR_bmat2_tot_to_fov(R_R_mat_tot)
        self.progress.pop()
            
    def convert_T_LTR_mat_tot_to_fov(self):
        T_LTR_mat_tot = self.T_LTR_mat_tot
        self.progress.push_print("converting T_LTR_mat_tot to T_LTR_mat_fov")
        self.T_LTR_mat_fov = self.grid.convert_TR_bmat2_tot_to_fov(T_LTR_mat_tot)
        self.progress.pop()            
                    
    def convert_T_RTL_mat_tot_to_fov(self):
        T_RTL_mat_tot = self.T_RTL_mat_tot
        self.progress.push_print("converting T_RTL_mat_tot to T_RTL_mat_fov")
        self.T_RTL_mat_fov = self.grid.convert_TR_bmat2_tot_to_fov(T_RTL_mat_tot)
        self.progress.pop()    
        
    def prop(self, in_A: clsLightField, in_B: clsLightField, 
             idx_from, idx_to, direction: Dir):
        """ 
        one-way-propagation of lightfields in_A and in_B 
        from left to right (direction = Dir.LTR)
        or from right-to-left (direction = Dir.RTL)
        from optical component indicated by idx_from (incl) 
        to optical compenent indicated by idx_to (incl) 
        """
        if idx_from<0:
            idx_from = 0
        
        if idx_to>=self.component_count:
            idx_to  = self.component_count-1
            
        if in_A is None:
            out_A = clsLightField()
        else:
            out_A = in_A
        
        if in_B is None:
            out_B = clsLightField()
        else:
            out_B = in_B
            
        if direction == Dir.LTR:
            # left-to-right propagation
            if idx_from > idx_to:
                idx_from, idx_to = idx_to, idx_from
            idx_to += 1
            idx_dir = 1
            
        else:
            # right-to-left propagation
            if idx_from < idx_to:
                idx_from, idx_to = idx_to, idx_from
            idx_to -= 1
            idx_dir = -1
            
        for idx in range(idx_from, idx_to, idx_dir):
            component = self.get_component(idx)
            print("processing component",idx,":",component.name)
            out_A, out_B = component.prop(out_A, out_B, direction)
            
        return out_A, out_B
    
    def get_dist_phys(self, idx1=0, idx2=999):
        """ 
        returns the physical distance in meters between 
        the left side of the component identified by idx1
        and the right side of the component identified by idx2
        """
        
        idx2 += 1
        if idx1<0:
            idx1 = 0
        if idx2>self.component_count:
            idx2 = self.component_count
            
        dist_A, dist_B = 0, 0
        for i in range(idx1, idx2):
            component = self.components[i]
            da, db = component.get_dist_phys()
            dist_A += da
            dist_B += db
        
        return dist_A, dist_B
    
    def get_dist_opt(self, idx1=0, idx2=999):
        """ 
        returns the optical distance in meters between 
        the left side of the component identified by idx1
        and the right side of the component identified by idx2
        """
        
        idx2 += 1
        if idx1<0:
            idx1 = 0
        if idx2>self.component_count:
            idx2 = self.component_count
            
        dist_A, dist_B = 0, 0
        for i in range(idx1, idx2):
            component = self.components[i]
            da, db = component.get_dist_opt()
            dist_A += da
            dist_B += db
        
        return dist_A, dist_B