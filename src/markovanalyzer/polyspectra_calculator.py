# This file is part of QuantumPolyspectra: a Python Package for the
# Analysis and Simulation of Quantum Measurements
#
#    Copyright (c) 2020 and later, Markus Sifft and Daniel Hägele.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
from numpy.linalg import inv, eig
from scipy.linalg import eig
from numba import njit

from itertools import permutations
from cachetools import cached
from cachetools import LRUCache
from cachetools.keys import hashkey
import psutil
from tqdm import tqdm_notebook
import pickle

import arrayfire as af
from arrayfire.interop import from_ndarray as to_gpu

from signalsnap.spectrum_plotter import SpectrumPlotter
from signalsnap.plot_config import PlotConfig

#  from pympler import asizeof

# ------ setup caches for a speed up when summing over all permutations -------
cache_fourier_g_prim = LRUCache(maxsize=int(200))
cache_first_matrix_step = LRUCache(maxsize=int(10e3))
cache_second_matrix_step = LRUCache(maxsize=int(10e3))
cache_third_matrix_step = LRUCache(maxsize=int(10e3))
cache_second_term = LRUCache(maxsize=int(20e5))
cache_third_term = LRUCache(maxsize=int(20e5))

# ------ new cache_fourier_g_prim implementation -------
# Initial maxsize
initial_max_cache_size = 1e6  # Set to 1 to allow the first item to be cached

# Create a cache with initial maxsize
cache_dict = {'cache_fourier_g_prim': LRUCache(maxsize=initial_max_cache_size),
              'cache_first_matrix_step': LRUCache(maxsize=initial_max_cache_size),
              'cache_second_matrix_step': LRUCache(maxsize=initial_max_cache_size),
              'cache_third_matrix_step': LRUCache(maxsize=initial_max_cache_size),
              'cache_second_term': LRUCache(maxsize=initial_max_cache_size),
              'cache_third_term': LRUCache(maxsize=initial_max_cache_size)}


def clear_cache_dict():
    for key in cache_dict.keys():
        cache_dict[key].clear()


# Function to get available GPU memory in bytes
def get_free_gpu_memory():
    device_props = af.device_info()
    return device_props['device_memory'] * 1024 * 1024


def get_free_system_memory():
    return psutil.virtual_memory().available


@cached(cache=cache_dict['cache_fourier_g_prim'],
        key=lambda nu, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0: hashkey(
            nu))
def _fourier_g_prim_gpu(nu, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates the fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143

    Parameters
    ----------
    nu : float
        The desired frequency
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    Fourier_G : array
        Fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143
    """

    small_indices = np.abs(eigvals.to_ndarray()) < 1e-12
    if sum(small_indices) > 1:
        raise ValueError(f'There are {sum(small_indices)} eigenvalues smaller than 1e-12. '
                         f'The Liouvilian might have multiple steady states.')

    diagonal = 1 / (-eigvals - 1j * nu)
    diagonal[zero_ind] = gpu_0  # 0
    diag_mat = af.data.diag(diagonal, extract=False)

    tmp = af.matmul(diag_mat, eigvecs_inv)
    Fourier_G = af.matmul(eigvecs, tmp)

    return Fourier_G


@cached(cache=cache_dict['cache_fourier_g_prim'],
        key=lambda nu, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0: hashkey(
            nu))
@njit(fastmath=True)
def _fourier_g_prim_njit(nu, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates the fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143

    Parameters
    ----------
    nu : float
        The desired frequency
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    Fourier_G : array
        Fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143
    """

    small_indices = np.abs(eigvals) < 1e-12
    if sum(small_indices) > 1:
        raise ValueError(f'There are {sum(small_indices)} eigenvalues smaller than 1e-12. '
                         f'The Liouvilian might have multiple steady states.')

    # diagonal = 1 / (-eigvals - 1j * nu)
    # diagonal[zero_ind] = 0

    diagonal = np.zeros_like(eigvals)
    diagonal[~small_indices] = 1 / (-eigvals[~small_indices] - 1j * nu)
    diagonal[zero_ind] = 0

    Fourier_G = eigvecs @ np.diag(diagonal) @ eigvecs_inv

    return Fourier_G


def update_cache_size(cachename, out, enable_gpu):
    cache = cache_dict[cachename]

    if cache.maxsize == 1:

        if enable_gpu:
            # Calculate the size of the array in bytes
            # object_size = Fourier_G.elements() * Fourier_G.dtype_size()

            dims = out.dims()
            dtype_size = out.dtype_size()
            object_size = dims[0] * dims[1] * dtype_size  # For a 2D array

            # Calculate max GPU memory to use (90% of total GPU memory)
            max_gpu_memory = get_free_gpu_memory() * 0.9 / 6

            # Update the cache maxsize
            new_max_size = int(max_gpu_memory / object_size)

        else:
            # Calculate the size of the numpy array in bytes
            object_size = out.nbytes

            # Calculate max system memory to use (90% of available memory)
            max_system_memory = get_free_system_memory() * 0.9 / 6

            # Update the cache maxsize
            new_max_size = int(max_system_memory / object_size)

        cache_dict[cachename] = LRUCache(maxsize=new_max_size)


def _g_prim(t, eigvecs, eigvals, eigvecs_inv):
    """
    Calculates the fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143

    Parameters
    ----------
    t : float
        The desired time
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian

    Returns
    -------
    G_prim : array
        \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143
    """
    zero_ind = np.argmax(np.real(eigvals))
    diagonal = np.exp(eigvals * t)
    diagonal[zero_ind] = 0
    eigvecs_inv = diagonal.reshape(-1, 1) * eigvecs_inv
    G_prim = eigvecs.dot(eigvecs_inv)
    return G_prim


@cached(cache=cache_dict['cache_first_matrix_step'],
        key=lambda rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0: hashkey(omega))
def _first_matrix_step(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates first matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143. Used
    for the calculation of power- and bispectrum.
    Parameters
    ----------
    rho : array
        rho equals matmul(A,Steadystate desity matrix of the system)
    omega : float
        Desired frequency
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    out : array
        First matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143
    """

    if enable_gpu:
        G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = af.matmul(G_prim, rho)
        out = af.matmul(a_prim, rho_prim)
    else:
        G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = G_prim @ rho
        out = a_prim @ rho_prim

    return out


# ------ can be cached for large systems --------
@cached(cache=cache_dict['cache_second_matrix_step'],
        key=lambda rho, omega, omega2, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0: hashkey(
            omega, omega2))
def _second_matrix_step(rho, omega, omega2, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates second matrix multiplication in Eqs. 110 in 10.1103/PhysRevB.98.205143. Used
    for the calculation of bispectrum.
    Parameters
    ----------
    rho : array
        A @ Steadystate desity matrix of the system
    omega : float
        Desired frequency
    omega2 : float
        Frequency used in :func:_first_matrix_step
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    out : array
        second matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143
    """

    _ = omega2

    if enable_gpu:
        G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = af.matmul(G_prim, rho)
        out = af.matmul(a_prim, rho_prim)
    else:
        G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = G_prim @ rho
        out = a_prim @ rho_prim

    return out


@cached(cache=cache_dict['cache_third_matrix_step'],
        key=lambda rho, omega, omega2, omega3, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind,
                   gpu_0: hashkey(omega, omega2))
def _third_matrix_step(rho, omega, omega2, omega3, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates second matrix multiplication in Eqs. 110 in 10.1103/PhysRevB.98.205143. Used
    for the calculation of bispectrum.
    Parameters
    ----------
    rho : array
        A @ Steadystate desity matrix of the system
    omega : float
        Desired frequency
    omega2 : float
        Frequency used in :func:_first_matrix_step
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    out : array
        Third matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143
    """
    _ = omega2
    _ = omega3

    if enable_gpu:
        G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = af.matmul(G_prim, rho)
        out = af.matmul(a_prim, rho_prim)
    else:
        G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = G_prim @ rho
        out = a_prim @ rho_prim

    return out


def _matrix_step(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates one matrix multiplication in Eqs. 109 in 10.1103/PhysRevB.98.205143. Used
    for the calculation of trispectrum.
    Parameters
    ----------
    rho : array
        A @ Steadystate desity matrix of the system
    omega : float
        Desired frequency
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    out : array
        output of one matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143
    """

    if enable_gpu:
        G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = af.matmul(G_prim, rho)
        out = af.matmul(a_prim, rho_prim)
    else:
        G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = G_prim @ rho
        out = a_prim @ rho_prim
    return out


# ------- Second Term of S(4) ---------

#  @njit(parallel=True, fastmath=True)
def small_s(rho_steady, a_prim, eigvecs, eigvec_inv, enable_gpu, zero_ind, gpu_zero_mat):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the small s (Eq. 7) from 10.1103/PhysRevB.102.119901

    Parameters
    ----------
    zero_ind : int
        Index of the steadystate eigenvector
    enable_gpu : bool
        Specify if GPU should be used
    gpu_zero_mat : af array
        Zero array stored on the GPU
    rho_steady : array
        A @ Steadystate density matrix of the system
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvec_inv : array
        The inverse eigenvectors of the Liouvillian
    reshape_ind : array
        Indices that give the trace of a flattened matrix.

    Returns
    -------
    s_k : array
        Small s (Eq. 7) from 10.1103/PhysRevB.102.119901
    """

    if enable_gpu:
        s_k = to_gpu(np.zeros_like(rho_steady))

    else:
        s_k = np.zeros_like(rho_steady)

    for i in range(len(s_k)):
        if enable_gpu:
            S = gpu_zero_mat.copy()  # to_gpu(np.zeros_like(eigvecs))
        else:
            S = np.zeros_like(eigvecs)

        if i == zero_ind:
            s_k[i] = 0
        else:
            S[i, i] = 1
            if enable_gpu:
                temp1 = af.matmul(a_prim, rho_steady)
                temp2 = af.matmul(eigvec_inv, temp1)
                temp3 = af.matmul(S, temp2)
                temp4 = af.matmul(eigvecs, temp3)
                temp5 = af.matmul(a_prim, temp4)
                s_k[i] = af.algorithm.sum(temp5)
            else:
                s_k[i] = (a_prim @ eigvecs @ S @ eigvec_inv @ a_prim @ rho_steady).sum()
    return s_k


@njit(fastmath=True)
def second_term_njit(omega1, omega2, omega3, s_k, eigvals):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the second sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Second correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3

    out_sum = 0
    iterator = np.array(list(range(len(s_k))))
    iterator = iterator[np.abs(s_k) > 1e-10 * np.max(np.abs(s_k))]

    for k in iterator:
        for l in iterator:
            out_sum += s_k[k] * s_k[l] * 1 / ((eigvals[l] + 1j * nu1) * (eigvals[k] + 1j * nu3)
                                              * (eigvals[k] + eigvals[l] + 1j * nu2))

    return out_sum


def second_term_gpu(omega1, omega2, omega3, s_k, eigvals):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the second sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Second correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3

    temp1 = af.matmulNT(s_k, s_k)
    temp2 = af.matmulNT(eigvals + 1j * nu1, eigvals + 1j * nu3)
    temp3 = af.tile(eigvals, 1, eigvals.shape[0]) + af.tile(eigvals.T, eigvals.shape[0]) + 1j * nu2
    out = temp1 * 1 / (temp2 * temp3)
    out_sum = af.algorithm.sum(af.algorithm.sum(out, dim=0), dim=1)

    return out_sum


@cached(cache=cache_dict['cache_second_term'],
        key=lambda omega1, omega2, omega3, s_k, eigvals, enable_gpu: hashkey(omega1, omega2, omega3))
def second_term(omega1, omega2, omega3, s_k, eigvals, enable_gpu):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the second sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    enable_gpu : bool
        Specify if GPU should be used
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Second correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    if enable_gpu:
        return second_term_gpu(omega1, omega2, omega3, s_k, eigvals)
    else:
        return second_term_njit(omega1, omega2, omega3, s_k, eigvals)


@njit(fastmath=True)
def third_term_njit(omega1, omega2, omega3, s_k, eigvals):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the third sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Third correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    out = 0
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3
    iterator = np.array(list(range(len(s_k))))
    iterator = iterator[np.abs(s_k) > 1e-10 * np.max(np.abs(s_k))]

    for k in iterator:
        for l in iterator:
            out += s_k[k] * s_k[l] * 1 / ((eigvals[k] + 1j * nu1) * (eigvals[k] + 1j * nu3)
                                          * (eigvals[k] + eigvals[l] + 1j * nu2))
    return out


def third_term_gpu(omega1, omega2, omega3, s_k, eigvals):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the third sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Third correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3

    temp1 = af.matmulNT(s_k, s_k)
    temp2 = af.tile((eigvals + 1j * nu1) * (eigvals + 1j * nu3), 1, eigvals.shape[0])
    temp3 = af.tile(eigvals, 1, eigvals.shape[0]) + af.tile(eigvals.T, eigvals.shape[0]) + 1j * nu2
    out = temp1 * 1 / (temp2 * temp3)
    out = af.algorithm.sum(
        af.algorithm.sum(af.data.moddims(out, d0=eigvals.shape[0], d1=eigvals.shape[0], d2=1, d3=1), dim=0), dim=1)
    return out


# @njit(fastmath=True)
@cached(cache=cache_dict['cache_third_term'],
        key=lambda omega1, omega2, omega3, s_k, eigvals, enable_gpu: hashkey(omega1, omega2, omega3))
def third_term(omega1, omega2, omega3, s_k, eigvals, enable_gpu):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the third sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    enable_gpu : bool
        Specify if GPU should be used
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Third correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    if enable_gpu:
        return third_term_gpu(omega1, omega2, omega3, s_k, eigvals)
    else:
        return third_term_njit(omega1, omega2, omega3, s_k, eigvals)


# ------- Hepler functions ----------


def _full_bispec(r_in, one_quadrant=True):
    """
    Turns the partial bispectrum (only the half of quadrant) into a full plain.

    Parameters
    ----------
    r_in : array
        Partial spectrum (one twelfth of the full plane)

    Returns
    -------
    m_full : array
        Full plain of spectrum
    """
    r = np.flipud(r_in)
    s, t = r.shape
    m = 1j * np.zeros((2 * s - 1, 2 * s - 1))
    r_padded = np.vstack((r, np.zeros((s - 1, s))))
    r_rolled = np.empty_like(r_padded)
    for i in range(s):
        r_rolled[:, i] = np.roll(r_padded[:, i], -i)
    r_left = r_rolled[:s, :]
    r_mirrored = r_left + np.flipud((np.flipud(r_left)).T) - np.fliplr(np.diag(np.diagonal(np.fliplr(r_left))))
    r_top_left = np.fliplr(r_mirrored)
    if one_quadrant:
        return np.flipud(r)
    m[:s, :s] = r_top_left
    m[:s, s - 1:] = r
    m_full = np.fliplr(np.flipud(m)) + m
    m_full[s - 1, :] -= m[s - 1, :]
    return np.fliplr(m_full)


def _full_trispec(r_in, one_quadrand=True):
    """
    Turns the partial trispectrum (only the half of quadrant) into a full plain.

    Parameters
    ----------
    r_in : array
        Partial spectrum
    Returns
    -------
    m : array
        Full plain of spectrum
    """
    r = np.flipud(r_in)
    if one_quadrand:
        return r_in
    s, t = r.shape
    m = 1j * np.zeros((2 * s - 1, 2 * s - 1))
    m[:s, s - 1:] = r
    m[:s, :s - 1] = np.fliplr(r)[:, :-1]
    m[s:, :] = np.flipud(m[:s - 1, :])
    return m


def pickle_save(path, obj):
    """
    Helper function to pickle system objects

    Parameters
    ----------
    path : str
        Location of saved data
    obj : System obj

    """
    f = open(path, mode='wb')
    pickle.dump(obj, f)
    f.close()


import numpy as np


def rates_to_matrix(rates):
    """
    Convert a dictionary of rates to a continuous-time Markov process transition rate matrix.

    Parameters
    ----------
    rates : dict
        Dictionary containing the rates for each transition. Keys should be in the format
        'from_state->to_state' and values should be non-negative floats representing the rates.

    Returns
    -------
    np.ndarray
        A NumPy array representing the transition rate matrix.

    Raises
    ------
    ValueError
        If the keys in the dictionary are not in the 'from_state->to_state' format or if rates are negative.

    Example
    -------
    >>> rates = {"1->2": 0.5, "2->1": 0.7, "2->3": 0.2}
    >>> rates_to_matrix(rates)
    array([[-0.5,  0.5,  0. ],
           [ 0.7, -0.9,  0.2],
           [ 0. ,  0. ,  0. ]])
    """

    # Validate input and identify unique states
    states = set()
    for key, rate in rates.items():
        try:
            from_state, to_state = key.split("->")
            rate = float(rate)
        except ValueError:
            raise ValueError(
                "Invalid key or rate value. Keys should be 'from_state->to_state' and rates should be non-negative numbers.")

        if rate < 0:
            raise ValueError("Rates should be non-negative numbers.")

        states.add(from_state)
        states.add(to_state)

    states = sorted(list(states))
    n = len(states)

    # Initialize a zero matrix
    matrix = np.zeros((n, n))

    # Fill the transition rates
    for key, rate in rates.items():
        from_state, to_state = key.split("->")
        i, j = states.index(from_state), states.index(to_state)
        matrix[i, j] = rate

    # Fill the diagonal elements such that each row sums to zero
    for i in range(n):
        matrix[i, i] = -sum(matrix[i])

    return matrix.T


class System:  # (SpectrumCalculator):
    """
    Class that will represent the system of interest. It contains the parameters of the system and the
    methods for calculating and storing the polyspectra.

    Parameters
    ----------

    Attributes
    ----------
    freq : dict
        Stores the frequencies from the analytic spectra, order 2 to 4
    S : dict
        Stores the analytic spectra, order 2 to 4
    numeric_f_data : dict
        Stores the frequencies from the numeric spectra, order 2 to 4
    numeric_spec_data : dict
        Stores the numeric spectra, order 2 to 4
    eigvals : array
        Stores eigenvalues of Liouvillian
    eigvecs : array
        Stores eigenvectors of Liouvillian
    eigvecs_inv : array
        Stores the matrix inversion of the eigenvector matrix
    zero_ind : int
        Contains the index of the steady state in the eigvalues
    A_prim : array
        Stores the measurement superoperator \mathcal{A} as defined in 10.1103/PhysRevB.98.205143
    rho_steady : array
        Steady state of the Liouvillian
    s_k : array
        Stores small s (Eq. 7) from 10.1103/PhysRevB.102.119901
    N : int
        Number of points in time series in window for the calculation of numerical spectra
    fs : float
        Sampling rate of the simulated signal for numerical spectra
    a_w : array
        Fourier coefficients of simulated signal for numerical spectra
    a_w_cut : array
        Contains only the frequencies of interest from a_w (to speed up calculations)
    enable_gpu : bool
        Set if GPU should be used for analytic spectra calculation
    gpu_0 : int
        Stores pointer to zero an the GPU
    """

    def __init__(self, transition_dict, measurement_op, gamma_ph=None, gamma_det=None):

        self.measurement_op = measurement_op

        if gamma_det is not None and gamma_ph is not None:
            transition_dict = self.extension_for_single_photon(transition_dict, measurement_op, gamma_ph, gamma_det)
            self.measurement_op = self.transform_m_op(measurement_op)

        self.transtion_matrix = rates_to_matrix(transition_dict)

        self.freq = {2: np.array([]), 3: np.array([]), 4: np.array([])}
        self.S = {1: 0, 2: np.array([]), 3: np.array([]), 4: np.array([])}

        self.numeric_f_data = {2: np.array([]), 3: np.array([]), 4: np.array([])}
        self.numeric_spec_data = {2: np.array([]), 3: np.array([]), 4: np.array([])}

        self.eigvals = np.array([])
        self.eigvecs = np.array([])
        self.eigvecs_inv = np.array([])
        self.zero_ind = 0
        self.A_prim = np.array([])
        self.rho_steady = 0
        self.s_k = 0

        self.N = None  # Number of points in time series
        self.fs = None
        self.a_w = None
        self.a_w_cut = None

        # ------- Enable GPU for large systems -------
        self.enable_gpu = False
        self.gpu_0 = 0

    def transform_m_op(self, old_m_op):
        """
        Transforms the input array old_m_op to a new array new_m_op.
        The new array is twice as long, with the original entries set to 0,
        and the new entries set to 1.

        Parameters
        ----------
        old_m_op : numpy.ndarray
            Original m_op array.

        Returns
        -------
        new_m_op : numpy.ndarray
            Transformed m_op array.

        Example
        -------
        >>> old_m_op = np.array([1, 0, 0])
        >>> transform_m_op(old_m_op)
        array([0, 0, 0, 1, 1, 1])
        """

        # Set all entries in the old_m_op to 0
        old_m_op_zeroed = np.zeros_like(old_m_op)

        # Create an array of 1s with the same shape as old_m_op
        extended_part = np.ones_like(old_m_op)

        # Concatenate old_m_op_zeroed and extended_part to form new_m_op
        new_m_op = np.concatenate((old_m_op_zeroed, extended_part))

        return new_m_op

    def save_spec(self, path):
        """
        Save System class with spectral data

        Parameters
        ----------
        path : str
            Location of file
        """
        self.gpu_0 = 0
        self.eigvals = np.array([])
        self.eigvecs = np.array([])
        self.eigvecs_inv = np.array([])
        self.A_prim = np.array([])
        self.rho_steady = 0
        self.s_k = 0

        pickle_save(path, self)

    def g_prim(self, t):
        """
        Helper method to move function out of the class. njit is not working within classes
        """
        return _g_prim(t, self.eigvecs, self.eigvals, self.eigvecs_inv)

    def first_matrix_step(self, rho, omega):
        """
        Helper method to move function out of the class. njit is not working within classes
        """
        return _first_matrix_step(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                                  self.enable_gpu, self.zero_ind, self.gpu_0)

    def second_matrix_step(self, rho, omega, omega2):
        """
        Helper method to move function out of the class. njit is not working within classes
        """
        return _second_matrix_step(rho, omega, omega2, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                                   self.enable_gpu, self.zero_ind, self.gpu_0)

    def matrix_step(self, rho, omega):
        """
        Helper method to move function out of the class. njit is not working within classes
        """
        return _matrix_step(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                            self.enable_gpu, self.zero_ind, self.gpu_0)

    def plot(self, plot_orders=(2, 3, 4)):
        config = PlotConfig(plot_orders=plot_orders, s2_f=self.freq[2], s2_data=self.S[2], s3_f=self.freq[3],
                            s3_data=self.S[3], s4_f=self.freq[4], s4_data=self.S[4])

        self.f_lists = {1: None, 2: None, 3: None, 4: None}
        self.S_err = {1: None, 2: None, 3: None, 4: None}
        self.config = config
        self.config.f_unit = 'Hz'
        plot = SpectrumPlotter(self, config)

        if self.S[1] is not None:
            print('s1:', self.S[1])
        fig = plot.plot()
        return fig

    def calculate_spectrum(self, f_data, order_in, bar=True, verbose=False,
                               correction_only=False, beta_offset=True, enable_gpu=False, cache_trispec=True):

        if order_in == 'all':
            orders = [1, 2, 3, 4]
        else:
            orders = order_in

        for order in orders:
            self.calculate_one_spectrum(f_data, order, bar=bar, verbose=verbose,
                               correction_only=correction_only, beta_offset=beta_offset, enable_gpu=enable_gpu, cache_trispec=cache_trispec)

    def replicate_and_extend_rates(self, rates):
        """
        Replicates the rates dictionary by adding the highest state number + 1 to all states.
        The replicated rates are then added to the original rates dictionary.

        Parameters
        ----------
        rates : dict
            Original rates dictionary where keys are in 'from_state->to_state' format and values are the rates.

        Returns
        -------
        dict
            Extended rates dictionary containing both the original and replicated rates.

        Example
        -------
        >>> rates = {'0->1': 'gamma_in', '0->2': 'gamma_in', '1->0': 'gamma_A', '1->2': 'gamma_spin + gamma_R', '2->1': 'gamma_spin'}
        >>> replicate_and_extend_rates(rates)
        {'0->1': 'gamma_in',
         '0->2': 'gamma_in',
         '1->0': 'gamma_A',
         '1->2': 'gamma_spin + gamma_R',
         '2->1': 'gamma_spin',
         '3->4': 'gamma_in',
         '3->5': 'gamma_in',
         '4->3': 'gamma_A',
         '4->5': 'gamma_spin + gamma_R',
         '5->4': 'gamma_spin'}
        """
        # Find the highest state number in the original rates dictionary
        highest_state = max([int(state) for key in rates for state in key.split("->")])

        # The new state numbers should start from highest_state + 1
        offset = highest_state + 1

        # Create the replicated rates dictionary
        rates_2 = {}
        for key, value in rates.items():
            from_state, to_state = map(int, key.split("->"))
            new_key = f"{from_state + offset}->{to_state + offset}"
            rates_2[new_key] = value

        # Extend the original rates dictionary with the replicated rates
        rates.update(rates_2)

        return rates

    def extension_for_single_photon(self, rates, m_op, gamma_ph, gamma_det):
        """
        Adds further connections to the rates dictionary based on the input array m_op and float gamma_det.

        Parameters
        ----------
        rates : dict
            Existing rates dictionary where keys are in 'from_state->to_state' format and values are the rates.
        m_op : list
            Array of rates used to connect the original levels with their corresponding replicated levels.
        gamma_det : float
            Rate used to connect each replicated level back to its original level.

        Returns
        -------
        dict
            Extended rates dictionary containing both the original, replicated, and new connection rates.

        Example
        -------
        >>> rates = {'0->1': 'gamma_in', '0->2': 'gamma_in', '1->0': 'gamma_A', '1->2': 'gamma_spin + gamma_R', '2->1': 'gamma_spin'}
        >>> m_op = ['m_op[0]', 'm_op[1]', 'm_op[2]']
        >>> gamma_det = 'gamma_det'
        >>> extension_for_single_photon(rates, m_op, gamma_det)
        {
            '0->1': 'gamma_in',
            '0->2': 'gamma_in',
            '1->0': 'gamma_A',
            '1->2': 'gamma_spin + gamma_R',
            '2->1': 'gamma_spin',
            '3->4': 'gamma_in',
            '3->5': 'gamma_in',
            '4->3': 'gamma_A',
            '4->5': 'gamma_spin + gamma_R',
            '5->4': 'gamma_spin',
            '0->3': 'm_op[0]',
            '1->4': 'm_op[1]',
            '2->5': 'm_op[2]',
            '3->0': 'gamma_det',
            '4->1': 'gamma_det',
            '5->2': 'gamma_det'
        }
        """
        # Replicate and extend the existing rates
        extended_rates = self.replicate_and_extend_rates(rates)

        # Add further connections based on m_op
        for i, emits in enumerate(m_op):
            # Find the replicated state corresponding to the original state
            replicated_state = i + len(m_op)
            new_key = f"{i}->{replicated_state}"
            if emits == 1:
                extended_rates[new_key] = gamma_ph
            else:
                extended_rates[new_key] = 0

        # Add connections from each replicated level back to its original level
        for i in range(len(m_op)):
            # Find the replicated state corresponding to the original state
            replicated_state = i + len(m_op)
            new_key = f"{replicated_state}->{i}"
            extended_rates[new_key] = gamma_det

        return extended_rates


    def calculate_one_spectrum(self, f_data, order, bar=True, verbose=False,
                               correction_only=False, beta_offset=True, enable_gpu=False, cache_trispec=True):
        """
        Calculates analytic polyspectra (order 2 to 4) as described in 10.1103/PhysRevB.98.205143
        and 10.1103/PhysRevB.102.119901

        Parameters
        ----------
        f_data : array
            Frequencies at which the spectra are calculated
        order : int {2,3,4}
            Order of the polyspectra to be calculated
        g_prim : bool
            Set if mathcal_a should be applied twice/squared (was of use when defining the current operator)
            But unnecessary for standard polyspectra
        bar : bool
            Set if progress bars should be shown during calculation
        verbose : bool
            Set if more details about the current state of the calculation are needed
        correction_only : bool
            Set if only the correction terms of the S4 from erratum 10.1103/PhysRevB.102.119901 should be
            calculated
        beta_offset : bool
            Set if constant offset due to deetector noise should be added to the power spectrum
        enable_gpu : bool
            Set if GPU should be used for calculation
        cache_trispec : bool
            Set if Matrix multiplication in the calculation of the trispectrum should be cached

        Returns
        -------
        S[order] : array
            Returns spectral value at specified frequencies

        """

        self.enable_gpu = enable_gpu
        af.device_gc()
        clear_cache_dict()

        if f_data[0] < 0:
            print('Only positive frequencies allowed')
            return None

        omegas = 2 * np.pi * f_data  # [kHz]
        self.freq[order] = f_data

        n_states = self.transtion_matrix.shape[0]
        self.eigvals, self.eigvecs = eig(self.transtion_matrix)
        self.eigvecs_inv = inv(self.eigvecs)

        self.eigvals = self.eigvals.astype(dtype=np.complex128)
        self.eigvecs = self.eigvecs.astype(dtype=np.complex128)
        self.eigvecs_inv = self.eigvecs_inv.astype(dtype=np.complex128)

        self.zero_ind = np.argmax(np.real(self.eigvals))

        rho_steady = self.eigvecs[:, self.zero_ind]
        rho_steady = rho_steady / np.sum(rho_steady)

        self.rho_steady = rho_steady

        if order == 2:
            spec_data = 1j * np.ones_like(omegas)
        elif order == 3 or order == 4:
            spec_data = 1j * np.zeros((len(omegas), len(omegas)))

        if type(self.rho_steady) == af.array.Array:
            rho_steady = self.rho_steady.to_ndarray()
        else:
            rho_steady = self.rho_steady

        self.A_prim = np.diag(self.measurement_op) - np.eye(n_states) * np.sum((self.measurement_op @ rho_steady))

        rho = self.A_prim @ rho_steady

        n_states = rho_steady.shape[0]

        if self.enable_gpu:
            if type(self.eigvals) != af.array.Array:
                self.eigvals, self.eigvecs, self.eigvecs_inv = to_gpu(self.eigvals), to_gpu(self.eigvecs), to_gpu(
                    self.eigvecs_inv)

                self.rho_steady = to_gpu(self.rho_steady)
                self.gpu_0 = to_gpu(np.array([0. * 1j]))

            self.A_prim = to_gpu(self.A_prim)
            rho = to_gpu(rho)
            measurement_op = to_gpu(self.measurement_op)

            if order == 2:
                rho_prim_sum = to_gpu(1j * np.zeros((len(omegas), n_states)))
            elif order == 3:
                rho_prim_sum = to_gpu(1j * np.zeros((len(omegas), len(omegas), n_states)))
            elif order == 4:
                rho_prim_sum = to_gpu(1j * np.zeros((len(omegas), len(omegas), n_states)))
                second_term_mat = to_gpu(1j * np.zeros((len(omegas), len(omegas))))
                third_term_mat = to_gpu(1j * np.zeros((len(omegas), len(omegas))))

        else:
            self.gpu_0 = 0
            measurement_op = self.measurement_op

        # estimate necessary cachesize (TODO: Anteile könnten noch anders gewählt werden)
        # update_cache_size('cache_fourier_g_prim', self.A_prim, enable_gpu)
        # update_cache_size('cache_first_matrix_step', rho, enable_gpu)
        # update_cache_size('cache_second_matrix_step', rho, enable_gpu)
        # update_cache_size('cache_third_matrix_step', rho, enable_gpu)
        # update_cache_size('cache_second_term', rho[0], enable_gpu)
        # update_cache_size('cache_third_term', rho[0], enable_gpu)

        if order == 1:
            if bar:
                print('Calculating first order')
            if enable_gpu:
                rho = af.matmul(measurement_op, self.rho_steady)
                self.S[order] = af.algorithm.sum(rho)
            else:
                rho = measurement_op @ self.rho_steady
                self.S[order] = rho.sum()

        if order == 2:
            if bar:
                print('Calculating power spectrum')
                counter = tqdm_notebook(enumerate(omegas), total=len(omegas))
            else:
                counter = enumerate(omegas)
            for (i, omega) in counter:
                rho_prim = self.first_matrix_step(rho, omega)  # measurement_op' * G'
                rho_prim_neg = self.first_matrix_step(rho, -omega)

                if enable_gpu:
                    rho_prim_sum[i, :] = rho_prim + rho_prim_neg
                else:
                    spec_data[i] = rho_prim.sum() + rho_prim_neg.sum()

            if enable_gpu:
                spec_data = af.algorithm.sum(rho_prim_sum, dim=1).to_ndarray()

            self.S[order] = spec_data
            self.S[order] = self.S[order]
            if beta_offset:
                self.S[order] += 1 / 4
        if order == 3:
            if bar:
                print('Calculating bispectrum')
                counter = tqdm_notebook(enumerate(omegas), total=len(omegas))
            else:
                counter = enumerate(omegas)
            for ind_1, omega_1 in counter:
                for ind_2, omega_2 in enumerate(omegas[ind_1:]):
                    # Calculate all permutation for the trace_sum
                    var = np.array([omega_1, omega_2, - omega_1 - omega_2])
                    perms = list(permutations(var))
                    trace_sum = 0
                    for omega in perms:
                        rho_prim = self.first_matrix_step(rho, omega[2] + omega[1])
                        rho_prim = self.second_matrix_step(rho_prim, omega[1], omega[2] + omega[1])
                        if enable_gpu:
                            rho_prim_sum[ind_1, ind_2 + ind_1, :] += af.data.moddims(rho_prim, d0=1, d1=1,
                                                                                     d2=n_states)
                        else:
                            trace_sum += rho_prim.sum()

                    if not enable_gpu:
                        spec_data[ind_1, ind_2 + ind_1] = trace_sum

            if enable_gpu:
                spec_data = af.algorithm.sum(rho_prim_sum, dim=2).to_ndarray()

            spec_data[(spec_data == 0).nonzero()] = spec_data.T[(spec_data == 0).nonzero()]

            if np.max(np.abs(np.imag(np.real_if_close(_full_bispec(spec_data))))) > 0 and verbose:
                print('Bispectrum might have an imaginary part')

            self.S[order] = _full_bispec(spec_data)

        if order == 4:
            if bar:
                print('Calculating correlation spectrum')
                counter = tqdm_notebook(enumerate(omegas), total=len(omegas))
            else:
                counter = enumerate(omegas)

            if verbose:
                print('Calculating small s')
            if enable_gpu:
                gpu_zero_mat = to_gpu(np.zeros_like(self.eigvecs))  # Generate the zero array only ones
            else:
                gpu_zero_mat = 0
            #  gpu_ones_arr = to_gpu(0*1j + np.ones(len(self.eigvecs[0])))
            s_k = small_s(self.rho_steady, self.A_prim, self.eigvecs, self.eigvecs_inv,
                          enable_gpu, self.zero_ind, gpu_zero_mat)

            if verbose:
                print('Done')

            self.s_k = s_k

            for ind_1, omega_1 in counter:

                for ind_2, omega_2 in enumerate(omegas[ind_1:]):
                    # for ind_2, omega_2 in enumerate(omegas[:ind_1+1]):

                    # Calculate all permutation for the trace_sum
                    var = np.array([omega_1, -omega_1, omega_2, -omega_2])
                    perms = list(permutations(var))
                    trace_sum = 0
                    second_term_sum = 0
                    third_term_sum = 0

                    if correction_only:

                        for omega in perms:
                            second_term_sum += second_term(omega[1], omega[2], omega[3], s_k, self.eigvals)
                            third_term_sum += third_term(omega[1], omega[2], omega[3], s_k, self.eigvals)

                        spec_data[ind_1, ind_2 + ind_1] = second_term_sum + third_term_sum
                        spec_data[ind_2 + ind_1, ind_1] = second_term_sum + third_term_sum

                    else:

                        for omega in perms:

                            if cache_trispec:
                                rho_prim = self.first_matrix_step(rho, omega[1] + omega[2] + omega[3])
                                rho_prim = self.second_matrix_step(rho_prim, omega[2] + omega[3],
                                                                   omega[1] + omega[2] + omega[3])
                            else:
                                rho_prim = self.matrix_step(rho, omega[1] + omega[2] + omega[3])
                                rho_prim = self.matrix_step(rho_prim, omega[2] + omega[3])

                            rho_prim = self.matrix_step(rho_prim, omega[3])

                            if enable_gpu:

                                rho_prim_sum[ind_1, ind_2 + ind_1, :] += af.data.moddims(rho_prim, d0=1,
                                                                                         d1=1,
                                                                                         d2=n_states)
                                second_term_mat[ind_1, ind_2 + ind_1] += second_term(omega[1], omega[2], omega[3], s_k,
                                                                                     self.eigvals, enable_gpu)
                                third_term_mat[ind_1, ind_2 + ind_1] += third_term(omega[1], omega[2], omega[3], s_k,
                                                                                   self.eigvals, enable_gpu)
                            else:

                                trace_sum += rho_prim.sum()
                                second_term_sum += second_term(omega[1], omega[2], omega[3], s_k, self.eigvals,
                                                               enable_gpu)
                                third_term_sum += third_term(omega[1], omega[2], omega[3], s_k, self.eigvals,
                                                             enable_gpu)

                        if not enable_gpu:
                            spec_data[ind_1, ind_2 + ind_1] = second_term_sum + third_term_sum + trace_sum
                            spec_data[ind_2 + ind_1, ind_1] = second_term_sum + third_term_sum + trace_sum

            if enable_gpu:
                spec_data = af.algorithm.sum(rho_prim_sum, dim=2).to_ndarray()
                spec_data += af.algorithm.sum(af.algorithm.sum(second_term_mat + third_term_mat, dim=3),
                                              dim=2).to_ndarray()

                spec_data[(spec_data == 0).nonzero()] = spec_data.T[(spec_data == 0).nonzero()]

            if np.max(np.abs(np.imag(np.real_if_close(_full_trispec(spec_data))))) > 0:
                print('Trispectrum might have an imaginary part')

            self.S[order] = _full_trispec(spec_data)

        clear_cache_dict()
        return self.S[order]

    def plot_all(self, f_max=None):
        """
        Method for quick plotting of polyspectra

        Parameters
        ----------
        f_max : float
            Maximum frequencies upto which the spectra should be plotted

        Returns
        -------
        Returns matplotlib figure
        """
        if f_max is None:
            f_max = self.freq[2].max()
        fig = self.plot(order_in=(2, 3, 4), f_max=f_max, s2_data=self.S[2], s3_data=self.S[3], s4_data=self.S[4],
                        s2_f=self.freq[2],
                        s3_f=self.freq[3], s4_f=self.freq[4])
        return fig

    def calc_a_w3(self, a_w):
        """
        Preparation of a_(w1+w2) for the calculation of the bispectrum

        Parameters
        ----------
        a_w : array
            Fourier coefficients of signal

        Returns
        -------
        a_w3 : array
            Matrix corresponding to a_(w1+w2)
        """
        mat_size = len(self.a_w_cut)
        a_w3 = 1j * np.ones((mat_size, mat_size))
        for i in range(mat_size):
            a_w3[i, :] = a_w[i:i + mat_size]
        return a_w3.conj()
