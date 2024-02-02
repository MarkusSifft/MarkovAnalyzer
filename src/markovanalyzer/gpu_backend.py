# This file is part of MarkovAnalyzer
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


import arrayfire as af
from cachetools import LRUCache
from cachetools.keys import hashkey
import numpy as np
from cachetools import cached
import psutil
from itertools import permutations


# Function to get available GPU memory in bytes
def get_free_gpu_memory():
    device_props = af.device_info()
    return device_props['device_memory'] * 1024 * 1024


def get_free_system_memory():
    return psutil.virtual_memory().available


# ------ new cache_fourier_g_prim implementation -------
# Initial maxsize
initial_max_cache_size = 1e9  # Set to 1 to allow the first item to be cached

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


@cached(cache=cache_dict['cache_fourier_g_prim'],
        key=lambda nu, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0: hashkey(
            nu))
def _fourier_g_prim_gpu(nu, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
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


@cached(cache=cache_dict['cache_first_matrix_step'],
        key=lambda rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0: hashkey(omega))
def _first_matrix_step_gpu(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
    """
    Calculates first matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143. Used
    for the calculation of power- and bispectrum.
    Parameters
    ----------
    rho : array
        rho equals matmul(A, Steadystate desity matrix of the system)
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

    G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
    rho_prim = af.matmul(G_prim, rho)
    out = af.matmul(a_prim, rho_prim)

    return out


# ------ can be cached for large systems --------
@cached(cache=cache_dict['cache_second_matrix_step'],
        key=lambda rho, omega, omega2, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0: hashkey(
            omega, omega2))
def _second_matrix_step_gpu(rho, omega, omega2, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
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

    G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
    rho_prim = af.matmul(G_prim, rho)
    out = af.matmul(a_prim, rho_prim)

    return out


@cached(cache=cache_dict['cache_third_matrix_step'],
        key=lambda rho, omega, omega2, omega3, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind,
                   gpu_0: hashkey(omega, omega2))
def _third_matrix_step_gpu(rho, omega, omega2, omega3, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
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

    G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
    rho_prim = af.matmul(G_prim, rho)
    out = af.matmul(a_prim, rho_prim)

    return out


def _matrix_step_gpu(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
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

    G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
    rho_prim = af.matmul(G_prim, rho)
    out = af.matmul(a_prim, rho_prim)

    return out


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


def calculate_order_3_inner_loop_gpu(counter, omegas, rho, rho_prim_sum, n_states, a_prim, eigvecs, eigvals,
                                     eigvecs_inv, zero_ind, gpu_0):
    for ind_1, omega_1 in counter:
        for ind_2, omega_2 in enumerate(omegas[ind_1:]):
            # Calculate all permutation for the trace_sum
            var = np.array([omega_1, omega_2, - omega_1 - omega_2])
            perms = list(permutations(var))
            for omega in perms:
                rho_prim = _first_matrix_step_gpu(rho, omega[2] + omega[1],
                                                  a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
                rho_prim = _second_matrix_step_gpu(rho_prim, omega[1], omega[2] + omega[1], a_prim, eigvecs,
                                                   eigvals, eigvecs_inv, zero_ind, gpu_0)

                rho_prim_sum[ind_1, ind_2 + ind_1, :] += af.data.moddims(rho_prim, d0=1, d1=1, d2=n_states)

    spec_data = af.algorithm.sum(rho_prim_sum, dim=2).to_ndarray()

    return spec_data


def calculate_order_4_inner_loop_gpu(counter, omegas, rho, rho_prim_sum, n_states, a_prim, eigvecs, eigvals,
                                     eigvecs_inv, zero_ind, gpu_0, s_k, second_term_mat, third_term_mat):
    for ind_1, omega_1 in counter:

        for ind_2, omega_2 in enumerate(omegas[ind_1:]):
            # for ind_2, omega_2 in enumerate(omegas[:ind_1+1]):

            # Calculate all permutation for the trace_sum
            var = np.array([omega_1, -omega_1, omega_2, -omega_2])
            perms = list(permutations(var))
            trace_sum = 0
            second_term_sum = 0
            third_term_sum = 0

            for omega in perms:
                rho_prim = _first_matrix_step_gpu(rho, omega[1] + omega[2] + omega[3], a_prim, eigvecs, eigvals,
                                                  eigvecs_inv, zero_ind, gpu_0)
                rho_prim = _second_matrix_step_gpu(rho_prim, omega[2] + omega[3],
                                                   omega[1] + omega[2] + omega[3], a_prim, eigvecs,
                                                   eigvals, eigvecs_inv, zero_ind, gpu_0)

                rho_prim = _matrix_step_gpu(rho_prim, omega[3], a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)

                rho_prim_sum[ind_1, ind_2 + ind_1, :] += af.data.moddims(rho_prim, d0=1,
                                                                         d1=1,
                                                                         d2=n_states)
                second_term_mat[ind_1, ind_2 + ind_1] += second_term_gpu(omega[1], omega[2], omega[3], s_k, eigvals)
                third_term_mat[ind_1, ind_2 + ind_1] += third_term_gpu(omega[1], omega[2], omega[3], s_k, eigvals)

    spec_data = af.algorithm.sum(rho_prim_sum, dim=2).to_ndarray()
    spec_data += af.algorithm.sum(af.algorithm.sum(second_term_mat + third_term_mat, dim=3),
                                  dim=2).to_ndarray()

    spec_data[(spec_data == 0).nonzero()] = spec_data.T[(spec_data == 0).nonzero()]

    return spec_data


__all__ = ['_first_matrix_step_gpu', '_second_matrix_step_gpu', '_matrix_step_gpu',
           'second_term_gpu', 'third_term_gpu', 'calculate_order_3_inner_loop_gpu',
           'cache_dict', 'clear_cache_dict', 'calculate_order_4_inner_loop_gpu']
