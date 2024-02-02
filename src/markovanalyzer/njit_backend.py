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

from numba import njit, prange
import numpy as np


@njit('int64(int64)')
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


@njit("void(float64[:], int64, float64[:,:], int64[:])")
def generate_permutations(arr, index, result, counter):
    n = len(arr)
    if index == n - 1:
        result[counter[0], :] = arr
        counter[0] += 1
        return
    for i in range(index, n):
        arr[i], arr[index] = arr[index], arr[i]
        generate_permutations(arr, index + 1, result, counter)
        arr[i], arr[index] = arr[index], arr[i]


@njit("complex128[:,:](float64, complex128[:,:], complex128[:], complex128[:,:], int64, int64)")
def _fourier_g_prim_njit(nu, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
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

    eigvecs = np.ascontiguousarray(eigvecs)
    diagonal = np.ascontiguousarray(np.diag(diagonal))
    eigvecs_inv = np.ascontiguousarray(eigvecs_inv)

    Fourier_G = eigvecs @ diagonal @ eigvecs_inv

    return Fourier_G


@njit(
    "complex128[:](complex128[:], float64, complex128[:,:], complex128[:,:], complex128[:], complex128[:,:], int64, int64)",
    fastmath=False)
def _first_matrix_step_njit(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
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

    G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
    G_prim = np.ascontiguousarray(G_prim)
    rho = np.ascontiguousarray(rho)
    rho_prim = G_prim @ rho
    a_prim = np.ascontiguousarray(a_prim)
    out = a_prim @ rho_prim

    return out


@njit(
    "complex128[:](complex128[:], float64, float64, complex128[:,:], complex128[:,:], complex128[:], complex128[:,:], int64, int64)",
    fastmath=False)
def _second_matrix_step_njit(rho, omega, omega2, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
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

    G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
    G_prim = np.ascontiguousarray(G_prim)
    rho = np.ascontiguousarray(rho)
    rho_prim = G_prim @ rho
    a_prim = np.ascontiguousarray(a_prim)
    out = a_prim @ rho_prim

    return out


@njit(
    "complex128[:](complex128[:], float64, float64, float64, complex128[:,:], complex128[:,:], complex128[:], complex128[:,:], int64, int64)",
    fastmath=False)
def _third_matrix_step_njit(rho, omega, omega2, omega3, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
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

    G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
    G_prim = np.ascontiguousarray(G_prim)
    rho = np.ascontiguousarray(rho)
    rho_prim = G_prim @ rho
    a_prim = np.ascontiguousarray(a_prim)
    out = a_prim @ rho_prim

    return out


@njit(
    "complex128[:](complex128[:], float64, complex128[:,:], complex128[:,:], complex128[:], complex128[:,:], int64, int64)",
    fastmath=False)
def _matrix_step_njit(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
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

    G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
    G_prim = np.ascontiguousarray(G_prim)
    rho = np.ascontiguousarray(rho)
    rho_prim = G_prim @ rho
    a_prim = np.ascontiguousarray(a_prim)
    out = a_prim @ rho_prim

    return out


@njit("complex128(float64, float64, float64, complex128[:], complex128[:])", fastmath=False)
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

    out_sum = 0 * eigvals[0]
    iterator = np.array(list(range(len(s_k))))
    iterator = iterator[np.abs(s_k) > 1e-20 * np.max(np.abs(s_k))]

    for k in iterator:
        for l in iterator:
            out_sum += s_k[k] * s_k[l] * 1 / ((eigvals[l] + 1j * nu1) * (eigvals[k] + 1j * nu3)
                                              * (eigvals[k] + eigvals[l] + 1j * nu2))

    return out_sum


@njit("complex128(float64, float64, float64, complex128[:], complex128[:])", fastmath=False)
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
    iterator = iterator[np.abs(s_k) > 1e-20 * np.max(np.abs(s_k))]

    for k in iterator:
        for l in iterator:
            out += s_k[k] * s_k[l] * 1 / ((eigvals[k] + 1j * nu1) * (eigvals[k] + 1j * nu3)
                                          * (eigvals[k] + eigvals[l] + 1j * nu2))
    return out


@njit(
    'complex128(float64[:,:], complex128[:], complex128[:,:], complex128[:,:], complex128[:], complex128[:,:], int64, int64)',
    parallel=False)
def calculate_order_3_parallel_loop(perms, rho, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
    trace_sum = 0
    for perms_ind in prange(len(perms)):
        omega = perms[perms_ind]
        rho_prim = _first_matrix_step_njit(rho, omega[2] + omega[1], a_prim,
                                           eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
        rho_prim_2 = _second_matrix_step_njit(rho_prim, omega[1], omega[2] + omega[1], a_prim, eigvecs,
                                              eigvals, eigvecs_inv, zero_ind, gpu_0)

        trace_sum += rho_prim_2.sum()
    return trace_sum


@njit(
    "complex128[:,:](float64[:], complex128[:], complex128[:,:], complex128[:,:], complex128[:,:], complex128[:], complex128[:,:], int64, int64)",
    parallel=False)
def calculate_order_3_inner_loop_njit(omegas, rho, spec_data, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0):
    for ind_1 in range(len(omegas)):
        omega_1 = omegas[ind_1]
        for ind_2 in range(len(omegas) - ind_1):
            omega_2 = omegas[ind_1 + ind_2]

            # Calculate all permutation for the trace_sum
            var = np.array([omega_1, omega_2, - omega_1 - omega_2])
            n = len(var)
            num_permutations = factorial(n)
            result_shape = (num_permutations, n)

            # Pre-allocate a NumPy array to hold the results
            perms = np.zeros(result_shape, dtype=var.dtype)

            # Counter for keeping track of how many permutations have been stored
            perms_counter = np.array([0])

            # Generate permutations
            generate_permutations(var, 0, perms, perms_counter)

            trace_sum = calculate_order_3_parallel_loop(perms, rho, a_prim, eigvecs, eigvals,
                                                        eigvecs_inv, zero_ind, gpu_0)

            spec_data[ind_1, ind_2 + ind_1] = trace_sum

    return spec_data


@njit(
    '(float64[:,:], complex128[:], complex128[:,:], complex128[:,:], complex128[:], complex128[:,:], int64, int64, complex128[:])',
    parallel=False)
def calculate_order_4_parallel_loop(perms, rho, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0, s_k):
    trace_sum = 0
    second_term_sum = 0
    third_term_sum = 0

    for perms_ind in range(len(perms)):
        omega = perms[perms_ind]
        rho_prim = _first_matrix_step_njit(rho, omega[1] + omega[2] + omega[3], a_prim,
                                           eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)
        rho_prim = _second_matrix_step_njit(rho_prim, omega[2] + omega[3],
                                            omega[1] + omega[2] + omega[3], a_prim, eigvecs,
                                            eigvals, eigvecs_inv, zero_ind, gpu_0)

        rho_prim = _matrix_step_njit(rho_prim, omega[3], a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0)

        trace_sum += rho_prim.sum()
        second_term_sum += second_term_njit(omega[1], omega[2], omega[3], s_k, eigvals)
        third_term_sum += third_term_njit(omega[1], omega[2], omega[3], s_k, eigvals)

    return trace_sum, second_term_sum, third_term_sum


@njit(
    "complex128[:,:](float64[:], complex128[:], complex128[:,:], complex128[:,:], complex128[:,:], complex128[:], complex128[:,:], int64, int64, complex128[:])",
    fastmath=False)
def calculate_order_4_inner_loop_njit(omegas, rho, spec_data, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0,
                                      s_k):
    for ind_1 in range(len(omegas)):
        omega_1 = omegas[ind_1]
        for ind_2 in range(len(omegas) - ind_1):
            omega_2 = omegas[ind_1 + ind_2]

            # Calculate all permutation for the trace_sum
            var = np.array([omega_1, -omega_1, omega_2, -omega_2])
            n = len(var)
            num_permutations = factorial(n)
            result_shape = (num_permutations, n)

            # Pre-allocate a NumPy array to hold the results
            perms = np.zeros(result_shape, dtype=var.dtype)

            # Counter for keeping track of how many permutations have been stored
            perms_counter = np.array([0])

            # Generate permutations
            generate_permutations(var, 0, perms, perms_counter)

            trace_sum, second_term_sum, third_term_sum = calculate_order_4_parallel_loop(perms, rho, a_prim, eigvecs, eigvals, eigvecs_inv, zero_ind, gpu_0, s_k)

            spec_data[ind_1, ind_2 + ind_1] = second_term_sum + third_term_sum + trace_sum
            spec_data[ind_2 + ind_1, ind_1] = second_term_sum + third_term_sum + trace_sum

    return spec_data


__all__ = ['_first_matrix_step_njit', '_second_matrix_step_njit', '_matrix_step_njit',
           'second_term_njit', 'third_term_njit', 'calculate_order_3_inner_loop_njit',
           'calculate_order_4_inner_loop_njit']
