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

import numpy as np
from numpy.linalg import inv, eig
from scipy.linalg import eig, expm
from numba import njit

from tqdm import tqdm_notebook
import pickle

import arrayfire as af
from arrayfire.interop import from_ndarray as to_gpu

from signalsnap.spectrum_plotter import SpectrumPlotter
from signalsnap.plot_config import PlotConfig

from .njit_backend import *
from .gpu_backend import *

import matplotlib.pyplot as plt


#  from pympler import asizeof


# ------- Second Term of S(4) ---------

#  @njit(parallel=True, fastmath=False)
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


# @cached(cache=cache_dict['cache_second_term'],
#        key=lambda omega1, omega2, omega3, s_k, eigvals, enable_gpu: hashkey(omega1, omega2, omega3))
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


# @njit(fastmath=False)
# @cached(cache=cache_dict['cache_third_term'],
#        key=lambda omega1, omega2, omega3, s_k, eigvals, enable_gpu: hashkey(omega1, omega2, omega3))
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


@njit
def simulate_markov_chain(cumulative_P, initial_state, num_steps):
    """ Simulate the Markov chain using Numba for acceleration.
    """
    states = np.empty(num_steps + 1, dtype=np.int32)  # Pre-allocate array for speed
    states[0] = initial_state
    current_state = initial_state

    for i in range(1, num_steps + 1):
        random_value = np.random.random()  # Generate a single random number
        # Find the next state using searchsorted
        current_state = np.searchsorted(cumulative_P[current_state], random_value)
        states[i] = current_state

    return states


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

        # ------- for unreavaling simulation -------
        self.simulated_observed_values = None
        self.simulated_states = None
        self.simulated_jump_times = None

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
        extended_part = np.ones_like(old_m_op) * old_m_op.max()

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

    def first_matrix_step(self, rho, omega):
        """
        Helper method to move function out of the class. njit is not working within classes
        """
        if self.enable_gpu:
            return _first_matrix_step_gpu(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                                          self.zero_ind, self.gpu_0)
        else:
            return _first_matrix_step_njit(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                                           self.zero_ind, self.gpu_0)

    def second_matrix_step(self, rho, omega, omega2):
        """
        Helper method to move function out of the class. njit is not working within classes
        """

        if self.enable_gpu:
            return _second_matrix_step_gpu(rho, omega, omega2, self.A_prim, self.eigvecs, self.eigvals,
                                           self.eigvecs_inv,
                                           self.zero_ind, self.gpu_0)
        else:
            return _second_matrix_step_njit(rho, omega, omega2, self.A_prim, self.eigvecs, self.eigvals,
                                            self.eigvecs_inv,
                                            self.zero_ind, self.gpu_0)

    def matrix_step(self, rho, omega):
        """
        Helper method to move function out of the class. njit is not working within classes
        """

        if self.enable_gpu:
            return _matrix_step_gpu(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                                    self.zero_ind, self.gpu_0)
        else:
            return _matrix_step_njit(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                                     self.zero_ind, self.gpu_0)

    def calculate_order_3_inner_loop(self, counter, omegas, rho, spec_data, rho_prim_sum, n_states):
        if self.enable_gpu:
            return calculate_order_3_inner_loop_gpu(counter, omegas, rho, rho_prim_sum, n_states, self.A_prim,
                                                    self.eigvecs, self.eigvals, self.eigvecs_inv, self.zero_ind,
                                                    self.gpu_0)

        else:
            return calculate_order_3_inner_loop_njit(omegas, rho, spec_data, self.A_prim,
                                                     self.eigvecs, self.eigvals, self.eigvecs_inv, self.zero_ind,
                                                     self.gpu_0)

    def calculate_order_4_inner_loop(self, counter, omegas, rho, spec_data, rho_prim_sum, n_states, second_term_mat,
                                     third_term_mat):
        if self.enable_gpu:
            return calculate_order_4_inner_loop_gpu(counter, omegas, rho, rho_prim_sum, n_states, self.A_prim,
                                                    self.eigvecs, self.eigvals, self.eigvecs_inv, self.zero_ind,
                                                    self.gpu_0, self.s_k, second_term_mat, third_term_mat)

        else:
            return calculate_order_4_inner_loop_njit(omegas, rho, spec_data, self.A_prim,
                                                     self.eigvecs, self.eigvals, self.eigvecs_inv, self.zero_ind,
                                                     self.gpu_0, self.s_k)

    def plot(self, plot_orders=(2, 3, 4), imag_plot=False, s2_f=None, s2_data=None, s3_f=None, s3_data=None, s4_f=None,
             s4_data=None):

        if s2_f is None:
            s2_f = self.freq[2]
        if s3_f is None:
            s3_f = self.freq[3]
        if s4_f is None:
            s4_f = self.freq[4]

        if s2_data is None:
            s2_data = self.S[2]
        if s3_data is None:
            s3_data = self.S[3]
        if s4_data is None:
            s4_data = self.S[4]

        config = PlotConfig(plot_orders=plot_orders, imag_plot=imag_plot, s2_f=s2_f, s2_data=s2_data, s3_f=s3_f,
                            s3_data=s3_data, s4_f=s4_f, s4_data=s4_data)

        self.f_lists = {1: None, 2: None, 3: None, 4: None}
        self.S_err = {1: None, 2: None, 3: None, 4: None}
        self.config = config
        self.config.f_unit = 'Hz'
        plot_obj = SpectrumPlotter(self, config)

        if self.S[1] is not None:
            print('s1:', self.S[1])
        fig = plot_obj.plot()
        return fig

    def calculate_spectrum(self, f_data, order_in, bar=True, verbose=False, beta_offset=True,
                           enable_gpu=False, cache_trispec=True):

        if order_in == 'all':
            orders = [1, 2, 3, 4]
        else:
            orders = order_in

        for order in orders:
            self.calculate_one_spectrum(f_data, order, bar=bar, verbose=verbose,
                                        beta_offset=beta_offset, enable_gpu=enable_gpu, cache_trispec=cache_trispec)

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
            if emits > 0:
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

    def calculate_order_one(self, measurement_op, enable_gpu, bar):
        if bar:
            print('Calculating first order')
        if enable_gpu:
            rho = af.matmul(measurement_op, self.rho_steady)
            self.S[1] = af.algorithm.sum(rho)
        else:
            rho = measurement_op @ self.rho_steady
            self.S[1] = np.array([rho.sum()])

    def calculate_order_two(self, omegas, rho, rho_prim_sum, spec_data, enable_gpu, beta_offset, bar):
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

        order = 2
        self.S[order] = spec_data
        if beta_offset:
            self.S[order] += 1 / 4

    def calculate_order_three(self, omegas, n_states, rho, rho_prim_sum, spec_data, enable_gpu, verbose, bar):

        if bar:
            print('Calculating bispectrum')
            counter = tqdm_notebook(enumerate(omegas), total=len(omegas))
        else:
            counter = omegas

        spec_data = self.calculate_order_3_inner_loop(counter, omegas, rho, spec_data, rho_prim_sum, n_states)

        spec_data[(spec_data == 0).nonzero()] = spec_data.T[(spec_data == 0).nonzero()]

        if np.max(np.abs(np.imag(np.real_if_close(_full_bispec(spec_data))))) > 0 and verbose:
            print('Bispectrum might have an imaginary part')

        order = 3
        self.S[order] = _full_bispec(spec_data)

    def calculate_order_four(self, omegas, n_states, rho, rho_prim_sum, spec_data, second_term_mat, third_term_mat,
                             enable_gpu, verbose, bar, cache_trispec):
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

        spec_data = self.calculate_order_4_inner_loop(counter, omegas, rho, spec_data, rho_prim_sum, n_states,
                                                      second_term_mat, third_term_mat)

        if np.max(np.abs(np.imag(np.real_if_close(_full_trispec(spec_data))))) > 0 and verbose:
            print('Trispectrum might have an imaginary part')

        order = 4
        self.S[order] = _full_trispec(spec_data)

    def calculate_rho_steady(self):
        self.eigvals, self.eigvecs = eig(self.transtion_matrix.astype(dtype=np.complex128))
        self.eigvecs_inv = inv(self.eigvecs)

        self.eigvals = self.eigvals.astype(dtype=np.complex128)
        self.eigvecs = self.eigvecs.astype(dtype=np.complex128)
        self.eigvecs_inv = self.eigvecs_inv.astype(dtype=np.complex128)

        self.zero_ind = np.argmax(np.real(self.eigvals))

        rho_steady = self.eigvecs[:, self.zero_ind]
        rho_steady = rho_steady / np.sum(rho_steady)

        self.rho_steady = rho_steady
        return rho_steady

    def calculate_WTD(self, t, down_states, up_states):

        temp1 = self.transtion_matrix - np.diag(np.diag(self.transtion_matrix))
        jump_op_in = np.zeros_like(temp1)
        jump_op_in[up_states, down_states] = temp1[up_states, down_states]
        jump_op_out = np.zeros_like(temp1)
        jump_op_out[down_states, up_states] = temp1[down_states, up_states]

        self.jump_op_in = jump_op_in
        self.jump_op_out = jump_op_out

        wtd_in = np.zeros_like(t)
        wtd_out = np.zeros_like(t)

        self.eigvals, self.eigvecs = eig(self.transtion_matrix.astype(dtype=np.complex128))
        self.eigvecs_inv = inv(self.eigvecs)

        self.eigvals = self.eigvals.astype(dtype=np.complex128)
        self.eigvecs = self.eigvecs.astype(dtype=np.complex128)
        self.eigvecs_inv = self.eigvecs_inv.astype(dtype=np.complex128)

        self.zero_ind = np.argmax(np.real(self.eigvals))

        rho_steady = self.eigvecs[:, self.zero_ind]
        rho_steady = rho_steady / np.sum(rho_steady)

        self.rho_steady = rho_steady

        for i in tqdm_notebook(range(t.shape[0])):
            self.G = expm((self.transtion_matrix - jump_op_in - jump_op_out) * t[i])
            # self.G = self.eigvecs @ diagonal @ self.eigvecs_inv

            wtd_in[i] = np.sum(jump_op_out @ self.G @ jump_op_in @ rho_steady) / np.sum(jump_op_in @ rho_steady)
            wtd_out[i] = np.sum(jump_op_in @ self.G @ jump_op_out @ rho_steady) / np.sum(jump_op_out @ rho_steady)

        return wtd_in, wtd_out

    def calculate_one_spectrum(self, f_data, order, bar=True, verbose=False, beta_offset=True,
                               enable_gpu=False, cache_trispec=True):
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
        if self.enable_gpu:
            af.device_gc()
            clear_cache_dict()

        if f_data[0] < 0:
            print('Only positive frequencies allowed')
            return None

        omegas = 2 * np.pi * f_data  # [kHz]
        self.freq[order] = f_data

        n_states = self.transtion_matrix.shape[0]
        self.eigvals, self.eigvecs = eig(self.transtion_matrix.astype(dtype=np.complex128))
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

        # self.rho_steady = np.ascontiguousarray(self.rho_steady)
        # self.eigvals = np.ascontiguousarray(self.eigvals)
        # self.eigvecs = np.ascontiguousarray(self.eigvecs)
        # self.eigvecs_inv = np.ascontiguousarray(self.eigvecs_inv)
        # self.A_prim = np.ascontiguousarray(self.A_prim)

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
            rho_prim_sum = None
            second_term_mat = None
            third_term_mat = None

        # estimate necessary cachesize (TODO: Anteile könnten noch anders gewählt werden)
        # if self.enable_gpu:
        #     update_cache_size('cache_fourier_g_prim', self.A_prim, enable_gpu)
        #     update_cache_size('cache_first_matrix_step', rho, enable_gpu)
        #     update_cache_size('cache_second_matrix_step', rho, enable_gpu)
        #     update_cache_size('cache_third_matrix_step', rho, enable_gpu)
        #     update_cache_size('cache_second_term', rho[0], enable_gpu)
        #     update_cache_size('cache_third_term', rho[0], enable_gpu)

        if order == 1:
            self.calculate_order_one(measurement_op, enable_gpu, bar)

        if order == 2:
            self.calculate_order_two(omegas, rho, rho_prim_sum, spec_data, enable_gpu, beta_offset, bar)

        if order == 3:
            self.calculate_order_three(omegas, n_states, rho, rho_prim_sum, spec_data, enable_gpu, verbose, bar)

        if order == 4:
            self.calculate_order_four(omegas, n_states, rho, rho_prim_sum, spec_data, second_term_mat, third_term_mat,
                                      enable_gpu, verbose, bar, cache_trispec)

        if self.enable_gpu:
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

    def simulate_trace(self, initial_dist, total_time):
        """
        Simulates a continuous-time Markov chain.

        Parameters:
        - initial_state: Initial distribution of states (numpy array).
        - total_time: Total time to simulate.

        Returns:
        - simulated_jump_times: Times at which transitions occur.
        - simulated_states: States at these times.
        - simulated_observed_values: Observed values at these times.
        """

        # Normalize transtion_matrix to get transition probabilities and compute holding times

        holding_rates = -np.diag(self.transtion_matrix.T)
        transition_probs = self.transtion_matrix.T / holding_rates[:, np.newaxis]
        np.fill_diagonal(transition_probs, 0)

        current_time = 0.0
        current_state = np.random.choice(len(initial_dist), p=initial_dist)
        self.simulated_jump_times = [current_time]
        self.simulated_states = [current_state]
        self.simulated_observed_values = [self.measurement_op[current_state]]

        while current_time < total_time:
            rate = holding_rates[current_state]
            time_to_next = np.random.exponential(1 / rate)
            current_time += time_to_next

            if current_time > total_time:
                break

            # Transition to the next state
            next_state = np.random.choice(len(transition_probs[current_state]), p=transition_probs[current_state])
            current_state = next_state

            self.simulated_jump_times.append(current_time)
            self.simulated_states.append(current_state)
            self.simulated_observed_values.append(self.measurement_op[current_state])

    def plot_simulation(self):
        """
        Plots the observations of a continuous-time Markov chain over time.

        Parameters:
        - times: List of times at which transitions occur.
        - observations: List of observed values corresponding to the states at these times.
        """
        plt.figure(figsize=(10, 5))
        plt.step(self.simulated_jump_times, self.simulated_observed_values, where='post', label='Observation',
                 linewidth=2)
        plt.title('Observations over Time')
        plt.xlabel('Time')
        plt.ylabel('Observation')
        plt.ylim(min(self.simulated_observed_values) - 0.1 * abs(min(self.simulated_observed_values)),
                 max(self.simulated_observed_values) + 0.1 * abs(max(self.simulated_observed_values)))
        plt.legend()
        plt.show()

    def simulate_discrete_trace(self, total_time, sampling_rate, initial_state=0):
        num_steps = int(total_time * sampling_rate)

        # Create the discrete-time transition matrix
        P = expm(self.transtion_matrix.T * 1 / sampling_rate)
        cumulative_P = np.cumsum(P, axis=1)

        # Simulate the Markov chain
        states = simulate_markov_chain(cumulative_P, initial_state, num_steps)

        measurement = np.zeros(len(states))
        for i, state in enumerate(states):
            measurement[i] = self.measurement_op[state]

        return states, measurement
