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

import random
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, objmode, prange
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.sparse.linalg import expm


@njit
def sample_end_state(G_new, start_state, verbose=False):
    probabilities = G_new[:, start_state]  # ---- vll nur einen eintrag aus G rechnen
    cumulative_probs = np.cumsum(probabilities)

    r = random.random()

    index = -1
    for i, cp in enumerate(cumulative_probs):
        if cp > r:
            index = i
            break

    # end_state = gausian_state(means[index], means, sigmas)
    end_state = index

    if verbose:
        print('probabilities:', probabilities)
        print('r:', r)
        print('index:', index)
        print('end_state:', end_state)

    return end_state


@njit
def sample_path(n_steps, state_0, params, delta_t):
    state_array = np.zeros((n_steps), dtype=np.int8)
    state_array[0] = state_0

    n_states = int(params.shape[0] / 3 / 2 + 1)
    n_rates = int(params.shape[0] / 3)

    gammas_old = np.zeros(n_rates, dtype=np.float64)
    G_old = np.zeros((n_states, n_states))

    for i in range(n_steps - 1):
        G_new, gammas_old = calc_G(i, params, delta_t, gammas_old, G_old)
        end_state = sample_end_state(G_new, state_array[i])
        state_array[i + 1] = end_state
        G_old = G_new

    return state_array


@njit
def calc_measurement(state_array, means):
    return means[state_array]


@njit(parallel=False)
def calc_u_array(state_array, params, delta_t):
    n_steps = state_array.shape[0]

    u_array_prob = np.zeros((n_steps, 2))

    u_array_prob[0] = np.array([1, 1])  # Speicher die Wahrscheinlichkeit des remainen und des jumps

    time_stamps_of_jumps = state_array[:, 2] * delta_t

    for i in prange(n_steps - 1):
        duration = state_array[i, 1]  # given in number of points
        # print('duration', duration)

        # ---------- probability for staying in the same state during t=duration --------
        # ---------- Raten werden über die Verweildauer als konstant angenommen --------

        with objmode(G_new='float64[:,:]'):
            G_new = calc_G(delta_t, time_stamps_of_jumps[i], params)  # G(i) oder G(i+1)

        # print('G_new:', G_new)
        # next_u = G_new[state_array[i,0],:]

        u_array_prob[i + 1, 0] = G_new[state_array[i, 0], state_array[i, 0]] ** duration

        # --------- probability of jumping to the next state ---------

        with objmode(G_new='float64[:,:]'):
            G_new = calc_G(delta_t, time_stamps_of_jumps[i + 1], params)  # G(i) oder G(i+1)

        # print('delta_t',delta_t)
        # print('G_new:', G_new)

        next_u = G_new[state_array[i, 0], :]

        u_array_prob[i + 1, 1] = next_u[state_array[i + 1, 0]]

        # return G_new, next_u, u_array_prob

    return u_array_prob


@njit
def calc_path_prob(u_array_prob):
    u_array_prob = u_array_prob.flatten()
    # u_array_prob = u_array_prob[u_array_prob!=0.]
    return - np.log(u_array_prob).sum()


# @njit #-------Anpassen auf digitallisierte Daten
def simulation_path_prob(n_steps, state_0, means, params, delta_t):
    state_array = sample_path(n_steps, state_0, params, delta_t)
    u_array, u_array_prob = calc_u_array(state_array, params, means, delta_t)
    return calc_path_prob(u_array_prob), state_array


@njit
def decay(t, a, b, c):
    return a * (1 / (t + b) + c)


@njit
def calc_V(time_stamps_of_jump, params):
    t = time_stamps_of_jump

    # ----- nur bestimme Raten sind zeitabhängig -----
    gamma_array = params_to_gamma_array(t, params)

    # ----- Berechnung der Übungsgangsmatrix -----

    n_states = int(params.shape[0] / 3 / 2 + 1)

    W = np.zeros((n_states, n_states))
    W += np.diag(gamma_array[:n_states - 1], k=1)
    W += np.diag(gamma_array[n_states - 1:], k=-1)

    kron_delta = np.identity(n_states)

    V = np.empty_like(W)

    for i in range(n_states):
        for j in range(n_states):
            V[i, j] = sum([W[k, j] * kron_delta[i, j] for k in range(n_states)]) - W[i, j]

    return V


@njit
def calc_G(delta_t, time_stamps_of_jump, params):
    tmp = calc_V(time_stamps_of_jump, params)

    # print('V:', tmp)

    with objmode(G_new='float64[:,:]'):
        G_new = expm(-tmp * delta_t)

    return G_new


@njit
def system_to_probability_array(params, state_array):
    delta_t = 1 / 400e3

    # ------ probability of unknown measurement -------

    u_array_prob = calc_u_array(state_array, params, delta_t)
    p_trace = calc_path_prob(u_array_prob)

    return p_trace


def objective(params, state_array):
    # print(params)

    p_test = system_to_probability_array(params, state_array)

    print('---------p_test:', p_test)

    return p_test


@njit
def calc_G(delta_t, time_stamps_of_jump, params):
    tmp = calc_V(time_stamps_of_jump, params)

    # print('V:', tmp)

    with objmode(G_new='float64[:,:]'):
        G_new = expm(-tmp * delta_t)

    return G_new


@njit
def system_to_probability_array(params, state_array):
    delta_t = 1 / 400e3

    # ------ probability of unknown measurement -------

    u_array_prob = calc_u_array(state_array, params, delta_t)
    p_trace = calc_path_prob(u_array_prob)

    return p_trace


@njit
def params_to_gamma_array(t, params):
    params = np.abs(params)

    n_transitions = int(params.shape[0] / 3 / 2)

    gamma_array = np.zeros(n_transitions * 2, dtype=np.float64)

    for i, j in enumerate(range(0, n_transitions * 6, 6)):
        gamma_array[i] = decay(t, params[j], params[j + 1], params[j + 2])
        gamma_array[i + n_transitions] = decay(t, params[j + 3], params[j + 4], params[j + 5])

    return gamma_array


class DiscreteHaugSystem:
    def __init__(self, dataset, initial_state, lower_bound, upper_bound):
        self.dataset = dataset
        self.initial_state = initial_state
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.all_p = []
        # Initialize matplotlib plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([])  # Start with empty plot
        self.ax.set_ylim([0, 1])  # Assuming p_test is in range [0, 1]
        plt.show()

    def fit_system(self, with_bounds=False):
        bounds = Bounds(self.lower_bound, self.upper_bound)

        assert int(self.initial_state.shape[0] / 3 / 2) == int(self.dataset[:, 0].max())

        if with_bounds:
            result = minimize(self.objective_function, self.initial_state, args=self.dataset, method = 'transition_matrix-BFGS-B', bounds = bounds)
        else:
            result = minimize(self.objective_function, self.initial_state, args=self.dataset)

        return result

    def simulate_path(self, n_steps, state_0, means, params, delta_t):
        probability, simulation  = simulation_path_prob(n_steps, state_0, means, params, delta_t)
        return probability, simulation

    def objective_function(self, params, state_array):
        p_test = system_to_probability_array(params, state_array)
        self.all_p.append(p_test)  # Store all p_test values
        # Update the plot
        self.line.set_ydata(self.all_p)  # Update the y-data of the plot
        self.line.set_xdata(range(len(self.all_p)))  # Update the x-data of the plot
        self.ax.relim()  # Recompute the data limits based on the actual data
        self.ax.autoscale_view()  # Rescale the view
        self.fig.canvas.draw()  # Redraw the current figure
        return p_test