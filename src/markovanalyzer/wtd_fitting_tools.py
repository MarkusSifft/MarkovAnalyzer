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
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from lmfit import Parameters, Minimizer


class WtdFitter:
    def __init__(self, signal = None, threshold_up = None, threshold_down = None, delta_t = None, monte_carlo = False,
                 parameter = None, set_system = None, t_up_to_down = None, t_down_to_up = None, down_states = None, up_states = None,
                 wtd_up_to_down_mean = None, wtd_up_to_down_std = None, wtd_down_to_up_mean = None, wtd_down_to_up_std= None):

        self.waiting_times_down_to_up = None
        self.waiting_times_up_to_down = None
        self.binary_signal = None
        self.signal = signal
        self.delta_t = delta_t
        self.threshold_up = threshold_up
        self.threshold_down = threshold_down
        self.monte_carlo = monte_carlo
        self.parameter = parameter
        self.set_system = set_system
        self.t_up_to_down = t_up_to_down
        self.t_down_to_up = t_down_to_up
        self.down_states = down_states
        self.up_states = up_states
        self.wtd_up_to_down_mean = wtd_up_to_down_mean
        self.wtd_up_to_down_std = wtd_up_to_down_std
        self.wtd_down_to_up_mean = wtd_down_to_up_mean
        self.wtd_down_to_up_std = wtd_down_to_up_std
        self.counter = 0

    def calculate_waiting_times(self):
        # Initialize variables
        current_level = 'up' if self.signal[0] > self.threshold_up else 'down'
        transition_times = []
        waiting_times_up_to_down = []
        waiting_times_down_to_up = []
        last_transition_time = 0
        binary_signal = np.zeros_like(self.signal)  # Binarized signal array

        # Iterate through the signal to find transitions
        for i, value in enumerate(self.signal):
            if current_level == 'up' and value < self.threshold_down:
                if last_transition_time > 0:
                    waiting_times_up_to_down.append(i - last_transition_time)
                current_level = 'down'
                last_transition_time = i
            elif current_level == 'down' and value > self.threshold_up:
                if last_transition_time > 0:
                    waiting_times_down_to_up.append(i - last_transition_time)
                current_level = 'up'
                last_transition_time = i
            binary_signal[i] = 1 if current_level == 'up' else 0

        waiting_times_up_to_down = np.array(waiting_times_up_to_down) * self.delta_t
        waiting_times_down_to_up = np.array(waiting_times_down_to_up) * self.delta_t

        if self.monte_carlo:
            random_dt = np.random.rand(len(waiting_times_up_to_down)) * self.delta_t - self.delta_t / 2
            waiting_times_up_to_down += random_dt

            random_dt = np.random.rand(len(waiting_times_down_to_up)) * self.delta_t - self.delta_t / 2
            waiting_times_down_to_up += random_dt

        # Return the binary signal and lists of waiting times
        self.binary_signal = binary_signal
        self.waiting_times_up_to_down = waiting_times_up_to_down
        self.waiting_times_down_to_up = waiting_times_down_to_up

        return binary_signal, waiting_times_up_to_down, waiting_times_down_to_up

    def calc_wtd_histogram(self, n_parts=1, n_bins=250):
        n_times = self.waiting_times_down_to_up.shape[0] // n_parts
        all_hist = []
        all_bin_centers = []

        # Determine the global range of data for consistent binning
        data_min = np.min(self.waiting_times_down_to_up)
        data_max = np.max(self.waiting_times_down_to_up)

        # Define bins based on the global range
        bins = np.linspace(data_min, data_max, n_bins)  # 58 bins + 1 for edge

        for i in range(5):
            hist, bin_edges = np.histogram(self.waiting_times_down_to_up[i * n_times:(i + 1) * n_times], bins=bins,
                                           density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            all_hist.append(hist)
            all_bin_centers.append(bin_centers)

        # Since we use the same bins, all bin_centers arrays are the same
        # Now, all_hist contains histograms with consistent binning

        # Optionally, you can handle the bin centers globally if needed outside the loop
        common_bin_centers = bin_centers  # As all are same, use the last computed

        # Convert all_hist from a list of arrays to a 2D NumPy array
        all_hist_array = np.array(all_hist)

        # Create a mask where all histogram values are positive across all segments
        positive_mask = (all_hist_array > 0).all(axis=0)

        # Filter histograms and bin centers based on the mask
        filtered_hist_down_to_up = [hist[positive_mask] for hist in all_hist_array]
        filtered_bin_centers_down_to_up = common_bin_centers[positive_mask]

    def perform_wtd_fit(self, parameter):
        fit_params = Parameters()

        for i, name in enumerate(parameter):
            fit_params.add(name, value=parameter[name][0] / 1e3, min=parameter[name][1] / 1e3, max=parameter[name][2] / 1e3,
                           vary=parameter[name][3])

        mini = Minimizer(self.objective, fit_params)
        out = mini.minimize(method='leastsq', xtol=1e-8, ftol=1e-8, )

        return out

    def objective(self, fit_params):
        markov_system = self.set_system(fit_params)

        # ----- calculation in kHz ------
        model_wtd_up_to_down, _ = markov_system.calculate_WTD(1e3 * self.t_up_to_down, self.down_states, self.up_states)
        _, model_wtd_down_to_up = markov_system.calculate_WTD(1e3 * self.t_down_to_up, self.down_states, self.up_states)

        # ----- conversion to Hz ------
        model_wtd_up_to_down *= 1e3
        model_wtd_down_to_up *= 1e3

        resid = []

        if self.wtd_down_to_up_std is not None and self.wtd_up_to_down_std is not None:
            resid.append(np.abs(model_wtd_up_to_down - self.wtd_up_to_down_mean) / self.wtd_up_to_down_std)
            resid.append(np.abs(model_wtd_down_to_up - self.wtd_down_to_up_mean) / self.wtd_down_to_up_std)

        else:
            resid.append(np.abs(model_wtd_up_to_down - self.wtd_up_to_down_mean))
            resid.append(np.abs(model_wtd_down_to_up - self.wtd_down_to_up_mean))

        self.counter += 1

        if self.counter % len(fit_params)//2 == 0:
            fig, ax = plt.subplots(ncols=2, figsize=(14, 4))
            ax[0].plot(self.t_up_to_down, model_wtd_up_to_down, label='model up -> down')
            ax[0].plot(self.t_up_to_down, self.wtd_up_to_down_mean, label='meas. up -> down')
            ax[0].set_yscale('log')
            ax[1].plot(self.t_down_to_up, model_wtd_down_to_up, label='model down -> up')
            ax[1].plot(self.t_down_to_up, self.wtd_down_to_up_mean, label='meas. down -> up')
            plt.yscale('log')
            plt.show()

        return np.concatenate(resid)
