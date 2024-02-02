# MarkovAnalyzer: Simplifying Markov Analysis in Python
Authored by M. Sifft and D. Hägele We are excited to introduce MarkovAnalyzer, a high-speed Python toolkit designed for 
the analysis of hidden Markov models. This toolkit offers two distinct approaches: the well-established forward algorithm 
and our innovative polyspectra fitting method, both of which facilitate the study of complex Markov systems. The forward 
algorithm is a widely recognized technique, detailed on [Wikipedia](https://en.wikipedia.org/wiki/Forward_algorithm). 
In contrast, the polyspectra method is a novel 
contribution from our team at Ruhr University Bochum. This approach involves a comparative analysis between the 
experimental polyspectra—advanced extensions of the traditional power spectrum—and their theoretical models, which 
are derived from a given transition matrix and a measurement operator. This operator translates the state of the 
Markov system into measurable values. By matching the theoretical polyspectra with the experimental data, we can 
accurately estimate variable parameters within the hidden Markov model. The use of polyspectra in analyzing Markov 
processes comes with several significant benefits:

* Direct Input Utilization: Our method accepts the raw, experimentally measured Markov process data. 
This eliminates the need to categorize noisy measurements into distinct output levels, allowing for the analysis 
of data where noise is prevalent and distinct levels are obscured.
    
* Speed and Efficiency: Polyspectra-based analysis can outpace the forward algorithm, especially with large datasets. 
When dealing with terabyte-sized data, polyspectra need to be computed only once. These computations result in compact 
kilobyte-sized data, which are then used exclusively for the remainder of the analysis process.

MarkovAnalyzer is poised to revolutionize the way researchers approach the analysis of Markov systems, offering both 
efficiency and precision in one user-friendly package.

## Installation
MarkovAnalyzer is available on `pip` and can be installed with 
```bash
pip install markovanalyzer
```

### Installation of Arrayfire 
Besides running on CPU, MarkovAnalyzer offers GPU support for Nvidia and AMD cards. Depending on the hardware used, the
usage of a GPU is highly recommended for Markov systems with more 
than about 100 states. A comprehensive installation guide for Linux + NVidia GPU can be found [here](https://github.com/MarkusSifft/MarkovAnalyzer/wiki/Installation-Guide). 
For GPU calculations the high performance library Arrayfire is used. The Python wrapper ([see here](https://github.com/arrayfire/arrayfire-python)) 
is automatically installed when installing SignalSnap, however, [ArrayFire C/C++ libraries](https://arrayfire.com/download) need to be installed separately. 
Instructions can be found can be found [here](https://github.com/arrayfire/arrayfire-python) and [here](https://arrayfire.org/docs/installing.htm#gsc.tab=0).

## Documentation
The documentation of the package can be found [here](https://markussifft.github.io/MarkovAnalyzer/). 
The package is divided into two parts: the **polyspectra-calculator** module, the **fitting-tools** module, 
and the **forward** module.
### Polyspectra-Calculator Module
The Simulation Module allows for the calculation of the theoretical quantum 
polyspectra directly from the system's transition matrix.
### Fitting-Tools Module
The Fitting-Tools Module enables a user-friendly characterization of a Markov system in the lab based on the 
polyspectra of a measurement of a Markov process. These polyspectra can be calculated
via our [SignalSnap](https://github.com/MarkusSifft/SignalSnap) package. After providing a model transition matrix with
one or more variable parameters, these parameters are estimated by fitting the theoretical model prediction of the 
polyspectra to their measured counterparts.
### Forward Module
Here, the Forward Algorithm for the estimation of Markov models is implemented.

## Example: Characterization of a Two-State Markov Model via Polyspectra
We want to deduce a Markov model (i.e., it's transition matrix) from the observation of the Markov process. We are given 
data, that looks like this:

![two level example trace](Examples/example_data/two_level_example_trace.png)

This is only a short excerpt of the full 6 min dataset. Using the [SignalSnap library](https://github.com/MarkusSifft/SignalSnap)
we are firstly calculating the polyspectra of that measurement. More details about the SignalSnap Code can be found on
its GitHub page.

```python
from signalsnap import SpectrumCalculator, SpectrumConfig, PlotConfig
import numpy as np
import h5py
```
Here is polyspectra are calculated and stored.
```python
path = 'example_data/long_measurement.h5'
group_key = 'day1'
dataset = 'measurement1'

config = SpectrumConfig(dataset=dataset, group_key=group_key, path=path, f_unit='Hz', 
                        spectrum_size=150, f_max=2000, order_in=[1,2,3,4], 
                        backend='cpu')

spec = SpectrumCalculator(config)

f, s, serr = spec.calc_spec()
```

```python
plot_config = PlotConfig(plot_orders=[2,3,4], arcsinh_plot=False, arcsinh_const=0.0002)
fig = spec.plot(plot_config)
```
![two level spectra](Examples/example_data/two_level_spectra.png)

```python
path = 'example_data/two_state_example_spectra.pkl'
spec.save_spec(path, remove_S_stationarity=True)
```
Characterization is performed by fitting the theoretical polyspectra of a Markov model with variable 
parameters to their experimental counterparts calculated above. We are assuming a two-state model 
for the system that produced the data; hence, we begin with defining such a model with two variable 
transition rates.    
The system undergoes transitions from the 0 to 1 state and from the 1 to the 0 state at 
rates gamma_01 and gamma_10, respectively. Each needs to be associated with a measurement value. 
From the histogram above we know that the value of the 0 state might be around 0, whereas the 
value of the 1 state might be around 26. Since we don't know for sure, we leave these measurement 
values also as variable parameters n_0 and n_1.

```python
import markovanalyzer as ma

def set_system(params):
      
    rates = {'0->1': params['gamma_01'],
             '1->0': params['gamma_10']}
             
    m_op = np.array([params['n_0'],params['n_1']])
    
    markov_system = ma.System(rates, m_op)

    return markov_system
```
Now, we can set start values and bounds for the fit of the parameters. A parameter c is always 
part of the fit and acconts for constant white noise outset in the power spectrum.

```python
system_fit = ma.FitSystem(set_system)

parameter = {'gamma_01': [2.0396975e+04, 0, 1e5, True],
             'gamma_10': [1.0345057e+04, 0, 1e5, True],
             'n_0': [0, 0, 1e8, True],
             'n_1': [20, 0, 1e8, True],
             'c': [-2.7651282e-01 , -1e14, 1e14, True]}

path = 'example_data/two_state_example_spectra.pkl'

result = system_fit.complete_fit(path, parameter, 
                        method='least_squares', xtol=1e-6, ftol=1e-6, show_plot=True, fit_modus='order_based',
                        fit_orders=(1,2,3,4), beta_offset=False)
```
![two level fit](Examples/example_data/two_level_final_fit.png)
