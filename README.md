# Correctness is its own reward: bootstrapping error signals in self-guided reinforcement learning

## Dependencies
The scripts were ran in `Python 3.13.3` with `NumPy 2.2.6`, `SciPy 1.16.0`, `PyTorch 2.7.0`, `Scikit-Learn 1.6.1`, `Matplotlib 3.10.1`, `seaborn 0.13.2`, `Numba 0.61.2` and `tqdm 4.67.1`. 

Song preprocessing (optional; see below) requires `Autoencoded Vocal Analysis 0.3.1` ([AVA](https://autoencoded-vocal-analysis.readthedocs.io/en/latest/index.html)) and `Joblib 1.5.0`.

## Instructions
This repository contains the code for simulating the models and generating the core figures in the paper **Correctness is its own reward: bootstrapping error signals in self-guided reinforcement learning** [(bioRxiv)](https://www.biorxiv.org/content/10.1101/2025.07.18.665446).

The mathematical details are described in the paper. `src` contains the source code for the networks, the reinforcement learning environment, the sparse coding model, and other helper functions. 

For the basic training and simulations of the models, check out examples in `simulations`, which also contains the code to generate the panels such as neuronal firing rates, and joint distributions of mean rates between correct and perturbed singing cases in the paper. Notice that the networks themselves are implemented to take any time series input that matches the dimensions, but the simulations to reproduce the figures in the paper uses sparse embeddings of real spectrograms of adult songs. To generate this embedding, follow the steps in `realistic_auditory_processing` or directly [download](https://doi.org/10.7924/r4gf12r3d) the generated embeddings. 

`further_analysis` contains the code to run and visualize the analysis of the learned recurrent weights, as well as the reinforcement learning experiments. `robustness_tests` contains the code to plot the cancellation quality over different parameters.

## A spiking implementation

[`pearsonlab/spiking-ven`](https://github.com/pearsonlab/spiking-ven) is a spiking implementation of
the same cancellation computation, developed alongside the full song-circuit model. Where
the networks in `src/models.py` are rate-based (Wilson–Cowan `WCNet` / `EINet` with the
`bilin_hebb_*` plasticity rules), that repository uses leaky integrate-and-fire
excitatory/inhibitory populations with spike-timing-dependent plasticity on the
inhibitory-to-excitatory weights, driven by a sparse-coded spike train from a gammatone
cochleagram front end.

It reproduces the same qualitative result — the excitatory response to the bird's own song
is cancelled while responses to novel or perturbed sound survive — and is quantified
against Mandelblat-Cerf et al. (2014) using their own criterion: 50 ms noise bursts during
singing, with spike counts in 150 ms windows compared per neuron across renditions. The
modelled response has an onset latency of 32.7 ms (bird: 23 ± 12 ms), peaks 48 ms after
onset (bird: ~40–60 ms), and sits on a 2.2 Hz singing baseline (bird: 1–10 Hz), with 56%
of units responding (bird: 41%, 7/17). Response magnitude remains the open problem, ~2×
too strong: 1.79 extra spikes per responder against the bird's ~0.8–1.0. The pipeline is
fully seeded and reproducible end to end from the public song corpus in about twenty
minutes.

It is a separate, independent implementation rather than a port of this code, and its
dependency footprint is deliberately minimal (numpy and scipy for the core), so it can be
reused without this repository or the circuit model.

