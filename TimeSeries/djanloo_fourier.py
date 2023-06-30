from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyts.approximation import DiscreteFourierTransform
from scipy.interpolate import PchipInterpolator

import sys
import warnings

class STFTransformer:
    
    def __init__(self, n_time_bins=None, n_spectral_bins=None, 
                         max_frequency=3000, sampling_rate=48_000/8, 
                         enhancing_method=None):
        self.name = "STFTransformer"
        self.n_time_bins = n_time_bins
        self.n_spectral_bins = n_spectral_bins
        self.max_frequency = max_frequency
        self.sampling_rate = sampling_rate
        
        self.allow_window_padding = False
        
        self.allow_spectral_interpolation = False
        self.freqs=None
        if enhancing_method == "pad":
            self.allow_window_padding = True
            
        elif enhancing_method == "interpolate_spectrum":
            self.allow_spectral_interpolation = True
    
    def report(self, traces):
        assert self.n_time_bins is not None, "number of time bins not defined"
        assert self.n_spectral_bins is not None, "number of spectral bins not defined"
        print(f"Time bins = {self.n_time_bins}, Spectral bins = {self.n_spectral_bins}")
        lengths = [len(tr) for tr in traces]
        
        print(f"min length of window is roughly {np.min(lengths)//self.n_time_bins}")
        print(f"max number of Fourier coeffs is {np.min(lengths)//2//self.n_time_bins}")
        
    
    @classmethod
    def _bin_spectral_energy(cls, spectrum_energy, n_bins, max_freq, sampling_rate=48_000/8, plot_check=False):
        """bins spectral energy and get rid of variable number of spectral points due to different length of samples"""
        if len(spectrum_energy) < n_bins:
            raise ValueError(f"More bins than frequencies (bins are {n_bins}, spectrum has length {len(spectrum_energy)})")

        T = 1.0/sampling_rate

        # Removes higher-pitch part that makes bins unequal
        ss = spectrum_energy[:len(spectrum_energy) - (len(spectrum_energy)%n_bins) ]
        ffs = fftfreq(2*len(ss), T)[:len(ss)]

        ff_bin_edges = np.linspace(0, max_freq, n_bins + 1, endpoint=True)

        binned_energies = np.zeros(n_bins)
        for i in range(n_bins):
            binned_energies[i] = np.mean(ss[(ffs >= ff_bin_edges[i])&(ffs < ff_bin_edges[i+1])] )

        if plot_check:
            plt.plot(ffs, ss, ls="", marker=".", ms=3)
            plt.step(ff_bin_edges[:-1],binned_energies, where="post",alpha=0.7)
            plt.yscale("log")
        return binned_energies
    
    def balance_n_coeff(self, traces):
        """Decides the number of time and spectral bins to have square stft"""
        min_trace_length = np.min([np.sum(~np.isnan(_)) for _ in traces])
        print(f"shortest trace has length {min_trace_length}")
        N = int(np.floor(np.sqrt(min_trace_length/2)))
        
        self.n_time_bins = N
        self.n_spectral_bins = N
        
    def _pad_windows(self, traces):
        """If too much spectral bins are requested, augments the traces adding zeros
        to make each track at least L = 2*n_spectral_bins long.
        """
        L = 2*self.n_spectral_bins
        padded_traces = []
        for tr in tqdm(traces, desc="Padding traces..."):
            padded_trace = []
            signal = tr[~np.isnan(tr)]
            windows = np.array_split(signal, self.n_time_bins)
            for window in windows:
                missing = L - len(window)
                if missing > 0:
                    padded_window = np.pad(window, (0, missing), 'constant', constant_values=(0.0, 0.0))
                else:
                    padded_window = window
                padded_trace.append(padded_window)
            padded_traces.append(np.hstack(padded_trace))
        return padded_traces
    
    def _prepare_spectral_interpolation(self):
        # Initializes the maximally dense frequencies allowed
        self.spectrum_length = max(self.max_window_length//2, self.n_spectral_bins)
        print(f"spectrum length set to { self.max_window_length//2}")
        self.freqs = np.linspace(0, self.max_frequency, self.spectrum_length)
        
    def _interpolate_spectral_energy(self, frequencies, spectral_energy):      
        return  PchipInterpolator(frequencies, spectral_energy)(self.freqs)
        
    def fit_transform(self, traces):
        if self.n_time_bins is None or self.n_spectral_bins is None:
            warnings.warn("time bins/spectral bins not specified, setting balanced bins")
            self.balance_n_coeff(traces)
            warnings.warn(f"time bins: {self.n_time_bins},  spectral_bins={self.n_spectral_bins}")
            
        self.min_window_length = np.min([np.sum(~np.isnan(tr))//self.n_time_bins for tr in traces])
        self.max_window_length = np.max([np.sum(~np.isnan(tr))//self.n_time_bins for tr in traces])

        print("estimated max fourier  bins:", self.min_window_length//2)
        if self.min_window_length//2 < self.n_spectral_bins:
            warnings.warn(f"Number of fourier is max {self.min_window_length//2}.")
            if  self.allow_window_padding:
                warnings.warn("Activating window padding.")
                traces = self._pad_windows(traces)
            elif self.allow_spectral_interpolation:
                self._prepare_spectral_interpolation()
            else:
                raise ValueError("enhancing method must be specified if the requested number of bins is more than maximum value.")
        
        self.spectral_centroid = []
        self.spectral_mode = []
        self.STFT = []
        
        for tr_indx, tr in tqdm(enumerate(traces), total=len(traces), desc="Transforming..."):

            # Removes Nans
            signal = tr[~np.isnan(tr)]

            # Gets values of stuff
            N, T = len(signal), 1.0/self.sampling_rate
            
            # Divides in windows
            windows = np.array_split(signal, self.n_time_bins)
            winlen = np.min([len(w) for w in windows])

            # Prepares containers for STFT and gets the right frequencies
            STFT = np.zeros((self.n_time_bins, self.n_spectral_bins))
            FF = fftfreq(winlen, T)[:winlen//2]
            
            # Prepares STFT stats
            STFT_CENTROID = np.zeros(self.n_time_bins)
            STFT_MODE = np.zeros(self.n_time_bins)
            
            for i, part in enumerate(windows):
                regularized_window = np.hanning(len(part))*part
                window_spectrum = np.abs(fft(regularized_window)[:winlen//2])**2
                
                if self.allow_spectral_interpolation:
                    window_spectrum = self._interpolate_spectral_energy(FF, window_spectrum)

                # Saves the binned STFT
                try:
                    STFT[i, :] = np.log(STFTransformer._bin_spectral_energy(window_spectrum, 
                                                                             self.n_spectral_bins, 
                                                                             3000,
                                                                             sampling_rate=self.sampling_rate)
                                       )
                except ValueError as e:
                    raise ValueError(f"Error in track {tr_indx}, that has window length equal to {winlen}") from e
                    

                # Gets distribution
                energy_density = window_spectrum/np.sum(window_spectrum)
                if np.isnan(energy_density).any():
                    raise ValueError("energy density is not normalisable")

                # Gets weighted mean
                if self.allow_spectral_interpolation:
                    STFT_CENTROID[i] = np.sum(energy_density*self.freqs)
                    STFT_MODE[i] = self.freqs[np.argmax(energy_density)]

                else:
                    STFT_CENTROID[i] = np.sum(energy_density*FF)
                    STFT_MODE[i] = FF[np.argmax(energy_density)]
                

            self.spectral_centroid.append(STFT_CENTROID)
            self.spectral_mode.append(STFT_MODE)
            self.STFT.append(STFT)

            
        # Makes them array
        self.spectral_centroid = np.array(self.spectral_centroid)
        self.spectral_mode = np.array(self.spectral_centroid)
        self.STFT = np.array(self.STFT)
        
        return self.STFT, self.spectral_centroid

# class FunkySTFT(STFTransformer):
    
#     def __init__(self, **kwargs):
#         return super().__init__(**kwargs)
    
#     @classmethod
#     def _bin_spectral_energy(cls, *args, **kwargs):
#         raise NotImplementedError("Funky version does not have to bin anything.")
        
#     def balance_n_coeff(self, traces):
#         raise NotImplementedError("boh")
        
#     def fit_transform(self, traces):
        
#         if self.n_time_bins is None or self.n_spectral_bins is None:
#             warnings.warn("time bins/spectral bins not specified, setting balanced bins")
#             self.balance_n_coeff(traces)
            
#         # This is the list of all windows of all traces
#         global_windows = []
        
#         for tr_indx, tr in tqdm(enumerate(traces), total=len(traces)):

#             # Achtung! Here NaNs are necessary 
#             #signal = tr[~np.isnan(tr)]
#             signal = tr
#             signal[np.isnan(signal)] = 0.0

#             # Divides in windows and adds them to the gigalist
#             global_windows += np.array_split(signal, self.n_time_bins)

#         transf = DiscreteFourierTransformer()

from scipy.signal import stft

class FixedResolutionSTFTransformer:
    
    def __init__(self, 
                 n_spectral_points=100,
                 pad_spectra=False,
                 verbose=True
                ):
        
        self.n_spectral_points = n_spectral_points
        self.pad_spectra = pad_spectra
        self.verbose = verbose
        
    def _pad_spectra(self, list_of_spectra):        
        longer_length = np.max([len(_) for _ in list_of_spectra])
        padded_spectra = np.zeros((len(list_of_spectra), 
                                   longer_length, 
                                   self.n_spectral_points))
        
        for i in range(len(list_of_spectra)):
            M = len(list_of_spectra[i])
            padded_spectra[i,:M, :] = list_of_spectra[i]
            
        return padded_spectra
        
    def transform(self, traces):
        
        transformed_traces = []
        
        iterable = tqdm(traces) if self.verbose else traces
        
        for tr in iterable:
            
            signal = tr[~np.isnan(tr)]
            
            f, t, STFT = stft(signal, nperseg=2*self.n_spectral_points - 1)
            STFT[np.abs(STFT) < 1e-16] = 1e-16
            log_stft = np.log(np.abs(STFT.T))
            transformed_traces.append(log_stft)
            
            
        if self.pad_spectra:
            return self._pad_spectra(transformed_traces)
        else:
            return transformed_traces
        
        
        
        
        
        
        
        