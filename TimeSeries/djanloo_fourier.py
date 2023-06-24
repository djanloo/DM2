from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings

class STFTransformer:
    
    def __init__(self, n_time_bins=None, n_spectral_bins=None, max_frequency=3000, sampling_rate=48_000/8):
        self.name = "STFTransformer"
        self.n_time_bins = n_time_bins
        self.n_spectral_bins = n_spectral_bins
        self.max_frequency = max_frequency
        self.sampling_rate = sampling_rate
    
    
    def report(self, traces):
        assert self.n_time_bins is not None, "number of time bins not defined"
        assert self.n_spectral_bins is not None, "number of spectral bins not defined"
        print(f"Time bins = {self.n_time_bins}, Spectral bins = {self.n_spectral_bins}")
        lengths = [len(tr) for tr in traces]
        
        print(f"min length of trace is roughly {np.min(lengths)//self.n_time_bins}")
        print(f"max number of Fourier coeffs is {np.min(lengths)//2//self.n_time_bins}")
        
    
    @classmethod
    def _bin_spectral_energy(cls, spectrum_energy, n_bins, max_freq, sampling_rate=48_000/8, plot_check=False):
        """bins spectral energy and get rid of variable number of spectral points due to different length of samples"""
        if len(spectrum_energy) < n_bins:
            raise ValueError(f"too few frequencies to bin (spectrum has {len(spectrum_energy)})")

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
        min_trace_length = np.min([len(_) for _ in traces])
        N = int(np.floor(np.sqrt(min_trace_length/2)))
        
        self.n_time_bins = N
        self.n_spectral_bins = N
        
    def fit_transform(self, traces):
        if self.n_time_bins is None or self.n_spectral_bins is None:
            warnings.warn("time bins/spectral bins not specified, setting balanced bins")
            self.equilibrate_n_coeff(traces)
            
        min_trace_length = np.min([len(tr) for tr in traces])//self.n_time_bins
        
        if min_trace_length//2 < self.n_spectral_bins:
            warnings.warn(f"Number of fourier is max {min_trace_length//2}")
        
        
        self.spectral_centroid = []
        self.spectral_mode = []
        self.STFT = []
        
        for tr_indx, tr in tqdm(enumerate(traces), total=len(traces)):

            # Removes Nans
            signal = tr[~np.isnan(tr)]

            # Gets values of stuff
            N, T = len(signal), 1.0/self.sampling_rate
            
            # Divides in windows
            windows = np.array_split(signal, self.n_time_bins)
            winlen = np.min([len(w) for w in windows])

            # Prepares cpntainers for STFT and gets the right frequencies
            STFT = np.zeros((self.n_time_bins, self.n_spectral_bins))
            FF = fftfreq(winlen, T)[:winlen//2]
            
            # Prepares STFT stats
            STFT_CENTROID = np.zeros(self.n_time_bins)
            STFT_MODE = np.zeros(self.n_time_bins)
            
            for i, part in enumerate(windows):
                regularized_window = np.hamming(len(part))*part
                window_spectrum = np.abs(fft(regularized_window)[:winlen//2])**2

                # Saves the binned STFT
                STFT[i, :] = np.log(STFTransformer._bin_spectral_energy(window_spectrum, 
                                                                         self.n_spectral_bins, 
                                                                         3000,
                                                                         sampling_rate=self.sampling_rate)
                                   )

                # Gets distribution
                energy_density = window_spectrum/np.sum(window_spectrum)

                # Gets weighted mean
                STFT_CENTROID[i] = np.sum(energy_density*FF)

                # Gets argmax (i.e. the mode)
                STFT_MODE[i] = FF[np.argmax(energy_density)]

            self.spectral_centroid.append(STFT_CENTROID)
            self.spectral_mode.append(STFT_MODE)
            self.STFT.append(STFT)

            
        # Makes them array
        self.spectral_centroid = np.array(self.spectral_centroid)
        self.spectral_mode = np.array(self.spectral_centroid)
        self.STFT = np.array(self.STFT)
        
        return self.STFT