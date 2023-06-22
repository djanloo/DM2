"""
Some shorthands to do stuff with heavily downsampled short-time moving intensity time series.

Author: djanloo
"""
import numpy as np
import pandas as pd
from scipy.signal import decimate, find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

class PhoneticTrack:
    """Basically is the mean intensity, where the mean is performed on a rolling
    window of given length.
    """
    def __init__(self, track, sampling_rate):
        self.track = track
        self.sampling_rate = sampling_rate
        self.duration_s = len(track)/sampling_rate
        self.decimation = None
        self.source_track = None
        self.sampling_rate_rescale = None
    
    @property
    def time(self):
        return np.linspace(0, self.duration_s, len(self.track))
    
    @property
    def source_time(self):
        return np.linspace(0, len(self.source_track)/self.sampling_rate/self.sampling_rate_rescale, len(self.source_track))
    
    def get_window_from_ms(self, window_in_ms):
        return int(window_in_ms*1e-3*self.sampling_rate)
    
    @classmethod
    def _get_window_from_ms(cls, window_in_ms, sampling_rate):
        return int(window_in_ms*1e-3*sampling_rate)
    
    @classmethod
    def _from_single_track(cls, track, sampling_rate, window_ms, decimation=None, **rolling_kwargs):
        if window_ms > 0.0:
            ph = pd.Series(track).rolling(
                                        window=PhoneticTrack._get_window_from_ms(window_ms, sampling_rate), 
                                        **rolling_kwargs
                                        ).std().fillna(0)
        else:
            ph = pd.Series(track)
        sampling_rate_rescale = 1
        if decimation is not None:
            ph = decimate(ph,q=decimation)
            sampling_rate_rescale = decimation
        new = PhoneticTrack(ph, sampling_rate/sampling_rate_rescale)
        new.decimation = decimation
        new.source_track = track
        new.sampling_rate_rescale = sampling_rate_rescale
        return new

    def get_index_in_original_audio(self, index):
        if self.decimation is None:
            raise ValueError("cound not infer decimation since track was not created by audio")
        else:
            return int(index*self.decimation)
    
class PhoneticList:
    """A list of phonetic tracks that wraps some functionalities"""
    def __init__(self):
        self.elements = []
    
    @classmethod
    def from_tracks(cls, tracks, sampling_rates, window_ms, decimate=None, **rolling_kwargs):
        obj = PhoneticList()
        for track,sr in zip(tracks, sampling_rates):
            new = PhoneticTrack._from_single_track(track, sr, window_ms, decimate, **rolling_kwargs)
            obj.elements.append(new)
        return obj
    
    @property
    def times(self):
        return [el.time for el in self.elements]
       
    @property
    def source_times(self):
        return [el.source_time for el in self.elements]
    
    @property
    def tracks(self):
        return [el.track for el in self.elements]
    
    @property
    def source_tracks(self):
        return [el.source_track for el in self.elements]

    def get_indexes_in_original_audios(self, indexes):
        return [el.get_index_in_original_audio(i) for el, i in zip(self.elements, indexes)]
    
    def standardize_length(self, length=np.inf, time_rescale=True):
        if time_rescale:
            # If time rescaling is on, chooses the standard length of traces
            std_len = min(length, np.min([len(tr) for tr in self.tracks]))
        if np.isfinite(length) and std_len != length:
            raise ValueError(f"specified length {length} is too long because smaller track is long {std_len}")
        standardised_phonetic_traces = pd.DataFrame()
        for track_idx,track in enumerate(self.tracks):

            # Generates the downsampling indexes
            if time_rescale:
                indexes = (len(track) - 1) * (np.linspace(0, 1, std_len))
                indexes = indexes.astype(int)
                time_rescale_factor = len(track) / std_len

                # Check repeated values
                indexes_unique, counts = np.unique(indexes, return_counts=True)
                if (counts > 1).any():
                    print(f"repeated index in track {track_idx}")
            else:
                indexes = np.arange(len(track))

            # Scales the signal here
            phonetic_signal = MinMaxScaler().fit_transform(track[indexes].reshape(-1, 1)).reshape(-1)
            row = pd.DataFrame(
                               [[phonetic_signal, time_rescale_factor]],
                               columns=["standard_phonetic_trace", "time_rescale"], 
                               index=[0]
                              )
            standardised_phonetic_traces = pd.concat([standardised_phonetic_traces, row], ignore_index=True)

        return standardised_phonetic_traces
    
    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, index):
        return self.elements[index]
            

class SyllablesDivider:
    """Divides a phonetic track in syllables by threshold on intensity derivative or by number of syllables"""
    def __init__(self, 
                 min_size_ms=0.1, 
                 smoothing_window_ms=0.5, 
                 derivative_threshold=0.2,
                 n_syllables=None,
                 endpoint=True
                ):
        self.min_size_ms = min_size_ms
        self.smoothing_window_ms = smoothing_window_ms
        
        # If number of syllables is specified, the other parameter is deactivated
        if n_syllables is not None:
            self._method = "syllables_number"
            self.n_syllables = n_syllables
            self.derivative_threshold = None
        else:
            self._method = "intensity_threshold"
            self.derivative_threshold = derivative_threshold
            
        self._phonetic_list = None
        self._start_indexes = None
        self._start_times = None
        self._endpoint = endpoint
        self._intensities = None
        
    @property
    def start_indexes(self):
        if self._start_indexes is None:
            raise RuntimeError("SyllablesDivider must be fitted to some phonetic list before")
        else:
            return self._start_indexes
       
    @property
    def start_times(self):
        if self._start_times is None:
            raise RuntimeError("SyllablesDivider must be fitted to some phonetic list before")
        else:
            return self._start_times
    
    @property
    def intensities(self):
        if self._phonetic_list is None:
            raise RuntimeError("SyllablesDivider must be fitted to some phonetic list before")
        
        # If already computed, return it
        if self._intensities is not None:
            return self._intensities
        
        # Otherwise does the computation
        else:
            self._intensities = []
            for i in range(len(self._phonetic_list)):
                phtr = self._phonetic_list[i]

                starts = [phtr.get_index_in_original_audio(st) for st in self._start_indexes[i][1:-1]]
                syll_audios = np.split(phtr.source_track, starts)
                track_intensities = np.array([np.std(fragment) for fragment in syll_audios])
                track_intensities = 20*np.log10(track_intensities)
                if np.isnan(track_intensities).any():
                    print(f"Some syllable had NaN intensity\nTrack {i}\nIndexes = {starts}")
                track_intensities[np.isnan(track_intensities)] = 0.0

                self._intensities.append(track_intensities)
                
            self._intensities = np.array(self._intensities)
            return self._intensities
    
    @property
    def duration_ms(self):
        if self._phonetic_list is None:
            raise RuntimeError("SyllablesDivider must be fitted to some phonetic list before")
            
        start_times = np.array(self._start_times)
        return np.diff(start_times, axis=-1)
        
    @classmethod       
    def _boolean_clusters(cls, x, min_size):
        """Finds clusters of True inside a boolean vector.
        Returns the center of mass of each cluster.
        
        Used when method == 'intensity_threshold'
        """
        u = np.arange(len(x))
        u[np.logical_not(x)] = 0
        du = np.diff(u)

        start_cluster = np.where(du > 1)[0]
        end_cluster  = np.where(du < -1)[0]

        if x[0]:
            start_cluster = np.concatenate(([0], start_cluster))
            if np.where(du != 1)[0][0] not in end_cluster:
                end_cluster = np.concatenate(([np.where(du != 1)[0][0]], end_cluster ))

        if x[-1]:
            end_cluster = np.concatenate((end_cluster, [len(x) -1]))

        if len(start_cluster) != len(end_cluster):
            raise RuntimeError(f"boolean clusters cannot be found\nstart {start_cluster}, end {end_cluster}")
            
        # Removes clusters that are too short
        good_clusters = (end_cluster - start_cluster) > min_size
        return (start_cluster[good_clusters], end_cluster[good_clusters])
   

    def _steepest_n_regions(self, derivative, min_size, endpoint=True):
        u = find_peaks(derivative, width=min_size)        
        indexes_of_bests = np.argsort(u[1]["prominences"])[-(self.n_syllables):]
        best_peaks = u[0][indexes_of_bests]
        widths = u[1]["widths"][indexes_of_bests]
        
        # Sorting
        argsort_peaks = np.argsort(best_peaks)
        starts, ends = (best_peaks - widths/2).astype(int)[argsort_peaks],  (best_peaks + widths/2).astype(int)[argsort_peaks]
        if endpoint:
            starts = np.append(starts, [len(derivative)])
            ends = np.append(ends, [len(derivative)])
        return np.sort(starts), np.sort(ends)
    
    def fit(self, phonetic_list, return_times=False):
        self._phonetic_list = phonetic_list
        self._start_indexes = []
        self._start_times = []
        for ph_tr in tqdm(phonetic_list.elements):            
            # Get the min size and the window for the given track
            min_size = ph_tr.get_window_from_ms(self.min_size_ms)
            window = ph_tr.get_window_from_ms(self.smoothing_window_ms)

            # Rolles the phonetic trace and removes nans
            rolled_trace = pd.Series(ph_tr.track).rolling(window=window, center=True, min_periods=5).mean()
            rolled_trace = rolled_trace.fillna(rolled_trace[window+1])

            # Takes the derivative of the smoothed phonetic trace
            derivative = np.diff(rolled_trace)
            
            # Smoothes the derivative and removes nans
            derivative = pd.Series(derivative).rolling(window=window).mean()
            derivative = derivative.fillna(derivative[window+1]).values
            
            # Normalizes the derivative
            derivative /= np.max(derivative)
            
            # Take the starting point and the end point
            # of the transition between two syllables
            if self._method == "intensity_threshold":
                start_transition, end_transition = SyllablesDivider._boolean_clusters(
                                                   derivative > self.derivative_threshold, 
                                                   min_size
                                                   )
            else:
                start_transition, end_transition = self._steepest_n_regions(derivative, min_size, endpoint=self._endpoint)
            
            
            start_syll = start_transition
            start_syll[0] = 0
            
            self._start_indexes.append(start_syll)
            self._start_times.append(start_syll/ph_tr.sampling_rate)
    
    def transform(self):
        """Returns the phonetic list of syllables"""
        syllables_tracks = []
        padded_syllables_tracks = []
        # Splitting
        for i in tqdm(range(len(self._phonetic_list))):
            phtr = self._phonetic_list[i]
            starts = [phtr.get_index_in_original_audio(st) for st in self._start_indexes[i][1:-1]]
            syll_audios = np.split(phtr.source_track, starts)
            syllables_tracks.append(syll_audios)
        
        # Padding is different for each syllable
        for sy in tqdm(range(self.n_syllables)):

            # Finds max length of the current syllable
            maxlen = -1
            for i in range(len(syllables_tracks)): 
                l = len(syllables_tracks[i][sy])
                maxlen = l if l>maxlen else maxlen
            
            Y = np.zeros((len(syllables_tracks), maxlen))
            for i in range(len(syllables_tracks)):
                Y[i] = np.pad(syllables_tracks[i][sy],
                                (0, maxlen - len(syllables_tracks[i][sy])),
                                constant_values=(np.nan, np.nan))
            padded_syllables_tracks.append(Y)
        return padded_syllables_tracks



    
    
            
