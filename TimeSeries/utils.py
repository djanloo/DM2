import numpy as np
import pandas as pd
from scipy.signal import decimate

class PhoneticTrack:
    """Basically is the mean intensity, where the mean is performed on a rolling
    window of given length.
    """
    def __init__(self, track, sampling_rate):
        self.track = track
        self.sampling_rate = sampling_rate
        self.duration_s = len(track)/sampling_rate
        self.time = np.linspace(0, self.duration_s, len(track))
        self.decimated =None
    
    def get_window_from_ms(self, window_in_ms):
        return int(window_in_ms*1e-3*self.sampling_rate)
    
    @classmethod
    def _get_window_from_ms(cls, window_in_ms, sampling_rate):
        return int(window_in_ms*1e-3*sampling_rate)
    
    @classmethod
    def _from_single_track(cls, track, sampling_rate, window_ms, decimate=None):
        ph = pd.Series(track).rolling(window=PhoneticTrack._get_window_from_ms(window_ms, sampling_rate)).std().fillna(0)
        sampling_rate_rescale = 1
        if decimate is not None:
            ph = decimate(ph,q=decimate)
            sampling_rate_rescale = decimate
        new = PhoneticTrack(ph, sampling_rate/sampling_rate_rescale)
        new.decimated = decimate
        return new

    def get_index_in_original_audio(self, index):
        if self.decimated is None:
            raise ValueError("cound not infer decimation since track was not created by audio")
        else:
            return int(index*self.decimated)
    
class PhoneticList:
    """A list of phonetic tracks that wraps some functionalities"""
    def __init__(self):
        self.elements = []
        self.tracks = []
        self.times = []
    
    @classmethod
    def from_tracks(cls, tracks, sampling_rates, window_ms, decimate=None):
        obj = PhoneticList()
        for track,sr in zip(tracks, sampling_rates):
            new = PhoneticTrack._from_single_track(track, sr, window_ms, decimate)
            obj.elements.append(new)
            obj.tracks.append(new.track)
            obj.times.append(new.time)
        return obj

    def get_indexes_in_original_audios(self, indexes):
        return [el.get_index_in_original_audio(i) for el, i in zip(self.elements, indexes)]
            

class SyllablesDivider:
    """Divides a phonetic track in syllables by threshold on intensity derivative"""
    def __init__(self, min_size_ms=0.1, smoothing_window_ms=0.5, derivative_threshold=0.2):
        self.min_size_ms = min_size_ms
        self.smoothing_window_ms = smoothing_window_ms
        self.derivative_threshold = derivative_threshold
        
    @classmethod       
    def _boolean_clusters(cls, x, min_size):
        """Finds clusters of True inside a boolean vector.
        Returns the center of mass of each cluster.
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

    
    def get_start_points(self, phonetic_list, return_times=False):
        start_indexes = []
        start_times = []
        for ph_tr in phonetic_list.elements:            
            # Get the min size and the window for the given track
            min_size = ph_tr.get_window_from_ms(self.min_size_ms)
            window = ph_tr.get_window_from_ms(self.smoothing_window_ms)

            # Rolles 
            rolled_trace = pd.Series(ph_tr.track).rolling(window=window).mean()
            rolled_trace = rolled_trace.fillna(rolled_trace[window+1])

            # Takes the derivative of the smoothed track
            derivative = np.diff(rolled_trace)/np.max(np.diff(rolled_trace))
            
            # Take the starting point and the end point
            # of the transition between two syllables
            start_transition, end_transition = SyllablesDivider._boolean_clusters(derivative > self.derivative_threshold, min_size)

            # Since the rolling window moves the track orward in time, I subtract half of the window
            # to take care of this effect
            start_syll = start_transition - window//2
            
            start_indexes.append(start_syll)
            start_times.append(start_syll/ph_tr.sampling_rate)
        if return_times:
            return start_indexes, start_times
        return start_indexes