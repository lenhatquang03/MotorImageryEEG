from typing import Literal, List
from DataFile import DataFile
from scipy.signal.windows import hann
import numpy as np
import mne

def extract_FFT_amplitudes(format_data : mne.io.RawArray, tmin = 0, tmax = 0.85, baseline = None) -> np.ndarray:
    """
    Extract the Fast Fourier transform amplitudes from an mne.io.RawArray object
    
    Parameters:
    ----------
    format_data : mne.io.RawArray
        Data converted to MNE format (details in the DataFile class)
    tmin, tmax: float
        Start and end time of the epochs in seconds (which are built around time-locked events as time point 0). Defaults to 0 and 0.85, respectively.
    baseline : None | tuple of length 2
        Time interval for baseline correction. If a tuple (a, b), the interval is between a and b (in seconds), including the endpoints. If a is None, the beginning of the data is used; and if b is None, it is set to the end of the data. If (None, None), the entire time interval is used.
        Correction is applied to each epoch and channel individually in the following way:
        1. Calculate the mean signal of the baseline period
        2. Subtract this mean from the entire epoch
    
    Returns:
    ----------
    np.ndarray
        The vector containing features for each epoch flatten over all 21 channels
    """
    events, event_id = mne.events_from_annotations(format_data)
    epochs = mne.Epochs(format_data, events, event_id, 
                        tmin = tmin, tmax = tmax, 
                        baseline = baseline, preload = True)
    epoched_data = epochs.get_data() # Shape: (n_samples, n_channels, n_signals)
    # Window
    window = hann(epoched_data.shape[2], sym = False)
    windowed_data = epoched_data * window
    # Zero-pad
    padded_len = 2 ** int(np.ceil(np.log2(windowed_data.shape[2])))
    padded_data = np.pad(windowed_data, 
                         pad_width = ((0, 0), (0, 0), (0, padded_len - windowed_data.shape[2])), 
                         mode = "constant")
    # Fast Fourier transform amplitudes
    fft_vals = np.fft.rfft(padded_data, axis = 2)
    amplitudes = np.abs(fft_vals)
    feature_vector = amplitudes.reshape(amplitudes.shape[0], amplitudes.shape[1] * amplitudes.shape[2])
    return feature_vector

class FeatureExtractor:
    """
    A class to extract EEG features from raw signal

    Attributes:
    ----------
    data_files : List[DataFile]
        a list of all formatted data file
    
    Methods:
    ----------
    extract(self, feature : Literal["FTA", "Wavelet", "time series"], merge = True)
        Extract signal features from the data files
    """
    def __init__(self, data_files : List[DataFile]):
        self.__data_files = data_files
    
    @property
    def data_files(self) -> List[DataFile]:
        return self.__data_files

    def extract(self, feature : Literal["FTA", "Wavelet", "time series"]):
        if (feature == "FTA"):
            features = [extract_FFT_amplitudes(data_file.format_data) for data_file in self.data_files]
            return features