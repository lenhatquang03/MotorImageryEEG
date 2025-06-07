from typing import Literal, List
from DataFile import DataFile
from scipy.signal.windows import hann
import numpy as np
import mne
import gc

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

def extract_WT_power(epochs : mne.EpochsArray, directory : str, save_prefix = Literal["train", "validation", "final_train", "test"], freqs = np.arange(4, 40, 2), batch_size = 1000):
    """
    Extract time-frequecy features using the Wavelet Transforms from raw EEG epochs and save them in file .npy
    Due to RAM limitation, feature extraction from the whole raw EEG signals can't be done. Here, I resolve to divide the raw data into batches, extract the WT features for each of those batches, save them and later on create a custom Dataset that can generate those features on the fly while training.

    Parameters:
    ----------
    epochs : mne.EpochsArray
        The object containing the raw EEG epochs
    directory : str
        The directory to save .npy files of WT features
    save_prefix : Literal["train", "validation", "final_train", "test"]
        The purpose of the extracted features, whether they will serve as the training, validation, final_train(training + validation), or test data
    freqs : np.ndarray
        The array of frequencies of Morlet's daughter wavelets we use, default is np.arange(4, 40, 2) for biometric relevance
    batch_size : int
        The number of raw EEG epochs for each batch
    """
    n_epochs = len(epochs)
    i = 1

    # Batch extraction
    for start in range(0, n_epochs, batch_size):
        stop = min(start + batch_size, n_epochs)
        print(f"Processing epochs {start} to {stop}")
        batch_epoch = epochs[start:stop]
        power = mne.time_frequency.tfr_morlet(
            batch_epoch,
            freqs=freqs,
            n_cycles=freqs/2,
            return_itc=False,
            use_fft=True,
            decim = 3,
            average=False, # Keep shape: (n_epochs, n_channels, n_freqs, n_times)
            verbose = False
        )

        # Save batches to .npy files - directory = "data\\wavelet_features\\fold_{i}"
        if save_prefix in ["final_train", "test"]:
            save_path = f"{directory}\\batch_{i}.npy"
        else:
            save_path = f"{directory}\\{save_prefix}\\batch_{i}.npy"
        i += 1
        np.save(save_path, power.data.astype(np.float32).transpose(0, 2, 3, 1))

        # Release memory
        del batch_epoch, power
        gc.collect()


class FTAExtractor:
    """
    A class to extract EEG features from raw signal

    Attributes:
    ----------
    data_files : List[DataFile]
        a list of all formatted data file
    
    Methods:
    ----------
    extract(self, merge = True)
        Extract Fourier transform amplitudes features from the data files
    """
    def __init__(self, data_files : List[DataFile]):
        self.__data_files = data_files
    
    @property
    def data_files(self) -> List[DataFile]:
        return self.__data_files


    def extract(self):
        features = [extract_FFT_amplitudes(data_file.format_data) for data_file in self.data_files]
        return features