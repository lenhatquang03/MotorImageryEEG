import numpy as np
import mne 
import tensorflow as tf

def recording_time(start_signal : int, end_signal : int, sampling_freq : int) -> float:
    """ 
    Calculate the recording time given the first, last signals, and the sampling frequency (Hz). Signals are indexed nominally starting from 1.

    Parameters:
    ----------
    start_signal : int
        The start signal
    end_signal : int
        The last signal
    sampling_freq : int
        Signals are sampled at this frequency
    
    Returns:
    ----------
    float
        Recording time
    """
    return (end_signal - start_signal + 1) / sampling_freq

def timestamp(curr_signal : int, start_signal : int, sampling_freq : int) -> float:
    """
    Calculate the time point (in seconds) when the signal is recorded. Signals are indexed nominally starting from 1.
    
    Parameters:
    ----------
    curr_signal : int
        You want the timestamp of this signal
    start_signal : int
        The first meaningful signal of the recording session with timestamp 0
    sampling_freq : int
        Signals are sampled at this frequency
    
    Returns:
    ----------
    float
        The timestamp of the signal"""
    return (curr_signal - start_signal) / sampling_freq

def add_gaussian_noise(epochs : mne.EpochsArray, std_ratio=0.01) -> mne.EpochsArray:
    """
    Add Gaussian noise to the raw EEG signals

    Parameters:
    ----------
    epochs : mne.EpochsArray
        The object containing raw EEG signals
    std_ratio : float
        The ratio between the standard deviation of the Gaussian noise and that of the EEG signals.
    
    Returns:
    ----------
    mne.EpochsArray
        The object containinh noisy EEG signals, default is 0.01
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    std = np.std(data)
    noise = np.random.normal(0, std_ratio * std, data.shape)
    noisy_data = data + noise

    # Create new epochs object with noisy data
    noisy_epochs = mne.EpochsArray(noisy_data, epochs.info,
                                   tmin = epochs.tmin, events = epochs.events, event_id = epochs.event_id,
                                   baseline = epochs.baseline, metadata = epochs.metadata)
    return noisy_epochs

def shuffle_epochs(epochs : mne.EpochsArray, random_seed = 42) -> mne.EpochsArray:
    """
    Shuffle epochs (built around trial stimulation events) created from raw EEG signals
    
    Parameters:
    ----------
    epochs : mne.EpochsArray
        The object containing the epochs
    random_seed : int
        A seed to intialize a random generator for reproducible results
    
    Returns:
    ----------
    mne.EpochsArray
        The object with the shuffled epochs
    """
    # Create a random generator
    rng = np.random.default_rng(random_seed)
    permuted_idx = rng.permutation(len(epochs))

    # Shuffle important information
    data = epochs.get_data()
    shuffled_data = data[permuted_idx]
    shuffled_events = epochs.events[permuted_idx]
    shuffled_metadata = epochs.metadata.iloc[permuted_idx]

    # Build mne.EpochsArray from shuffled information
    shuffled_epochs = mne.EpochsArray(shuffled_data, epochs.info,
                                      tmin = epochs.tmin, events = shuffled_events, 
                                      event_id = epochs.event_id, baseline = epochs.baseline, 
                                      metadata = shuffled_metadata)
    return shuffled_epochs 

def z_normalize(sample : np.ndarray) -> np.ndarray:
  """
  Do the Z-normalization for the vector of extracted WT features of an raw EEG epoch over 21 EEG channels
  
  Parameters:
  ----------
  sample : np.ndarray
    The feature vector after apply the Wavelet Transform for a raw EEG epoch

  Returns:
  ----------
  np.ndarray
    The z-normalized feature vector ready for model training
  """
  mean = tf.math.reduce_mean(sample, axis = [0, 1], keepdims = True)
  std = tf.math.reduce_std(sample, axis = [0, 1], keepdims = True)
  return (sample - mean) / (std + 1e-8)