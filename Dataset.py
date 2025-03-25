from typing import Union
from scipy.io import loadmat
import numpy as np
import mne

class Dataset:
    """
    A class to extract relevant data from EEG .mat files
    
    Attributes:
    ----------
    __file_path : str
        Path to the .mat file
    __ID : numpy.str_
        The unique identifier of the dataset
    __tag : numpy.str_
        Tag associated with the dataset
    __sampling_freq : numpy.uint8
        The sampling frequency of EEG signals (aka the number of samples recorded per second)
    __num_of_samples : numpy.int32
        The total number of samples recorded
    __marker : numpy.ndarray
        Events (classes) corresponding to each sample
    __data : numpy.ndarray | mne.io.RawArray
        EEG data samples for each channels
    __channels : list
        List of names of channels used
    __binsuV : numpy.uint8
        Binned microvolt value 
    
    Methods:
    ----------
    extract_file(self) -> None
        Extract key data fields from the Matlab structure "o"
    
    format_data(self) -> None
        Modify the data based on the many formats described in the MNE-Python package, where self.__data is transformed into an mne.io.RawArray object

    add_montage(self) -> None
        Montage: the specific arrangement and display of EEG channels on the EEG records
        Explicitly set the montage for our data (only if it isn't already in the .mat data file)
    """

    def __init__(self, file_path : str):
        self.__file_path = file_path
        self.__ID = None
        self.__tag = None
        self.__sampling_freq = None
        self.__num_of_samples = None
        self.__marker = None
        self.__data = None
        self.__channels = None
        self.__binsuV = None

    @property
    def ID(self) -> np.str_:
        return self.__ID
    
    @property
    def tag(self) -> np.str_:
        return self.__tag 
    
    @property
    def sampling_freq(self) -> np.uint8:
        return self.__sampling_freq
    
    @property
    def num_of_samples(self) -> np.int32:
        return self.__num_of_samples
    
    @property
    def marker(self) -> np.ndarray:
        return self.__marker
    
    @property
    def data(self) -> Union[np.ndarray, mne.io.RawArray]:
        return self.__data
    
    @data.setter
    def data(self, new_data : np.ndarray) -> None:
        self.__data = new_data
        
    @property
    def channels(self) -> list:
        return self.__channels
    
    @property
    def binsuV(self) -> np.uint8:
        return self.__binsuV
    

    def extract_file(self):
        data_file = loadmat(self.__file_path)
        file_structure = data_file["o"]
        self.__ID = file_structure["id"][0][0][0]
        self.__tag = file_structure["tag"][0][0][0]
        self.__sampling_freq = file_structure["sampFreq"][0][0][0][0]
        self.__num_of_samples = file_structure["nS"][0][0][0][0]
        self.__marker = file_structure["marker"][0][0].flatten()
        self.__data = file_structure["data"][0][0]
        self.__channels = [file_structure["chnames"][0][0][i][0][0] for i in range(22)]
        self.__binsuV = file_structure["binsuV"][0][0][0][0]

    def format_data(self):
        """
        If we are to use the MNE library, some formatting are needed.
        - Annotations (event markers) must be positive integers. MNE-Python views 0 as no event/neutral state. Here, we want to analyze even the neutral state.
        - Each EEG sampled values must be of Volt and NOT micro-Volt.
        - Our raw data must be of shape (num_of_channels, num_of_samples).
        """
        starts = np.arange(self.num_of_samples) / self.sampling_freq
        event_duration = np.array([1/self.sampling_freq] * self.num_of_samples)
        events = mne.Annotations(
            onset = starts,
            duration = event_duration, 
            description = self.marker
        )

        channel_type = ["eeg"] * len(self.channels)
        data_info = mne.create_info(
            ch_names = self.channels, 
            sfreq = self.sampling_freq, 
            ch_types = channel_type
        )
        self.data = self.data.T / 1e6
        # Transform self.__data into an mne.io.RawArray object
        self.data = mne.io.RawArray(self.data, data_info)
        self.data.set_annotations(events)
        # The channel X5 is purely used for synchronization between each event and its marker. Removing this channel after the events are added to the data is safe and sensible.
        self.data.drop_channels(["X5"])
        self.data.load_data()

    def add_montage(self) -> None:
        standard_1020 = mne.channels.make_standard_montage("standard_1020")
        channel_ids = [i for (i, channel) in enumerate(standard_1020.ch_names) if channel in self.data.ch_names]
        montage = standard_1020.copy()
        montage.ch_names = [standard_1020.ch_names[j] for j in channel_ids]
        channel_info = [standard_1020.dig[j + 3] for j in channel_ids]
        montage.dig = standard_1020.dig[0:3] + channel_info
        self.__data.set_montage(montage)