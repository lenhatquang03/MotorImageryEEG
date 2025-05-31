from typing import List
from Utilities import *
from scipy.io import loadmat
from datetime import datetime, timezone
import regex as re
import numpy as np
import mne

class Datafile:
    """
    A class to extract relevant data from EEG .mat files of the NoMT interaction paradigm
    
    Attributes:
    ----------
    __file_path : str
        Path to the .mat file
    __IMAGERY_ENCODE: dict
        The marker codes of the visual stimuli
    __ID : str
        The unique identifier of the dataset
    __tag : str
        Tag associated with the dataset
    __sampling_freq : int
        The sampling frequency of EEG signals (aka the number of samples recorded per second)
    __num_of_samples : int
        The total number of samples recorded
    __marker : numpy.ndarray
        Events (classes) corresponding to each sample
    __data : numpy.ndarray 
        EEG data samples for each channels
    __channels : List[str]
        List of names of channels used
    __binsuV : int
        Binned microVolt value 
    __format_data: mne.io.RawArray
        Formatted data for MNE-Python usage
    
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
    __IMAGERY_ENCODE = {
            1 : "blank",
            2 : "left hand",
            3: "right hand",
            4 : "passive or neutral",
            5 : "left leg",
            6 : "toungue",
            7 : "right leg",
            91 : "intersession breaks",
            92 : "experiment end",
            99 : "initial relaxation"
        }
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
        self.__format_data = None

    @property
    def file_path(self) -> str:
        return self.__file_path
    
    @classmethod
    def get_IMAGERY_ENCODE(cls) -> dict:
        return cls.__IMAGERY_ENCODE
    
    @property
    def ID(self) -> str:
        return self.__ID
    
    @property
    def tag(self) -> str:
        return self.__tag 
    
    @property
    def sampling_freq(self) -> int:
        return self.__sampling_freq
    
    @property
    def num_of_samples(self) -> int:
        return self.__num_of_samples
    
    @property
    def marker(self) -> np.ndarray:
        return self.__marker
    
    @property
    def data(self) -> np.ndarray:
        return self.__data
    
    @data.setter
    def data(self, new_data : np.ndarray) -> None:
        self.__data = new_data
        
    @property
    def channels(self) -> List[str]:
        return self.__channels
    
    @property
    def binsuV(self) -> int:
        return self.__binsuV
    
    @property
    def format_data(self) -> mne.io.RawArray:
        return self.__format_data
    
    @format_data.setter
    def format_data(self, format_data : mne.io.RawArray) -> None:
        self.__format_data = format_data

    def extract_file(self) -> None:
        """
        Extract key data fields from the Matlab structure "o" 
        """
        data_file = loadmat(self.__file_path)
        file_structure = data_file["o"]
        self.__ID = str(file_structure["id"][0][0][0])
        self.__tag = str(file_structure["tag"][0][0][0])
        self.__sampling_freq = int(file_structure["sampFreq"][0][0][0][0])
        self.__num_of_samples = int(file_structure["nS"][0][0][0][0])
        self.__marker = file_structure["marker"][0][0].flatten()
        self.__data = file_structure["data"][0][0]
        self.__channels = [str(file_structure["chnames"][0][0][i][0][0]) for i in range(22)]
        self.__binsuV = int(file_structure["binsuV"][0][0][0][0])
    
    def intersession_breaks(self, break_duration = 2) -> List[tuple]:
        """
        Detect breaks between recording sessions. There are at most 2 breaks.
        
        Parameters:
        ----------
        break_duration : int, optional
            How long breaks last in minutes (default is 2)
        
        Returns:
        ----------
        List[tuple]
            a list of 2-tuples containing the starting and ending signals of each break 
        """
        rest_signals = np.where(self.marker == 91)[0]
        num_of_breaks = len(rest_signals) // (self.sampling_freq * break_duration * 60)
        if (num_of_breaks == 1):
            breaks = [(rest_signals[0], rest_signals[-1])]
        else:
            first_break_start = rest_signals[0]
            second_break_end = rest_signals[-1]
            for i in range(len(rest_signals) - 1):
                if (rest_signals[i+1] - rest_signals[i] >= 24000):
                    first_break_end = rest_signals[i]
                    second_break_start = rest_signals[i+1]
                    break
            breaks = [(first_break_start, first_break_end), (second_break_start, second_break_end)]
        return breaks
    
    def trial_information(self) -> list:
        """
        Find the starting signal of all imagery trials, first and last MEANINGFUL trials. 
        - Each trial consisted of 1s of motor imagery and 1.5s - 2.5s random off-time which was also marked as 0, even for "intersession breaks", "experiment end", and "initial relaxation".
        - We want to remove the EEG data recorded in first 2.5 minutes of acclimatisation, the intersession breaks, and immediately after the experiment ends.

        Parameters:
        ----------
        None

        Returns:
        ----------
        list
            The list of all imagery trials, the indices of the first and last MEANINGFUL trials in that list
        """
        # Even elements in "trials" will have the 0 marker, and odd elements will have positive markers.
        trials = []
        trials.append((0, self.marker[0]))
        for i in range(len(self.marker) - 1):
            if self.marker[i] != self.marker[i+1]:
                trials.append((i+1, self.marker[i+1]))
        
        # The marker codes always start with 0
        for i in range(0, len(trials) - 1, 2):
            considered_marker = trials[i+1][1]
            if (bool(considered_marker)) and (considered_marker not in {91, 92, 99}):
                start_trial_idx = i
                break

        # The marker code always ends with 0
        for i in range(len(trials) - 1, 0, -1):
            considered_marker = trials[i - 1][1]
            if (bool(considered_marker)) and (considered_marker not in {91, 92, 99}):
                end_trial_idx = i
                break
        return trials, start_trial_idx, end_trial_idx
        
    def create_events(self):
        """
        Create events bassed on the appearance of eGUI stimulus action-signals for meaningful trials

        Parameters:
        ----------
        None

        Returns:
        mne.Annotations
            All good and bad events in the used data
        """
        # Collect trial information
        trials, start_trial_idx, end_trial_idx = self.trial_information()
        start_signal = trials[start_trial_idx][0]
        # Event descriptions
        event_onset = [timestamp(curr_signal=trials[k][0], 
                                 start_signal = start_signal, 
                                 sampling_freq= self.sampling_freq) for k in range(start_trial_idx, end_trial_idx + 1)
                        if trials[k][1] not in {91, 92, 99}]
        event_duration = [1] * len(event_onset)
        event_description = [Datafile.get_IMAGERY_ENCODE()[trials[k][1] + 1] for k in range(start_trial_idx, end_trial_idx + 1)
                        if trials[k][1] not in {91, 92, 99}]
        
        # "Good" events
        good_events = mne.Annotations(
            onset = event_onset,
            duration = event_duration,
            description = event_description 
        )

        # "Bad" events being the intersession breaks which are ignoreds
        breaks = self.intersession_breaks()
        bad_onset = [timestamp(curr_signal=breaks[i][0], 
                               start_signal=start_signal,
                               sampling_freq=self.sampling_freq) for i in range(len(breaks))]
        bad_duration = [recording_time(start_signal=breaks[i][0], 
                                       end_signal= breaks[i][1],
                                       sampling_freq=self.sampling_freq) for i in range(len(breaks))]
        bad_description = ["bad intersession break"] * len(breaks)
        bad_events = mne.Annotations(
            onset = bad_onset,
            duration = bad_duration,
            description = bad_description
        )

        return good_events + bad_events
    
    def convert_mne(self, drop_channels = ["X5"]) -> None:
        """
        Convert data to MNE's specified format.
        - Annotations (event markers) must be positive integers. MNE-Python views 0 as no event/neutral state. Here, we want to analyze even the neutral state.
        - Each EEG sampled values must be of Volt and NOT micro-Volt.
        - Our raw data must be of shape (num_of_channels, num_of_samples).

        Parameters:
        ----------
        drop_channels : list
            a list of channels to drop (default is X5 which is only used for synchronization)

        Returns:
        ----------
        None
        """
        # Create data info
        channel_types = ["eeg"] * len(self.channels)
        data_info = mne.create_info(
            ch_names = self.channels,
            sfreq = self.sampling_freq,
            ch_types = channel_types
        )

        # As binsuV = 1, our data is in microVolt. MNE-Python expects Volt though.
        self.data = self.data.T / 1e6

        # RawArray conversion
        trials, start_trial_idx, end_trial_idx = self.trial_information()
        start_signal = trials[start_trial_idx][0]
        end_signal = trials[end_trial_idx][0]
        self.format_data = mne.io.RawArray(self.data, data_info)
        self.format_data.crop(
            tmin = start_signal / self.sampling_freq, 
            tmax = end_signal / self.sampling_freq,
            include_tmax = True
        )
        events = self.create_events()
        self.format_data.set_annotations(events)
        self.format_data.drop_channels(drop_channels)
    
    def set_metadata(self) -> None:
        """
        Set additional information (experiment subject and date of the recording session) for our dataset

        Parameters:
        ----------
        None

        Returns:
        ----------
        None
        """
        # Extract metadata from the file name
        pattern = r"(Subject\w?)(\d{6})"
        file_name = re.search(r"NoMT.*", self.file_path).group(0)
        match = re.search(pattern, file_name)
        if match:
            subject = match.group(1)
            date = match.group(2)
        else:
            print("Collecting metdata fails!!")
        date = datetime(
            year = int('20'+ date[0:2]), 
            month = int(date[2:4]), 
            day = int(date[4:6]), 
            tzinfo = timezone.utc
        )

        # Set metadata
        self.format_data.set_meas_date(date)
        self.format_data.info["subject_info"] = {
            "his_id" : subject
        }

    def add_montage(self) -> None:
        """
        Add information about the used EEG montage in our dataset. A montage is a specific arrangement of EEG channels.
        
        Parameters:
        ----------
        None
         
        Returns:
        ----------
        None
         """
        standard_1020 = mne.channels.make_standard_montage("standard_1020")
        channel_ids = [i for (i, channel) in enumerate(standard_1020.ch_names) if channel in self.format_data.ch_names]
        montage = standard_1020.copy()
        montage.ch_names = [standard_1020.ch_names[j] for j in channel_ids]
        channel_info = [standard_1020.dig[j + 3] for j in channel_ids]
        montage.dig = standard_1020.dig[0:3] + channel_info
        self.format_data.set_montage(montage)