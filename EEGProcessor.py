import matplotlib.pyplot as plt
import mne
from Dataset import Dataset
import numpy as np
from typing import Literal

class EEGProcessor:
    """
    A class to preprocess data from extracted information
    """
    def __init__(self, dataset : Dataset):
        self.__ica = None
        self.__dataset = dataset
        self.__raw = dataset.data.copy()
        self.__cleaned_data = dataset.data.copy()

    @property
    def dataset(self) -> Dataset:
        return self.__dataset
    
    @property
    def raw(self) -> mne.io.RawArray:
        return self.__raw
    
    @raw.setter
    def raw(self, new_data : mne.io.RawArray):
        self.__raw = new_data
    
    @property
    def cleaned_data(self) -> mne.io.RawArray:
        return self.__cleaned_data
    
    @cleaned_data.setter
    def cleaned_data(self, new_data : mne.io.RawArray) -> None:
        self.__cleaned_data = new_data
    
    @property
    def ica(self) -> mne.preprocessing.ICA:
        return self.__ica
    
    @ica.setter
    def ica(self, technique : mne.preprocessing.ICA) -> None:
        self.__ica = technique
    
    def plot_psd(self, plot_name : str) -> None:
        """
        Compute the power spectral density of each channel and plot them out
        """
        psd_figure = self.cleaned_data.compute_psd().plot()
        # Renaming the figure
        for ax in psd_figure.axes:
            ax.set_title("")
        psd_figure.suptitle(plot_name, fontweight = "bold")
        plt.show()
        # Comment this for non-interactive plots
        plt.close("all")
    
    def set_eog_channel(self) -> None:
        """
        EOG (Electro-oculography): EEG signals produced by eye movements 

        Create an EOG channel from the EEG channels, which will be used solely for EOG artifact correction
        """
        # Setting for the original data
        self.raw = mne.set_bipolar_reference(self.raw, anode = "Fp1", cathode = "Fp2", ch_name = "EOG", drop_refs = False)
        self.raw.set_channel_types({"EOG" : "eog"})
        # Setting for the to-be-cleaned data 
        self.cleaned_data = mne.set_bipolar_reference(self.cleaned_data, anode = "Fp1", cathode = "Fp2", ch_name = "EOG", drop_refs = False)
        self.cleaned_data.set_channel_types({"EOG" : "eog"})

    def set_ecg_channel(self) -> None:
        """
        ECG (Electrocardiography): Electrical activity of the heart

        Create an ECG channel from the EEG channels, which will be used solely for ECG artifact correction. 
        """
        channels_for_ecg = ["C3", "C4", "Cz"]
        ecg_signal = self.raw.get_data(picks = channels_for_ecg).mean(axis = 0)
        # ecg_data is required to be of shape (1, num_of_samples)
        ecg_data = np.reshape(ecg_signal, (1, -1))
        ecg_info = mne.create_info(ch_names = ["ECG"], sfreq = self.dataset.sampling_freq, ch_types = ["ecg"])
        ecg_channel = mne.io.RawArray(ecg_data, ecg_info)
        # Setting the channel for both the original and to-be-cleaned data
        self.cleaned_data.add_channels([ecg_channel], force_update_info = True)
        self.raw.add_channels([ecg_channel], force_update_info = True)

    def detect_eog(self, plot_name : str) -> None:
        """
        Detecting and visualizing EOG artifacts on raw continuos data with the prerequisite of an EOG channel.
        """
        eog_id = {"EOG artifacts" : 998}
        eog_events = mne.preprocessing.find_eog_events(self.cleaned_data, ch_name = "EOG")
        # Each epoch starts 0.5s before an event and ends 1.5s after the event
        eog_epochs = mne.Epochs(self.cleaned_data, eog_events, event_id = eog_id, tmin = -0.5, tmax = 1.5, baseline = None, preload = True) 
        eog_figure = eog_epochs.average().plot()
        for ax in eog_figure.axes:
            ax.set_title("")
        eog_figure.suptitle(plot_name, fontweight = "bold")
        plt.show()
        plt.close("all")
    
    def detect_ecg(self, plot_name) -> None:
        """
        Detecting and visualizing EOG artifacts on raw continuos data with the prerequisite of an ECG channel.
        """
        ecg_id = {"ECG artifacts" : 999}
        ecg_events, _, _ = mne.preprocessing.find_ecg_events(self.cleaned_data, ch_name = "ECG")
        # Each epoch starts 0.5s before an event and ends 1.5s after the event
        ecg_epochs = mne.Epochs(self.cleaned_data, ecg_events, event_id = ecg_id, tmin = -0.5, tmax = 1.5, baseline = None, preload = True)
        # Renaming the figure
        ecg_figure = ecg_epochs.average().plot()
        for ax in ecg_figure.axes:
            ax.set_title("")
        ecg_figure.suptitle(plot_name, fontweight = "bold")
        plt.show()
        plt.close("all")


    def remove_artifacts(self, method : Literal["fastica", "picard", "infomax", "extended-infomax", "jade"]) -> None:
        """
        ICA (Independent Component Analysis) is used to remove EOG and ECG artifacts
        """
        self.ica = mne.preprocessing.ICA(random_state = 42, method = method)
        # High-pass filter for efficient ICA 
        self.cleaned_data.filter(l_freq = 1, h_freq = None)
        self.ica.fit(self.cleaned_data)
        # Automatically find components containing EOG and ECG artifacts
        eog_indices, _ = self.ica.find_bads_eog(self.cleaned_data)
        ecg_indices, _ = self.ica.find_bads_ecg(self.cleaned_data)
        # Artifact removal
        self.ica.exclude = ecg_indices + eog_indices
        self.ica.apply(self.cleaned_data)

    def process(self, saved = False) -> None:
        """
        Overwrite the raw data with the cleaned data 
        Automatize process for data processing of many data files 
        """
        # Data re-referencing
        self.cleaned_data.set_eeg_reference(["A1", "A2"])
        # EOG and ECG artifact removal
        self.set_eog_channel()
        self.set_ecg_channel()
        self.remove_artifacts()
        # Keeping only the most relevant information
        self.cleaned_data.filter(l_freq = 8, h_freq = 30)
        # Save the cleaned data
        if saved:
            self.cleaned_data.save("cleaned_data.fif")