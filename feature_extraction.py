import scipy
import numpy as np
from data import EEGDataset

"""
According to the paper: 'To avoid the problem of channel selection, each feature was
averaged across all the channels, after being extracted from each channel'. This is why
I'm averaging across channels.
"""
def extract_features_from_epoch(epoch, sampling_rate=200):
    features = []
    mean_amplitude = np.mean(epoch, axis=0)
    mean_mean_amplitude = np.mean(mean_amplitude)
    features.append(mean_mean_amplitude)

    std_amplitude = np.std(epoch, axis=0)
    mean_std_amplitude = np.mean(std_amplitude)
    features.append(mean_std_amplitude)

    range_amplitude = np.ptp(epoch, axis=0)
    mean_range_amplitude = np.mean(range_amplitude)
    features.append(mean_range_amplitude)

    kurtosis = compute_amplitude_kurtosis(epoch)
    mean_kurtosis = np.mean(kurtosis)
    features.append(mean_kurtosis)

    freqs, psd_array = compute_psd_from_epoch(epoch, sampling_rate)
    mean_psd_bands, std_psd_bands = compute_mean_and_std_psd_in_bands(freqs, psd_array)
    flattened_mean_psd_bands = np.array([mean_psd_bands[band] for band in mean_psd_bands]).flatten()
    features.extend(flattened_mean_psd_bands)
    flattened_std_psd_bands = np.array([std_psd_bands[band] for band in std_psd_bands]).flatten()
    features.extend(flattened_std_psd_bands)

    return features

def create_dataset_from_epoched_experiment_data_dict(epoched_experiment_data_dict, sampling_rate=200):
    extracted_features_dict = {}
    for subject in epoched_experiment_data_dict:
        extracted_features_dict[subject] = []
        for epoch in epoched_experiment_data_dict[subject]:
            features = extract_features_from_epoch(epoch, sampling_rate)
            extracted_features_dict[subject].append(features)

    return EEGDataset(extracted_features_dict)


def compute_psd_from_epoch(epoch, sampling_rate=200):
    psd_list = []
    freqs = None
    for channel in epoch.columns:
        freqs, psd = scipy.signal.welch(epoch[channel], fs=sampling_rate)
        psd_list.append(psd)
    psd_array = np.array(psd_list)
    return freqs, psd_array

def compute_mean_and_std_psd_in_bands(freqs, psd_array):
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 40)
    }
    mean_psd_bands = {}
    std_psd_bands = {}
    for band_name, (low_freq, high_freq) in bands.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        # Mean PSD within the band for each channel
        mean_psd_band = np.mean(psd_array[:, idx_band], axis=1)
        # standard deviation PSD within the band for each channel
        std_psd_band = np.std(psd_array[:, idx_band], axis=1)
        mean_psd_bands[band_name] = mean_psd_band  # Shape: (channels,)
        std_psd_bands[band_name] = std_psd_band
    return mean_psd_bands, std_psd_bands

def compute_amplitude_kurtosis(epoch):
    # epoch: array of shape (samples_per_epoch, channels)
    kurtosis_values = []
    for ch in epoch.columns:
        channel_data = epoch[ch]
        # Calculate kurtosis
        kurt = scipy.stats.kurtosis(channel_data, fisher=True, bias=False)
        kurtosis_values.append(kurt)
    return np.array(kurtosis_values)  # Shape: (channels,)