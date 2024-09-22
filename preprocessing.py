from scipy.signal import butter, iirnotch, filtfilt
import copy
import numpy as np

def butter_bandpass(lowcut=1, highcut=40, sampling_rate=200, order=1):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch(freq=50, sampling_rate=200, quality_factor=30):
    b, a = iirnotch(freq, quality_factor, sampling_rate)
    return b, a

def apply_filter(numerator, denominator, data):
    return filtfilt(numerator, denominator, data)

def preprocess_dataframe(df, lowcut=1, highcut=40, sampling_rate=200, order=1, notch_freq=50, notch_quality_factor=30):
    b, a = butter_bandpass(lowcut, highcut, sampling_rate, order)
    b_notch, a_notch = notch(notch_freq, sampling_rate, notch_quality_factor)
    for column in df.columns:
        df[column] = apply_filter(b, a, df[column])
        df[column] = apply_filter(b_notch, a_notch, df[column])
    return df

def preprocess_dataframe_dict(df_dict, lowcut=1, highcut=40, sampling_rate=200, order=1, notch_freq=50, notch_quality_factor=30):
    dict_copy = copy.deepcopy(df_dict)
    for subject in dict_copy:
        for experiment in dict_copy[subject]:
            experiment_data = dict_copy[subject][experiment]
            if isinstance(experiment_data, list):
                for i, session in enumerate(experiment_data):
                    dict_copy[subject][experiment][i] = preprocess_dataframe(session, lowcut, highcut, sampling_rate, order, notch_freq, notch_quality_factor)
            else:
                dict_copy[subject][experiment] = preprocess_dataframe(experiment_data, lowcut, highcut, sampling_rate, order, notch_freq, notch_quality_factor)
    return dict_copy

def epoch_data_from_dataframe(dataframe, epoch_length=4, sampling_rate=200):
    n_samples_per_epoch = epoch_length * sampling_rate
    n_epochs = len(dataframe) // n_samples_per_epoch
    return np.array_split(dataframe, n_epochs)

def create_subject_experiment_data_dict(dataframe_dict, experiment_id):
    subject_experiment_data_dict = {}
    for subject in dataframe_dict:
        subject_experiment_data_dict[subject] = []
        experiment_data = dataframe_dict[subject][experiment_id]
        if isinstance(experiment_data, list):
            subject_experiment_data_dict[subject].extend(experiment_data)
        else:
            subject_experiment_data_dict[subject].extend([experiment_data])
    return subject_experiment_data_dict

def epoch_subject_experiment_data_dict(subject_experiment_data_dict, epoch_length=4, sampling_rate=200):
    dict_copy = copy.deepcopy(subject_experiment_data_dict)
    for subject in dict_copy:
        epoched_data = []
        for i, session in enumerate(dict_copy[subject]):
            epoched_data.extend(epoch_data_from_dataframe(session, epoch_length, sampling_rate))
        dict_copy[subject] = epoched_data
    return dict_copy