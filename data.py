import os, re
import pandas as pd
import mne
import logging

logger = logging.getLogger('DecodingNeuronalActivityProject.Data')

filename_scheme_re = re.compile(r'^(?P<subject>s\d{2})_(?P<experiment>ex\d{2})(?P<session>_s\d{2})?.csv$')
DATA_PATH = "auditory-evoked-potential-eeg-biometric-dataset-1.0.0"
ELECTRODES = ["T7", "F8", "Cz", "P4"]
SECONDS_FOR_DATA_TYPE = {"Segmented_Data": 120, "Filtered_Data": 120}
SFREQ = 200

def parse_filename(filename):
    match = filename_scheme_re.match(filename)
    if match is None:
        return None
    groupdict = match.groupdict()
    if groupdict['session'] is not None:
        groupdict['session'] = int(groupdict['session'][2:])
    groupdict['subject'] = int(groupdict['subject'][1:])
    groupdict['experiment'] = int(groupdict['experiment'][2:])
    return groupdict

def create_rawarray_dict_from_df_dict(df_dict):
    rawarray_dict = {}
    for subject in df_dict:
        rawarray_dict[subject] = {}
        for experiment in df_dict[subject]:
            experiment_data = df_dict[subject][experiment]
            if isinstance(experiment_data, list):
                rawarray_dict[subject][experiment] = []
                for session in experiment_data:
                    info = mne.create_info(ch_names=ELECTRODES, sfreq = SFREQ, ch_types='eeg')
                    raw = mne.io.RawArray(session.T, info, verbose=False)
                    rawarray_dict[subject][experiment].append(raw)
            else:
                info = mne.create_info(ch_names=ELECTRODES, sfreq = SFREQ, ch_types='eeg')
                raw = mne.io.RawArray(experiment_data.T, info, verbose=False)
                rawarray_dict[subject][experiment] = raw
    return rawarray_dict

def load_data(data_path: str = DATA_PATH, subfolder: str = 'Segmented_Data'):
    specific_data_path = os.path.join(data_path, subfolder)
    data = {}
    data_as_df = {}
    for filename in os.listdir(specific_data_path):
        if filename.endswith('.csv'):
            parsed_filename = parse_filename(filename)
            subject = parsed_filename['subject']
            experiment = parsed_filename['experiment']
            if parsed_filename is not None:
                with open(os.path.join(specific_data_path, filename), 'r') as file:
                    df = pd.read_csv(file)
                    df = df[ELECTRODES]
                    n_channels = len(df.columns)
                    assert n_channels == len(ELECTRODES)
                    sampling_freq = len(df) / SECONDS_FOR_DATA_TYPE[subfolder]
                    if sampling_freq != SFREQ:
                        logger.warning(f"Sampling frequency might not be {SFREQ} Hz, but {sampling_freq} Hz for {filename}")
                    info = mne.create_info(ch_names=ELECTRODES, sfreq = SFREQ, ch_types='eeg')
                    raw = mne.io.RawArray(df.T, info, verbose=False)
                    if subject not in data:
                        data[subject] = {}
                        data_as_df[subject] = {}
                    if experiment <= 2:
                        if experiment not in data[subject]:
                            data[subject][experiment] = []
                            data_as_df[subject][experiment] = []
                            data[subject][experiment].append(raw)
                            data_as_df[subject][experiment].append(df)
                        else:
                            data[subject][experiment].append(raw)
                            data_as_df[subject][experiment].append(df)
                    else:
                        data[subject][experiment] = raw
                        data_as_df[subject][experiment] = df
    return data, data_as_df

if __name__ == '__main__':
    segmented_data = load_data()
    first_subject_first_experiment_first_session = segmented_data[1][1][0]
    print(first_subject_first_experiment_first_session.describe())
