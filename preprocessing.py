from scipy.signal import butter, iirnotch, filtfilt

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
    for subject in df_dict:
        for experiment in df_dict[subject]:
            experiment_data = df_dict[subject][experiment]
            if isinstance(experiment_data, list):
                for i, session in enumerate(experiment_data):
                    df_dict[subject][experiment][i] = preprocess_dataframe(session, lowcut, highcut, sampling_rate, order, notch_freq, notch_quality_factor)
            else:
                df_dict[subject][experiment] = preprocess_dataframe(experiment_data, lowcut, highcut, sampling_rate, order, notch_freq, notch_quality_factor)
    return df_dict