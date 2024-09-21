from scipy.signal import butter, lfilter, iirnotch, filtfilt
from sklearn.preprocessing import StandardScaler

def butter_bandpass(lowcut=1, highcut=40, sampling_rate=200, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch(freq=50, sampling_rate=200, quality_factor=30):
    nyquist = 0.5 * sampling_rate
    b, a = iirnotch(freq / nyquist, quality_factor, sampling_rate)

def apply_filter(numerator, denominator, data):
    return filtfilt(numerator, denominator, data)