import numpy as np
import scipy.signal as signal
from scipy.stats import zscore


def bandpass_filter(signal_data, fs, lowcut=20.0, highcut=450.0, order=4):
    """
    Apply a Butterworth bandpass filter to the signal.
    signal_data: 1D numpy array (single channel)
    fs: sampling frequency (Hz)
    lowcut, highcut: cutoff frequencies in Hz
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return filtered


def normalize_signal(signal_data):
    """
    Normalize a 1D signal using z-score normalization.
    """
    return zscore(signal_data)


def segment_signal(signal_data, window_size, step_size):
    """
    Segment a 1D signal into overlapping windows.
    Returns a list of windows.
    """
    segments = []
    for start in range(0, len(signal_data) - window_size + 1, step_size):
        segment = signal_data[start:start+window_size]
        segments.append(segment)
    return segments


def compute_features(segment, fs):
    """
    Compute time-domain features for a given segment.
    Features:
      - Root Mean Square (RMS)
      - Mean Absolute Value (MAV)
      - Zero Crossing (ZC) count (with threshold)
      - Slope Sign Changes (SSC)
      - Waveform Length (WL)
    You can add frequency-domain features as needed.
    """
    # RMS
    rms = np.sqrt(np.mean(segment**2))
    # MAV
    mav = np.mean(np.abs(segment))
    # Zero Crossing: count crossings with a threshold to avoid noise-induced crossings.
    threshold = 0.01  # Adjust threshold as needed
    # Basic version; can add threshold logic if needed.
    zc = np.sum(np.abs(np.diff(np.sign(segment))) > 0)
    # Slope Sign Changes
    diff_signal = np.diff(segment)
    ssc = np.sum(np.diff(np.sign(diff_signal)) != 0)
    # Waveform Length
    wl = np.sum(np.abs(np.diff(segment)))

    # Optionally add frequency domain features, e.g. dominant frequency
    freqs, psd = signal.welch(segment, fs)
    dom_freq = freqs[np.argmax(psd)]

    return [rms, mav, zc, ssc, wl, dom_freq]


def extract_features_from_signal(raw_signal, fs=1000, window_size=256, step_size=128):
    """
    Process a multi-channel signal by filtering, normalizing, segmenting, and then extracting features from each segment.
    For simplicity, this example processes only one channel (e.g., channel 0). For multi-channel, loop over channels or concatenate features.
    Returns:
        feature_matrix: list of feature vectors
    """
    channel_data = raw_signal

    signal1 = channel_data[:len(channel_data)//2]
    signal2 = channel_data[len(channel_data)//2:]

    signal1 = bandpass_filter(signal1, fs)
    signal2 = bandpass_filter(signal2, fs)

    feat1 = compute_features(signal1, fs)
    feat2 = compute_features(signal2, fs)

    feat = np.concatenate((feat1, feat2))

    return feat
