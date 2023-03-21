"""
Data handling functions.

Authors:
Justin Butler & Garrett Toews
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, sosfilt, windows
from scripts.utils import check_device
device = check_device()


def spectrogram(signal, sample_rate_hz, frame_overlap: float = 0.5, frame_duration_ms: float = 20.0, hann_weighting: bool = False):
    """
    A function to compute a spectrogram from an input signal.
    The spectrogram is computed by calculating individual FFTs in a sliding window fashion across the input signal.
    A spectrogram is a 2-d array of frequency components at increasing time steps with the magnitude of each frequency
    component at time t represented by the points value.

    Args:
        signal: The input signal used to compute the spectrogram.
        sample_rate_hz: The sample rate of the input signal in Hz.
        frame_overlap: The amount of overlap between FFT windows. Must be between 0 and 1.
        frame_duration_ms: The individual FFT frame duration in milliseconds.
        hann_weighting: Boolean whether to app;y Hann weighting. Defaults to False.

    Returns: A tuple containing: (the spectrogram with shape [Freq, Time], the sample rate of the signal in Hz, the computed width of a single FFT window)

    """
    if not 0 <= frame_overlap <= 1.0:
        raise ValueError("Frame overlap should be between 0 and 1")

    # Break signal into frames and calculate FT for each frame.
    # Frames should overlap so freq. info is not lost

    # Calculate frame size and stride size
    frame_size = int(sample_rate_hz * frame_duration_ms * 0.001)
    frame_stride_size = int(frame_size * frame_overlap)

    # Extract windows
    # -----------------
    # Where to remove the X samples lost from window extraction? Start? End? Split?
    # For now, remove X samples from end
    to_truncate_size = signal.size % frame_stride_size
    truncated_samples = signal[:(signal.size - to_truncate_size)]

    window_view = np.lib.stride_tricks.sliding_window_view(truncated_samples, frame_size)
    strided_windows = window_view[::frame_stride_size]
    single_window_size = strided_windows[0].size

    if hann_weighting:
        hann_weights = windows.hann(single_window_size)
        strided_windows = strided_windows * hann_weights

    # Apply FFT to all windows at once
    fft_windows = fft(strided_windows)
    fft_windows_abs = 2.0 / frame_size * np.abs(fft_windows[:, :frame_size // 2]).T  # Has shape [Freq, Time]
    dynamic_spectrogram = (10 * np.log10(fft_windows_abs))  # Increase dynamic range

    return dynamic_spectrogram, sample_rate_hz, single_window_size


def mel_spectrogram(spectrogram, sample_rate_hz, num_mel_filters=40, low_freq_hz=0, high_freq_hz=None):
    """

    Args:
        spectrogram:
        sample_rate_hz:
        num_mel_filters:
        low_freq_hz:
        high_freq_hz:

    Returns:

    """
    norm_filter_bank, filter_bank = mel_filter(num_freq_components=num_mel_filters,
                                               num_windows=spectrogram.shape[1],
                                               samplerate=sample_rate_hz,
                                               low_freq=low_freq_hz,
                                               high_freq=high_freq_hz)

    mel_spec = np.transpose(filter_bank.T).dot(spectrogram.T)
    # mel_spec = np.transpose(norm_filter_bank).dot(spectrogram.T)

    return mel_spec.T


def mel_filter(num_windows, num_freq_components, samplerate=16000, low_freq=0, high_freq=None):
    """

    Args:
        num_windows:
        num_freq_components:
        samplerate:
        low_freq:
        high_freq:

    Returns:

    """
    high_freq = high_freq if (high_freq and high_freq < samplerate / 2) else (samplerate / 2)
    min_mel = hz2mel(low_freq)
    max_mel = hz2mel(high_freq)
    mel_points = np.linspace(min_mel, max_mel, num_freq_components + 2)
    hz_points = mel2hz(mel_points)

    fft_bin = np.floor((num_windows + 1) * hz_points / samplerate)
    filter_bank = np.zeros([num_freq_components, num_windows])  # Each row is a single filter

    for j_filter in range(0, num_freq_components):
        for i_left_freq in range(int(fft_bin[j_filter]), int(fft_bin[j_filter+1])):
            filter_bank[j_filter, i_left_freq] = (i_left_freq - fft_bin[j_filter]) / (fft_bin[j_filter + 1] - fft_bin[j_filter])
        for i_right_freq in range(int(fft_bin[j_filter+1]), int(fft_bin[j_filter+2])):
            filter_bank[j_filter, i_right_freq] = (fft_bin[j_filter + 2] - i_right_freq) / (fft_bin[j_filter + 2] - fft_bin[j_filter + 1])

    norm_filter_bank = filter_bank.T / filter_bank.sum(axis=1)  # Normalize

    print(f'filter bank shape: {filter_bank.shape}')
    plt.plot(filter_bank[39])
    plt.show()

    plt.imshow(filter_bank, aspect='auto')
    plt.show()

    return norm_filter_bank, filter_bank


def hz2mel(freq_hz):
    """
        Function to calculate a mel from Hz.
        Args:
            freq_hz: The input frequency value in Hz to be converted to mel.

        Returns: The converted mel frequency.

        """
    return 2595 * np.log10(1 + (freq_hz / 700))


def mel2hz(mel):
    """
    Function to calculate a frequency in Hz from mel.
    Args:
        mel: The input mel value to be converted to Hz.

    Returns: The converted frequency in Hz.

    """
    return 700 * (np.power(10, mel / 2595) - 1)


def display_spectrogram(spec, sample_rate_hz: int, window_size: int, title_addon: str = None):
    """
    A function to plot a spectrogram. Can be used in conjunction with the spectrogram function in the manner:
    display_spectrogram(*spectrogram(signal, sample_rate_hz), title_addon='')

    Args:
        spec: The spectrogram with shape [Freq, Time].
        sample_rate_hz: The sample rate of the signal used to compute the spectrogram in Hz.
        window_size: The width of a single window used to compute the spectrogram.
        title_addon: Additional string to append to the spectrogram title during plotting.

    """
    plt.imshow(spec, aspect="auto", origin="lower")
    if title_addon:
        spec_title = f'Spectrogram - {title_addon}'
    else:
        spec_title = 'Spectrogram'
    plt.title(spec_title)

    # create y limit ticks and labels
    num_yticks = 10
    ks = np.linspace(0, spec.shape[0], num_yticks)
    ks_hz = ks * sample_rate_hz / window_size
    freq_hz = [int(i) for i in ks_hz]
    plt.yticks(ks, freq_hz)
    plt.ylabel("Frequency (Hz)")
    plt.show()


def apply_bandpass(signal: np.ndarray, sample_rate_hz, lower_freq_hz: int = 0, upper_freq_hz: int = 500000) -> np.ndarray:
    """
    A function to apply a bandpass filter to an input signal.
    The function can technically be used as both a low-pass and high-pass filter by setting the upper freq. limit to a
    sufficiently high number or the lower freq. limit to 0 Hz respectively.
    The upper frequency limit cannot exceed half the sampling rate (Nyquist Limit).

    Args:
        signal: The input signal to be filtered.
        sample_rate_hz: The sample rate of the input signal in Hz.
        lower_freq_hz: The lower frequency limit of the bandpass. All freq. below the limit will be removed.
        upper_freq_hz: The upper frequency limit of the bandpass. All freq. above the limit will be removed.

    Returns: The filtered signal in the form of a numpy array.

    """
    # Upper freq. limit must meet Nyquist freq.
    if upper_freq_hz >= (sample_rate_hz/2):
        logging.info('Setting upper limit of bandpass filter to Fs/2.')
        upper_freq_hz = sample_rate_hz/2 - 1

    sos_bandpass = butter(N=10, Wn=[lower_freq_hz, upper_freq_hz], btype='band', fs=sample_rate_hz, output='sos')
    banded_signal = sosfilt(sos_bandpass, signal)

    return banded_signal


def frame_audio(signal: np.ndarray, sample_rate_hz: float, frame_duration_seconds: float = 5.0) -> np.ndarray:
    """
    Function to split an audio signal into a numpy array of short clips.

    Args:
        signal: The input signal to be split.
        sample_rate_hz: The sample rate of the input signal in Hz.
        frame_duration_seconds: The duration of each framed clip in seconds.

    Returns: A numpy array of framed clips of the input signal with each clip having a duration of frame_duration_seconds.

    """
    num_segments_per_frame = int(np.floor(frame_duration_seconds * sample_rate_hz))
    windowed_signal = np.lib.stride_tricks.sliding_window_view(signal, num_segments_per_frame)
    framed_audio = windowed_signal[::num_segments_per_frame]

    return framed_audio


def apply_gaussian_noise(signal: np.ndarray) -> np.ndarray:
    """
    A function to apply random Gaussian noise to a signal.

    Args:
        signal: The input signal to have Gaussian noise applied.

    Returns: The output signal with Gaussian noise overlaid on the input audio signal.

    """
    gauss_noise = np.random.randn(*signal.shape).astype(np.float32)
    signal = signal + gauss_noise
    return signal


def add_background_birds(signal: np.ndarray) -> np.ndarray:
    """
    Useful function for fine-tuning a trained model. Adding low volume and clipped random birdcalls as background
    noise might assist the model with detecting correct calls during inference.

    Args:
        signal:

    Returns:

    """
    # TODO Create function

    return signal

