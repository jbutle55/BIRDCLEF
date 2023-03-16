import os
import numpy as np
import pandas as pd
import librosa
import logging
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, sosfilt, windows
import torch
from torch.utils.data import Dataset
device = torch.device("mps")


class LSTMBirdCallDataset(Dataset):
    """
    Base bird call audio dataset.
    """
    def __init__(self, dataset_dir: str, data_year='2023', eval_mode=False, train_audio_duration=5.0, sample_rate_hz=48000):
        self.dataset_csv = dataset_dir
        self.audio_loc = f'data/birdclef-{data_year}/train_audio'
        self.eval_mode = eval_mode
        if not self.eval_mode:
            self.df = pd.read_csv(dataset_dir, usecols=["primary_label", "secondary_labels", "common_name", "filename"])
            self.labels = list(set(self.df.common_name.unique()))
            logging.info(f"Number of labels in dataset: {len(self.labels)}")

        self.audio_duration_seconds = train_audio_duration if not self.eval_mode else 5
        self.sample_rate_hz = sample_rate_hz

    def __getitem__(self, item):
        return

    def __len__(self):
        return len(self.labels)


class LSTMBirdCallDatasetWaveform(LSTMBirdCallDataset):
    def __getitem__(self, idx):
        if not self.eval_mode:
            row = self.df.iloc[idx]
            filename = row['filename']
            primary_label = row['primary_label']
            common_name = row["common_name"]
            audio_path = os.path.join(self.audio_loc, filename)

        else:
            audio_path = os.path.join(self.audio_loc, idx)

        # All samples in a batch need to be an identical size, therefore use fixes sample rate and duration
        wav, sample_rate_hz = librosa.load(audio_path, sr=self.sample_rate_hz, duration=self.audio_duration_seconds)

        # TODO Apply bandpass and other transforms
        banded_wave = apply_bandpass(signal=wav, sample_rate=sample_rate_hz, lower_freq_hz=150, upper_freq_hz=15000)
        banded_wave = np.float32(banded_wave)

        # Apply Gaussian noise
        gaussian_wave = apply_gaussian_noise(banded_wave)

        if self.eval_mode:
            # Split test waveform into 5-second segments for inference
            framed_waveforms = frame_audio(signal=gaussian_wave, sample_rate_hz=sample_rate_hz, frame_duration_seconds=5.0)
            framed_waves_tensor = torch.tensor(framed_waveforms, device=device)
            return framed_waves_tensor

        waveform = torch.tensor(gaussian_wave, device=device).unsqueeze(0)

        return waveform, filename, common_name


class LSTMBirdCallDatasetSpectrogram(LSTMBirdCallDataset):
    def __getitem__(self, idx):
        if not self.eval_mode:
            row = self.df.iloc[idx]
            filename = row['filename']
            primary_label = row['primary_label']
            common_name = row["common_name"]
            audio_path = os.path.join(self.audio_loc, filename)

        else:
            audio_path = os.path.join(self.audio_loc, idx)

        # All samples in a batch need to be an identical size, therefore use fixes sample rate and duration
        wav, sample_rate_hz = librosa.load(audio_path, sr=self.sample_rate_hz, duration=self.audio_duration_seconds)

        # TODO Apply bandpass and other transforms
        banded_wave = apply_bandpass(signal=wav, sample_rate=sample_rate_hz, lower_freq_hz=150, upper_freq_hz=15000)
        banded_wave = np.float32(banded_wave)

        # Apply Gaussian noise
        gaussian_wave = apply_gaussian_noise(banded_wave)

        if self.eval_mode:
            # Split test waveform into 5-second segments for inference
            framed_waveforms = frame_audio(signal=gaussian_wave, sample_rate_hz=sample_rate_hz, frame_duration_seconds=5.0)

            framed_spects = []
            for frame in framed_waveforms:
                frame_spect, _, _ = spectrogram(frame, sample_rate_hz)
                framed_spects.append(frame_spect)

            framed_spects_tensor = torch.FloatTensor(np.array(framed_spects), device=device)
            return framed_spects_tensor  # Returns shape of [Num of frames, ]

        spect, _, _ = spectrogram(gaussian_wave, sample_rate_hz)
        spect = torch.tensor(spect, device=device)

        return spect, filename, common_name


def spectrogram(signal, sample_rate_hz, frame_overlap=0.5, frame_duration_ms=20, hann_weighting=False):
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
    fft_windows_abs = 2.0 / frame_size * np.abs(fft_windows[:, :frame_size // 2]).T
    dynamic_spectrogram = (10 * np.log10(fft_windows_abs))  # Increase dynamic range

    return dynamic_spectrogram, sample_rate_hz, single_window_size


def mel_spectrogram(spectrogram, sample_rate_hz, num_mel_filters=40, low_freq_hz=0, high_freq_hz=None):
    norm_filter_bank, filter_bank = mel_filter(num_freq_components=num_mel_filters,
                                               num_windows=spectrogram.shape[1],
                                               samplerate=sample_rate_hz,
                                               low_freq=low_freq_hz,
                                               high_freq=high_freq_hz)

    mel_spec = np.transpose(filter_bank.T).dot(spectrogram.T)
    # mel_spec = np.transpose(norm_filter_bank).dot(spectrogram.T)

    return mel_spec.T


def mel_filter(num_windows, num_freq_components, samplerate=16000, low_freq=0, high_freq=None):
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
    return 2595 * np.log10(1 + (freq_hz / 700))


def mel2hz(mel):
    return 700 * (np.power(10, mel / 2595) - 1)


def display_spectrogram(spec, sample_rate_hz, window_size, title_addon=None):
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

    return


def apply_bandpass(signal, sample_rate, lower_freq_hz=0, upper_freq_hz=500000):
    # Upper freq. limit must meet Nyquist freq.
    if upper_freq_hz >= (sample_rate/2):
        logging.info('Setting upper limit of bandpass filter to Fs/2.')
        upper_freq_hz = sample_rate/2 - 1

    sos_bandpass = butter(N=10, Wn=[lower_freq_hz, upper_freq_hz], btype='band', fs=sample_rate, output='sos')
    banded_signal = sosfilt(sos_bandpass, signal)

    return banded_signal


def frame_audio(signal: np.ndarray, sample_rate_hz: float, frame_duration_seconds: float = 5.0) -> np.ndarray:
    num_segs_per_frame = int(np.floor(frame_duration_seconds * (signal.size / sample_rate_hz)))
    windowed_signal = np.lib.stride_tricks.sliding_window_view(signal, num_segs_per_frame)
    framed_audio = windowed_signal[::num_segs_per_frame]

    return framed_audio


def apply_gaussian_noise(signal: np.ndarray):
    gauss_noise = np.random.randn(*signal.shape).astype(np.float32)
    signal = signal + gauss_noise
    return signal


def add_background_birds(signal: np.ndarray):
    """
    Useful function for fine-tuning a trained model. Adding low volume and clipped random birdcalls as background
    noise might assist the model with detecting correct calls during inference.
    :param signal:
    :return:
    """
    # TODO Create function

    return signal

