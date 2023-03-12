import os
import numpy as np
import pandas as pd
import librosa
import logging
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfilt, windows
import torch
from torch.utils.data import Dataset
device = torch.device("mps")


class BirdsDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_csv = dataset_dir
        self.audio_loc = '/Users/justinbutler/Documents/Coding_Projects/BIRDCLEF/data/train_audio'
        birds_ds = pd.read_csv(dataset_dir, usecols=["primary_label", "secondary_labels", "common_name", "filename"])
        logging.info(birds_ds.shape)
        self.df = birds_ds

        self.labels = list(set(birds_ds.common_name.unique()))
        logging.info(f"Number of Labels: {len(self.labels)}")

        self.duration = 15
        self.sample_rate = 32000

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        primary_label = row['primary_label']
        common_name = row["common_name"]
        audio_path = os.path.join(self.audio_loc, filename)

        # TODO Audio calls (wav) all need to be identical in length
        try:
            wav, _ = librosa.load(audio_path, sr=None, duration=self.duration)
            np.expand_dims(wav, 0)
            wav = torch.tensor(wav, device=device)
            wav.unsqueeze(0)
        except:
            logging.log(logging.WARNING, f"Failed loading of call with path {audio_path}")

        #return {"call": torch.tensor(wav), "filename": filename, "common_name": common_name}
        return (wav, filename, common_name)

    def __len__(self):
        return len(self.labels)


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

