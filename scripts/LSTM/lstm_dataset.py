"""
Dataset classes for LSTM models.

Authors:
Justin Butler & Garrett Toews
"""

import os
import numpy as np
import pandas as pd
import librosa
import logging
import torch
from torch.utils.data import Dataset
from scripts.train_2023_label_map import train_label_map_23
from scripts.dataset_functions import apply_bandpass, apply_gaussian_noise, frame_audio, spectrogram
from scripts.utils import check_device
device = check_device()


class LSTMBirdCallDataset(Dataset):
    """
    Base bird call audio dataset.
    """
    def __init__(self, dataset_dir: str, data_year='2023', eval_mode: bool = False, train_audio_duration=2.0, sample_rate_hz=48000, gaussian_aug: bool = False):
        self.dataset_csv = dataset_dir
        self.audio_loc = f'data/birdclef-{data_year}/train_audio'
        self.eval_mode = eval_mode
        if not self.eval_mode:
            self.df = pd.read_csv(dataset_dir, usecols=["primary_label", "secondary_labels", "common_name", "filename"])
            self.labels = list(set(self.df.common_name.unique()))
            logging.info(f"Number of labels in dataset: {len(self.labels)}")

        self.audio_duration_seconds = train_audio_duration if not self.eval_mode else 5
        self.sample_rate_hz = sample_rate_hz
        self.gaussian_augmentation = gaussian_aug
        self.rng = np.random.default_rng()

    def __getitem__(self, item):
        return

    def __len__(self):
        return len(self.labels)


class LSTMBirdCallDatasetWaveform(LSTMBirdCallDataset):
    """
    Bird call Dataset class for the LSTM model trained using audio waveforms.
    """
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
        print(wav.shape)
        print(f'sr: {sample_rate_hz}')

        # Apply bandpass and other transforms
        banded_wave = apply_bandpass(signal=wav, sample_rate_hz=sample_rate_hz, lower_freq_hz=150, upper_freq_hz=15000)
        banded_wave = np.float32(banded_wave)

        # Apply Gaussian noise
        if self.gaussian_augmentation:
            banded_wave = apply_gaussian_noise(banded_wave)

        # Split waveform into 5-second segments for inference
        framed_waveforms = frame_audio(signal=banded_wave, sample_rate_hz=sample_rate_hz, frame_duration_seconds=5.0)

        # If inference: perform inference over all 5 second clips of test audio waveform
        if self.eval_mode:
            # Split test waveform into 5-second segments for inference
            framed_waves_tensor = torch.tensor(framed_waveforms, device=device)
            return framed_waves_tensor

        # If training: Randomly select one 5 second frame of the training audio each step
        # TODO This implementation is less random than selecting a single 5s frame randomly from the original waveform
        training_frame_idx = self.rng.integers(0, len(framed_waveforms))
        five_sec_frame = framed_waveforms[training_frame_idx]

        waveform = torch.tensor(five_sec_frame, device=device).unsqueeze(0)

        return waveform, train_label_map_23[primary_label]


class LSTMBirdCallDatasetSpectrogram(LSTMBirdCallDataset):
    """
    Bird call Dataset class for the LSTM model trained using spectrograms from audio waveforms.
    """
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

        # Apply bandpass and other transforms
        banded_wave = apply_bandpass(signal=wav, sample_rate_hz=sample_rate_hz, lower_freq_hz=150, upper_freq_hz=15000)
        banded_wave = np.float32(banded_wave)

        # Apply Gaussian noise
        if self.gaussian_augmentation:
            banded_wave = apply_gaussian_noise(banded_wave)

        # Split waveform into 5-second segments for inference
        framed_waveforms = frame_audio(signal=banded_wave, sample_rate_hz=sample_rate_hz, frame_duration_seconds=5.0)

        # If inference: perform inference over all 5 second clips of test audio waveform
        if self.eval_mode:
            framed_spects = []
            for frame in framed_waveforms:
                frame_spect, _, _ = spectrogram(frame, sample_rate_hz)
                framed_spects.append(frame_spect)

            framed_spects_tensor = torch.FloatTensor(np.array(framed_spects), device=device)
            return framed_spects_tensor  # Returns shape of [Num of frames, ]

        # If training: Randomly select one 5 second frame of the training audio each step
        # TODO This implementation is less random than selecting a single 5s frame randomly from the original waveform
        # TODO This improved randomness could be solved by not slicing the framed windows in frame_audio()
        training_frame_idx = self.rng.integers(0, len(framed_waveforms))
        five_sec_frame = banded_wave[training_frame_idx]

        spect, _, _ = spectrogram(five_sec_frame, sample_rate_hz)
        spect = torch.tensor(spect, device=device)  # Tensor shape [Freq, Time]

        return spect, train_label_map_23[primary_label]
