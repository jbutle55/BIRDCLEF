"""
A class to handle the data wrangling for the image transformer method of the BirdCLEF 2023 Kaggle Challenge.

This class takes sound clips, splits them to an appropriate length, and prepares them to be passed to a transformer neural network.

Authors:
Justin Butler & Garrett Toews

"""
import os
from PIL import Image
import numpy as np
import pandas as pd
import librosa
import logging
import torch
from LSTM.lstm_dataset import LSTMBirdCallDataset
from torch.utils.data import Dataset
from train_2023_label_map import train_label_map_23
from dataset_functions import apply_bandpass, apply_gaussian_noise, frame_audio, spectrogram, pad_audio_seconds
from utils import check_device

device = check_device()

class TransformerDataset(LSTMBirdCallDataset):
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

        if banded_wave.size < self.minimum_duration_frames:
            banded_wave = pad_audio_seconds(banded_wave, self.minimum_duration_frames - banded_wave.size,
                                            sample_rate_hz=sample_rate_hz)

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

    def sound_to_image(self, clip):
        '''
        Convert the image file into a power spectral density
        '''
        spectogram = self.__getitem__(self, idx=12).first()
    
    def resize_image(self, img):
        '''
        Resize image into array of 16x16 images
        '''
        old_image=Image.open(img)
        #@TODO: Pad so its square
        new_image = old_image.thumbnail((16,16))
        pass

    def flatten_image(self, img):
        '''
        Flatten the 16x16 images into vector
        '''
        pass

    def encode(self, vector):
        '''
        Prepare the vector for the Transformer
        @TODO: Figure out what the hell this means
        '''
        pass

    def convert(self, sound_clips):
        '''
        Driving function that passes each clip through the preparatory methods.
        '''
        converted_data=[]
        for clip in sound_clips:
            image = self.sound_to_image(clip)
            tiny_image=self.resize_image(image)
            image_vector=self.flatten_image(tiny_image)
            converted_data.append(self.encode(image_vector))
        return converted_data

def main():
    '''
    Main to handle the operation of the code.
    '''
    pass

if __name__ == "main":
    main()