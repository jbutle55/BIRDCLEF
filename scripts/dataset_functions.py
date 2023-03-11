import torch
from torch.utils.data import Dataset
device = torch.device("mps")


import os
import numpy as np
import pandas as pd
import librosa
import logging

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


if __name__ == '__main__':
    ds = BirdsDataset('/Users/justinbutler/Documents/Coding_Projects/BIRDCLEF/data/train_metadata.csv')
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    train_call, train_filename, train_label = next(iter(train_dataloader))
