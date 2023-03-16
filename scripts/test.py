import torch
from torch.utils.data import DataLoader

from dataset_functions import LSTMBirdCallDatasetWaveform, LSTMBirdCallDatasetSpectrogram
from models import BasicModel
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# For MacOSX M1 GPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    # this ensures that the current MacOS version is at least 12.3+
    # and ensures that the current current PyTorch installation was built with MPS activated.
    device = torch.device("mps")
    logging.info("Using MPS GPU")
# cuda for NVIDIA GPU
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
    logging.info("Using CUDA GPU")
else:
    device = torch.device('cpu')
    logging.info("Using CPU")


def main():
    # Basic function test
    ds = LSTMBirdCallDatasetWaveform('/Users/justinbutler/Documents/Coding_Projects/BIRDCLEF/data/birdclef-2023/train_metadata.csv')
    train_dataloader = DataLoader(ds, batch_size=2, shuffle=True)
    train_call, train_label = next(iter(train_dataloader))
    print(train_call.shape)
    model = BasicModel().to(device)
    test = model(train_call)

    return


if __name__ == '__main__':
    main()

