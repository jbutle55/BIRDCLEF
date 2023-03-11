import torch
from torch.utils.data import DataLoader

from dataset_functions import BirdsDataset
from models import BasicModel
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# For MacOSX M1 GPU
# cuda for NVIDIA GPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    # this ensures that the current MacOS version is at least 12.3+
    # and ensures that the current current PyTorch installation was built with MPS activated.
    device = torch.device("mps")
    logging.info("Using MPS GPU")
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
    logging.info("Using CUDA GPU")
else:
    device = torch.device('cpu')
    logging.info("Using CPU")


def main():
    # Basic function test
    ds = BirdsDataset('/Users/justinbutler/Documents/Coding_Projects/BIRDCLEF/data/train_metadata.csv')
    train_dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    train_call, train_filename, train_label = next(iter(train_dataloader))
    model = BasicModel().to(device)
    print(model)
    test = model(train_call)

    return


if __name__ == '__main__':
    main()

