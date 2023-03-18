from torch import nn
from torch.utils.data import DataLoader
from scripts.LSTM.lstm_model import LSTMCell
from scripts.LSTM.lstm_dataset import LSTMBirdCallDatasetWaveform
from scripts.utils import check_device
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
device = check_device()


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 35, kernel_size=80, stride=16)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        return x


def main():
    # Basic function test
    ds = LSTMBirdCallDatasetWaveform('data/birdclef-2023/train_metadata.csv')
    train_dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    train_call, train_label = next(iter(train_dataloader))
    print(train_call.shape)
    model = LSTMCell(*train_call.shape[1:]).to(device)
    print(model)
    test = model(train_call)

    return


if __name__ == '__main__':
    main()

