import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = "models"


class SimpleSeriesDataset(Dataset):

    def __init__(self, series, maxlen=None, normalize=None):
        self.series = series
        self.normalize = normalize
        self.maxlen = maxlen
        self.scaler = StandardScaler() if normalize == 'zscore' else None

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        series = self.series.iloc[idx]
        if self.maxlen is not None and len(series) > self.maxlen:
            series = series[:self.maxlen]
        if self.normalize:
            series = self.scaler.transform(series.values.reshape(-1, 1))
        return torch.Tensor(series)


class TimeNet(nn.Module):

    def __init__(self, size, num_layers, dropout=0.0):
        super(TimeNet, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_layers = []
        encoder_layers.append(nn.GRU(1, self.size, num_layers=1, batch_first=True))
        for i in range(2, self.num_layers + 1):
            encoder_layers.append(nn.GRU(self.size, self.size, num_layers=1, batch_first=True))
        return nn.Sequential(*encoder_layers)

    def build_decoder(self):
        decoder_layers = []
        decoder_layers.append(nn.GRU(self.size, self.size, num_layers=1, batch_first=True))
        for i in range(2, self.num_layers + 1):
            decoder_layers.append(nn.GRU(self.size, self.size, num_layers=1, batch_first=True))
        decoder_layers.append(nn.Linear(self.size, 1))
        return nn.Sequential(*decoder_layers)

    def forward(self, x):
        # Encoder
        encode, _ = self.encoder(x)

        # Decoder
        decode, _ = self.decoder(torch.flip(encode, [1]))  # Reverse the encoded sequence
        return decode


def train(model, train_loader, optimizer, criterion, lr_scheduler=None, epochs=10, early_stop=5):
    run = model.get_run_id()
    log_dir = os.path.join(OUTPUT_DIR, run)
    weights_path = os.path.join(log_dir, 'weights.pt')

    loaded = False
    if os.path.exists(weights_path):
        print("Loading {}...".format(weights_path))
        model.load_state_dict(torch.load(weights_path))
        loaded = True

    if (not loaded):
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir)

    weights_path = os.path.join(log_dir, 'weights.pt')

    if lr_scheduler is not None:
        optimizer = optim.lr_scheduler(optimizer, mode='min', patience=early_stop, verbose=True, threshold=0.0001,
                                        threshold_mode='rel')

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch.unsqueeze(2))
            loss = criterion(outputs, batch.unsqueeze(2))
            loss.backward()
            optimizer.step()

        # Save model weights
        torch.save(model.state_dict(), weights_path)

    return log_dir

