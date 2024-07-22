import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wavfile
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
from scipy.signal import stft
import os
from datetime import datetime

warnings.filterwarnings('ignore', category=wavfile.WavFileWarning)  # Suppress WavFileWarning

# Parse terminal arguments
parser = argparse.ArgumentParser(description='Train LSTM model for audio regression')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train the model')
args = parser.parse_args()

# Paths to clean and distorted audio files
long = False
if long:
    clean_audio_path = './audio32fp/ts9_in.wav'
    distorted_audio_path = './audio32fp/ts9_out_drive=10.wav'
    predicted_audio_path = './audio32fp/predicted_out.wav'
    waveform_path = './audio32fp/waveform.png'
else:
    clean_audio_path = './shortaudio32fp/ts9_in.wav'
    distorted_audio_path = './shortaudio32fp/ts9_out_drive=10.wav'
    predicted_audio_path = './shortaudio32fp/predicted_out.wav'
    waveform_path = './shortaudio32fp/waveform.png'

# Hyperparameters
sequence_length = 150
batch_size = 4096
num_epochs = args.num_epochs
learning_rate = 1e-2
hidden_size = 64  # LSTM hidden size
test_ratio = 0.2

# Custom Dataset class
class Seq2SamDataset(Dataset):
    def __init__(self, clean_path, distorted_path, sequence_length, test_ratio):
        self.sequence_length = sequence_length
        
        # Read audio files
        self.clean_rate, self.clean_audio = wavfile.read(clean_path)
        self.distorted_rate, self.distorted_audio = wavfile.read(distorted_path)
        
        # Normalize audio waveforms
        self.clean_max = np.max(np.abs(self.clean_audio))
        self.distorted_max = np.max(np.abs(self.distorted_audio))
        self.clean_audio = self.normalize_waveform(self.clean_audio)
        self.distorted_audio = self.normalize_waveform(self.distorted_audio)
        
        # Ensure both audio files have the same length (trim if necessary)
        min_length = min(len(self.clean_audio), len(self.distorted_audio))
        self.clean_audio = self.clean_audio[:min_length]
        self.distorted_audio = self.distorted_audio[:min_length]
        
        # Calculate number of samples
        self.num_samples = len(self.clean_audio) - sequence_length

        # Split dataset into training and testing sets
        num_test_samples = int(self.num_samples * test_ratio)
        num_train_samples = self.num_samples - num_test_samples
        
        self.test_start_idx = np.random.randint(1, self.num_samples - num_test_samples - 1)
        self.test_end_idx = self.test_start_idx + num_test_samples
        self.test_indices = list(range(self.test_start_idx, self.test_end_idx))
        self.train_indices = list(range(0, self.test_start_idx)) + list(range(self.test_end_idx, self.num_samples))

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get a sequence of clean audio samples and its corresponding distorted audio sample
        clean_seq = self.clean_audio[idx:idx+self.sequence_length]
        distorted_sample = self.distorted_audio[idx+self.sequence_length]
        
        # Convert to torch tensors
        clean_seq = torch.tensor(clean_seq, dtype=torch.float32)
        distorted_sample = torch.tensor(distorted_sample, dtype=torch.float32)
        
        return clean_seq, distorted_sample
    
    def normalize_waveform(self, waveform):
        # Normalize waveform to range [-1, 1]
        waveform = waveform.astype(np.float32)
        max_val = np.max(np.abs(waveform))
        return waveform / max_val
    
    def get_train_test_datasets(self):
        train_dataset = torch.utils.data.Subset(self, self.train_indices)
        test_dataset = torch.utils.data.Subset(self, self.test_indices)
        return train_dataset, test_dataset
    
    def get_targets_between_indices(self, start_idx, end_idx):
        if start_idx < 0 or end_idx >= self.num_samples or start_idx >= end_idx:
            raise ValueError("Invalid start or end index.")
        
        targets = [self.distorted_audio[idx+self.sequence_length] for idx in range(start_idx, end_idx)]
        return np.array(targets, dtype=np.float32)

# Define the LSTM model
class pedaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(pedaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(0))  # LSTM input shape: (seq_len, batch, input_size)
        out = self.linear(out[-1])  # Take the last output of the sequence
        return out.squeeze()

# Function to compute various metrics
def compute_metrics(predicted, target, rate):
    mse = mean_squared_error(predicted, target)
    rmse = np.sqrt(mse)
    snr = 10 * np.log10(np.sum(target**2) / np.sum((target - predicted)**2))
    psnr = 10 * np.log10(np.max(target**2) / mse)
    pearson_corr = np.corrcoef(target, predicted)[0, 1]
    lsd = log_spectral_distance(target, predicted, rate)
    return mse, rmse, snr, psnr, pearson_corr, lsd

def log_spectral_distance(audio1, audio2, sr, frame_size=2048, hop_size=512):
    # Perform STFT on both audio signals
    _, _, Zxx1 = stft(audio1, fs=sr, nperseg=frame_size, noverlap=frame_size-hop_size)
    _, _, Zxx2 = stft(audio2, fs=sr, nperseg=frame_size, noverlap=frame_size-hop_size)
    
    # Convert the STFT results to log-magnitude spectra
    log_S1 = np.log(np.abs(Zxx1) + 1e-10)
    log_S2 = np.log(np.abs(Zxx2) + 1e-10)
    
    # Compute the Log-Spectral Distance
    lsd = np.sqrt(np.mean((log_S1 - log_S2) ** 2))
    return lsd

# Instantiate dataset and dataloader
dataset = Seq2SamDataset(clean_audio_path, distorted_audio_path, sequence_length, test_ratio)
train_dataset, test_dataset = dataset.get_train_test_datasets()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model, loss function, and optimizer
model = pedaLSTM(sequence_length, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
metrics_file = f'./metrics/test/training_metrics_{timestamp}.txt'
if os.path.exists(metrics_file):
    os.remove(metrics_file)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (clean_seq, distorted_sample) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(clean_seq)
        
        # Calculate loss
        loss = criterion(output, distorted_sample)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
    # Evaluate on the test set
    model.eval()
    predicted_distorted_audio = []
    target_distorted_audio = []
    with torch.no_grad():
        for clean_seq, distorted_sample in test_loader:
            output = model(clean_seq)
            predicted_distorted_audio.extend(output.numpy())
            target_distorted_audio.extend(distorted_sample.numpy())
    
    predicted_distorted_audio = np.array(predicted_distorted_audio)
    target_distorted_audio = np.array(target_distorted_audio)

    predicted_distorted_audio = predicted_distorted_audio * dataset.distorted_max
    target_distorted_audio = target_distorted_audio * dataset.distorted_max

    mse, rmse, snr, psnr, pearson_corr, lsd = compute_metrics(predicted_distorted_audio, target_distorted_audio, dataset.distorted_rate)
    
    with open(metrics_file, 'a') as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"SNR: {snr} dB\n")
        f.write(f"PSNR: {psnr} dB\n")
        f.write(f"Pearson Correlation Coefficient: {pearson_corr}\n")
        f.write(f"LSD: {lsd}\n\n")
    
    print(f"Epoch [{epoch+1}/{num_epochs}], MSE: {mse}, RMSE: {rmse}, SNR: {snr}, PSNR: {psnr}, Pearson Corr: {pearson_corr}, LSD: {lsd}")

print(f'Training complete. Metrics saved to {metrics_file}.')
