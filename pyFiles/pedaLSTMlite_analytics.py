import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from scipy.fftpack import fft
from scipy.signal import stft

warnings.filterwarnings('ignore', category=wavfile.WavFileWarning)  # Suppress WavFileWarning

# Paths to clean and distorted audio files
long = False
if (long):
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
num_epochs = 1
learning_rate = 1e-2
hidden_size = 64  # LSTM hidden size
test_ratio = 0.2

# Custom Dataset class
class Seq2SamDataset(Dataset):
    def __init__(self, clean_path, distorted_path, sequence_length, test_ratio):
        self.sequence_length = sequence_length
        
        # Read audio files
        self.clean_rate, self.clean_audio = wavfile.read(clean_path)
        #wavfile.write('./shortaudio32fp/predicted_out.wav', self.clean_rate, self.clean_audio)
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
        #print(test_start_idx)
        self.test_indices = list(range(self.test_start_idx, self.test_end_idx))
        self.train_indices = list(range(0, self.test_start_idx)) + list(range(self.test_end_idx, self.num_samples))

        #self.train_indices = list(range(num_train_samples))
        #self.test_indices = list(range(num_train_samples, self.num_samples))
    
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
        waveform=waveform.astype(np.float32)
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

# Instantiate dataset and dataloader
dataset = Seq2SamDataset(clean_audio_path, distorted_audio_path, sequence_length, test_ratio)
train_dataset, test_dataset = dataset.get_train_test_datasets()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# Instantiate the model, loss function, and optimizer
model = pedaLSTM(sequence_length, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (clean_seq, distorted_sample) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(clean_seq)
        
        # Calculate loss
        loss = criterion(output, distorted_sample)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print loss
        if False:#batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}')

# Evaluate on entire dataset and plot results
model.eval()
predicted_distorted_audio = []
with torch.no_grad():
    for idx in range(len(test_dataset)):
        clean_seq, distorted_sample = test_dataset[idx]
        output = model(clean_seq)
        predicted_distorted_audio.append(output.item())
        if False:#idx % 100000 == 0:
            print(f"Sample {idx}/{len(test_dataset)}: Predicted Distorted Audio: {output.item()}, Actual Distorted Audio: {distorted_sample.item()}")

# Convert predicted_distorted_audio to numpy array
predicted_distorted_audio = np.array(predicted_distorted_audio)
# Scale back to original range
predicted_distorted_audio = predicted_distorted_audio.astype(np.float32)
predicted_distorted_audio = predicted_distorted_audio * dataset.distorted_max
target_distorted_audio = dataset.get_targets_between_indices(dataset.test_start_idx, dataset.test_end_idx) * dataset.distorted_max
#print("Predicted size: ", predicted_distorted_audio.size)
#print("Target size: ", target_distorted_audio.size)
#wavfile.write(predicted_audio_path, dataset.distorted_rate, predicted_distorted_audio)
mse = mean_squared_error(predicted_distorted_audio, target_distorted_audio)
rmse = np.sqrt(mse)
snr = 10*np.log10(np.sum(target_distorted_audio**2) / np.sum((target_distorted_audio-predicted_distorted_audio)**2))
psnr = 10 * np.log10(np.max(target_distorted_audio**2) / mse)
pearson_corr = np.corrcoef(target_distorted_audio, predicted_distorted_audio)[0, 1]
#ssim = librosa.segment.recurrence_matrix(target_distorted_audio, predicted_distorted_audio)

def log_spectral_distance(audio1, audio2, sr, frame_size=2048, hop_size=512):
    # Perform STFT on both audio signals
    f1, t1, Zxx1 = stft(audio1, fs=sr, nperseg=frame_size, noverlap=frame_size-hop_size)
    f2, t2, Zxx2 = stft(audio2, fs=sr, nperseg=frame_size, noverlap=frame_size-hop_size)
    
    # Convert the STFT results to log-magnitude spectra
    log_S1 = np.log(np.abs(Zxx1) + 1e-10)
    log_S2 = np.log(np.abs(Zxx2) + 1e-10)
    
    # Compute the Log-Spectral Distance
    lsd = np.sqrt(np.mean((log_S1 - log_S2) ** 2))
    return lsd

lsd = log_spectral_distance(target_distorted_audio, predicted_distorted_audio, dataset.distorted_rate)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"SNR: {snr} dB")
print(f"PSNR: {psnr} dB")
print(f"Pearson Correlation Coefficient: {pearson_corr}")
print(f"LSD: {lsd}")

# Plotting
if (False):
    plt.figure(figsize=(12, 6))
    plt.plot(target_distorted_audio, label='Distorted Audio (Target)', alpha=0.5)
    plt.plot(predicted_distorted_audio, label='Predicted Distorted Audio', alpha =0.5)
    plt.title('Distorted Audio Waveform (Target vs Predicted)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(waveform_path)
