import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt

# Paths to clean and distorted audio files
clean_audio_path = './shortaudio32fp/ts9_in.wav'
distorted_audio_path = './shortaudio32fp/ts9_out_drive=05.wav'
predicted_audio_path = './shortaudio32fp/predicted_out.wav'
waveform_path = './shortaudio32fp/waveform.png'

# Hyperparameters
sequence_length = 150
batch_size = 4096
num_epochs = 4
learning_rate = 1e-2
hidden_size = 64  # LSTM hidden size

# Custom Dataset class
class Seq2SamDataset(Dataset):
    def __init__(self, clean_path, distorted_path, sequence_length):
        self.sequence_length = sequence_length
        
        # Read audio files
        self.clean_rate, self.clean_audio = wavfile.read(clean_path)
        self.distorted_rate, self.distorted_audio = wavfile.read(distorted_path)
        
        # Normalize audio waveforms
        self.clean_audio = self.normalize_waveform(self.clean_audio)
        self.distorted_audio = self.normalize_waveform(self.distorted_audio)
        
        # Ensure both audio files have the same length (trim if necessary)
        min_length = min(len(self.clean_audio), len(self.distorted_audio))
        self.clean_audio = self.clean_audio[:min_length]
        self.distorted_audio = self.distorted_audio[:min_length]
        
        # Calculate number of samples
        self.num_samples = len(self.clean_audio) - sequence_length
    
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
        max_val = np.max(np.abs(waveform))
        return waveform.astype(np.float32) / max_val

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
dataset = Seq2SamDataset(clean_audio_path, distorted_audio_path, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}')

# Evaluate on entire dataset and plot results
model.eval()
predicted_distorted_audio = []
with torch.no_grad():
    for idx in range(len(dataset)):
        clean_seq, distorted_sample = dataset[idx]
        output = model(clean_seq)
        predicted_distorted_audio.append(output.item())
        if idx % 100000 == 0:
            print(f"Sample {idx}/{len(dataset)}: Predicted Distorted Audio: {output.item()}, Actual Distorted Audio: {distorted_sample.item()}")

# Convert predicted_distorted_audio to numpy array
predicted_distorted_audio = np.array(predicted_distorted_audio)

# Scale back to original range
predicted_distorted_audio = predicted_distorted_audio * np.max(np.abs(dataset.distorted_audio))
predicted_distorted_audio = predicted_distorted_audio.astype(np.float32)

wavfile.write(predicted_audio_path, dataset.distorted_rate, predicted_distorted_audio)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(dataset.clean_audio, label='Clean Audio')
plt.title('Clean Audio Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(dataset.distorted_audio, label='Distorted Audio (Target)')
plt.plot(predicted_distorted_audio, label='Predicted Distorted Audio')
plt.title('Distorted Audio Waveform (Target vs Predicted)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.savefig(waveform_path)
