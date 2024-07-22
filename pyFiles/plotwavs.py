import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
import sys

def read_wav(file_path):
    """
    Read a WAV file and return the sample rate and waveform.
    """
    sample_rate, waveform = wavfile.read(file_path)
    return sample_rate, waveform

def plot_wav_files(file1, file2):
    """
    Read two WAV files and plot them on the same set of axes with 50% opacity.
    """
    # Read the WAV files
    sample_rate1, waveform1 = read_wav(file1)
    sample_rate2, waveform2 = read_wav(file2)
    
    # Ensure both files have the same sample rate
    if sample_rate1 != sample_rate2:
        print("The two WAV files have different sample rates.")
        return
    
    # Create time axis for plotting
    time1 = np.arange(len(waveform1)) / sample_rate1
    time2 = np.arange(len(waveform2)) / sample_rate2
    
    # Plot the waveforms
    plt.figure(figsize=(12, 6))
    plt.plot(time1, waveform1, alpha=0.5, label=file1.split('/')[-1])
    plt.plot(time2, waveform2, alpha=0.5, label=file2.split('/')[-1])
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform Comparison')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file1 = './audio32fp/predicted_out.wav'
    file2 = './audio32fp/ts9_out_drive=10.wav'
    plot_wav_files(file1, file2)
