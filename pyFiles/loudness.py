import scipy.io.wavfile as wavfile
import numpy as np
import sys

def compute_rms(waveform):
    """
    Compute the RMS level of a waveform.
    """
    rms = np.sqrt(np.mean(np.square(waveform)))
    return rms

def read_wav(file_path):
    """
    Read a WAV file and return the sample rate and waveform.
    """
    sample_rate, waveform = wavfile.read(file_path)
    return sample_rate, waveform

def compare_loudness(file1, file2):
    """
    Compare the loudness of the entire first WAV file to the first half of the second WAV file.
    """
    # Read the WAV files
    sample_rate1, waveform1 = read_wav(file1)
    sample_rate2, waveform2 = read_wav(file2)
    
    # Ensure both files have the same sample rate
    if sample_rate1 != sample_rate2:
        print("The two WAV files have different sample rates.")
        return
    
    # Compute the RMS level of the entire first file
    rms1 = compute_rms(waveform1)
    
    # Compute the RMS level of the first half of the second file
    half_length = len(waveform2) // 2
    rms2 = compute_rms(waveform2[:half_length])
    
    # Print the RMS levels
    print(f"RMS level of the entire {file1}: {rms1}")
    print(f"RMS level of the first half of {file2}: {rms2}")
    
    # Compare the RMS levels
    if rms1 > rms2:
        print(f"The entire {file1} is louder than the first half of {file2}")
    elif rms1 < rms2:
        print(f"The first half of {file2} is louder than the entire {file1}")
    else:
        print(f"The entire {file1} and the first half of {file2} have the same loudness")

if __name__ == "__main__":
        compare_loudness('./shortaudio32fp/predicted_out.wav', './shortaudio32fp/ts9_out_drive=10.wav')
