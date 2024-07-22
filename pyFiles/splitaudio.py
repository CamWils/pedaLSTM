import os
import scipy.io.wavfile as wavfile
import numpy as np

# Paths
input_dir = "./audio32fp/"  # directory where your wav files are stored
output_dir = "./split_out/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Ratio for splitting
test_ratio = 0.1
val_ratio = 0.1
train_ratio = 0.8

def split_wav(file_path, test_ratio, val_ratio):
    # Read the wav file
    rate, data = wavfile.read(file_path)
    
    # Calculate split indices
    total_length = len(data)
    test_end = int(total_length * test_ratio)
    val_end = test_end + int(total_length * val_ratio)
    
    # Split the data
    test_data = data[:test_end]
    val_data = data[test_end:val_end]
    train_data = data[val_end:]
    
    return rate, test_data, val_data, train_data

#clean
file_name = f"ts9_in.wav"
file_path = os.path.join(input_dir, file_name)
    
# Split the wav file
rate, test_data, val_data, train_data = split_wav(file_path, test_ratio, val_ratio)
    
# Save the splits
test_file_path = os.path.join(output_dir, f"ts9_in_test.wav")
val_file_path = os.path.join(output_dir, f"ts9_in_val.wav")
train_file_path = os.path.join(output_dir, f"ts9_in_train.wav")
    
wavfile.write(test_file_path, rate, test_data)
wavfile.write(val_file_path, rate, val_data)
wavfile.write(train_file_path, rate, train_data)
    
print(f"Saved splits for {file_name}")


for i in range(11):
    # Create file name
    file_name = f"ts9_out_drive={i:02d}.wav"
    file_path = os.path.join(input_dir, file_name)
    
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        continue
    
    # Split the wav file
    rate, test_data, val_data, train_data = split_wav(file_path, test_ratio, val_ratio)
    
    # Save the splits
    test_file_path = os.path.join(output_dir, f"ts9_out_drive={i:02d}_test.wav")
    val_file_path = os.path.join(output_dir, f"ts9_out_drive={i:02d}_val.wav")
    train_file_path = os.path.join(output_dir, f"ts9_out_drive={i:02d}_train.wav")
    
    wavfile.write(test_file_path, rate, test_data)
    wavfile.write(val_file_path, rate, val_data)
    wavfile.write(train_file_path, rate, train_data)
    
    print(f"Saved splits for {file_name}")

print("Splitting complete.")