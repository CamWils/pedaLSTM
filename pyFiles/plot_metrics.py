import matplotlib.pyplot as plt
import re

# Function to read the metrics from the file
def read_metrics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    epochs = []
    mse = []
    rmse = []
    snr = []
    psnr = []
    pearson_corr = []
    lsd = []

    epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")
    mse_pattern = re.compile(r"MSE:\s+([\d.]+)$")
    rmse_pattern = re.compile(r"RMSE:\s+([\d.]+)$")
    snr_pattern = re.compile(r"SNR:\s+([\d.]+) dB$")
    psnr_pattern = re.compile(r"PSNR:\s+([\d.]+) dB$")
    pearson_pattern = re.compile(r"Pearson Correlation Coefficient:\s+([\d.]+)$")
    lsd_pattern = re.compile(r"LSD:\s+([\d.]+)$")

    for line in lines:
        epoch_match = epoch_pattern.match(line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))
        mse_match = mse_pattern.match(line)
        if mse_match:
            mse.append(float(mse_match.group(1)))
        rmse_match = rmse_pattern.match(line)
        if rmse_match:
            rmse.append(float(rmse_match.group(1)))
        snr_match = snr_pattern.match(line)
        if snr_match:
            snr.append(float(snr_match.group(1)))
        psnr_match = psnr_pattern.match(line)
        if psnr_match:
            psnr.append(float(psnr_match.group(1)))
        pearson_match = pearson_pattern.match(line)
        if pearson_match:
            pearson_corr.append(float(pearson_match.group(1)))
        lsd_match = lsd_pattern.match(line)
        if lsd_match:
            lsd.append(float(lsd_match.group(1)))

    return epochs, mse, rmse, snr, psnr, pearson_corr, lsd

# Function to plot each metric
def plot_metrics(epochs, metric_values, metric_name):
    plt.figure()
    plt.plot(epochs, metric_values, linestyle='-')
    plt.title(f'{metric_name} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.show()

# Path to the metrics file
metrics_file = './metrics/training_metrics_20240712_111350_shortAudio.txt'

# Read the metrics from the file
epochs, mse, rmse, snr, psnr, pearson_corr, lsd = read_metrics(metrics_file)
print(len(epochs))
print(len(mse))
print(len(rmse))
print(len(snr))
print(len(psnr))
print(len(pearson_corr))
print(len(lsd))

# Plot each metric
#epochs = range(1, 50 + 1)
plot_metrics(epochs, mse, 'MSE')
plot_metrics(epochs, rmse, 'RMSE')
plot_metrics(epochs, snr, 'SNR (dB)')
plot_metrics(epochs, psnr, 'PSNR (dB)')
plot_metrics(epochs, pearson_corr, 'Pearson Correlation Coefficient')
plot_metrics(epochs, lsd, 'LSD')

