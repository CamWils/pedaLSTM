import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory containing the metric files
metrics_dir = './metrics/test/'

# Initialize dictionaries to store metrics for each epoch
metrics = {
    'MSE': defaultdict(list),
    'RMSE': defaultdict(list),
    'SNR': defaultdict(list),
    'PSNR': defaultdict(list),
    'Pearson Correlation Coefficient': defaultdict(list),
    'LSD': defaultdict(list)
}

# Regex patterns to extract data
epoch_pattern = re.compile(r'Epoch (\d+)/100$')
metric_patterns = {
    'MSE': re.compile(r'MSE: ([\d.]+)$'),
    'RMSE': re.compile(r'RMSE: ([\d.]+)$'),
    'SNR': re.compile(r'SNR: ([\d.]+) dB$'),
    'PSNR': re.compile(r'PSNR: ([\d.]+) dB$'),
    'Pearson Correlation Coefficient': re.compile(r'Pearson Correlation Coefficient: ([\d.]+)$'),
    'LSD': re.compile(r'LSD: ([\d.]+)$')
}

# Read each file and extract metrics
for filename in os.listdir(metrics_dir):
    if filename.startswith('training_metrics_') and filename.endswith('.txt'):
        with open(os.path.join(metrics_dir, filename), 'r') as file:
            current_epoch = None
            for line in file:
                # Match epoch
                epoch_match = epoch_pattern.match(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    continue
                
                # Match metrics
                for metric, pattern in metric_patterns.items():
                    match = pattern.match(line)
                    if match and current_epoch is not None:
                        metrics[metric][current_epoch].append(float(match.group(1)))

# Create box plots for each metric
for metric, data in metrics.items():
    plt.figure()
    sorted_epochs = sorted(data.keys())
    aggregated_data = [data[epoch] for epoch in sorted_epochs]
    plt.boxplot(aggregated_data, tick_labels=sorted_epochs)
    plt.title(f'Box and Whisker Plot for {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
