import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
import CoreAudioML.networks as networks
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import scipy.io.wavfile as wavfile
import numpy as np
import math
import os
from datetime import datetime
from contextlib import nullcontext

argparse = argparse.ArgumentParser()
# File names & directories
argparse.add_argument('--device', '-d', default='ts9', type=str)
argparse.add_argument('--data_path', '-d_path', default='./audio32fp', type=str)
argparse.add_argument('--clean_audio_path', '-c_path', default='./audio32fp_split/ts9_in.wav', type=str)
argparse.add_argument('--target_audio_path', '-t_path', default='./audio32fp/ts9_out_drive=05.wav', type=str)
argparse.add_argument('--model_save_path', '-m_path', default='./models/', type=str)
argparse.add_argument('--file_name', '-fn', default='ts9', type=str)
# Model architecture
argparse.add_argument('--input_size', '-is', default=1, type=int)
argparse.add_argument('--output_size', '-os', default=1, type=int)
argparse.add_argument('--hidden_size', '-hs', default=64, type=int)
argparse.add_argument('--num_layers', '-nl', default=1, type=int)
argparse.add_argument('--bias', '-b', default=True, type=bool)
# Data & training
argparse.add_argument('--sequence_length', '-sl', default=44100, type=int)
argparse.add_argument('--epochs', '-ne', default=10, type=int)
argparse.add_argument('--val_freq', '--vf', default=2, type=int)
argparse.add_argument('--val_patience', '-vp', default=20, type=int)
argparse.add_argument('--batch_size', '-bs', default=50, type=int)
argparse.add_argument('--learning_rate', '-lr', default=1e-3, type=float)
argparse.add_argument('--weight_decay', '-wd', default=5e-5, type=float)
argparse.add_argument('--init_len', '-il', default=200, type=int)
argparse.add_argument('--update_freq', '-uf', default=1000, type=int)
argparse.add_argument('--cuda', '-cu', default=1, type=int)
argparse.add_argument('--chunk', '-ch', default=100000, type=int)

args = argparse.parse_args()

class pedaLSTM(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=64, num_layers=1, bias=True, batch_first=False):
        super(pedaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = None
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_state = True

    def forward(self, x):
        out, self.hidden = self.lstm(x, self.hidden)
        out = self.linear(out)
        return out
    
    def reset(self):
        self.hidden = None

    def detach(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    def save_model(self, file_name=''):
        if self.save_state:
            if not file_name:
                file_name = './models/' + self.timestamp + '.pt'
            torch.save(self.state_dict(), file_name)

    def train_epoch(self, input_data, target_data, loss_fcn, optim, bs, init_len=200, up_fr=1000): # from CoreAudioML/networks.py
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # Iterate over the batches
        ep_loss = 0
        for batch_i in range(math.ceil(shuffle.shape[0] / bs)):
            # Load batch of shuffled segments
            input_batch = input_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]
            target_batch = target_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]

            # Initialise network hidden state by processing some samples then zero the gradient buffers
            self(input_batch[0:init_len, :, :])
            self.zero_grad()

            # Choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
            start_i = init_len
            batch_loss = 0
            # Iterate over the remaining samples in the mini batch
            for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
                # Process input batch with neural network
                output = self(input_batch[start_i:start_i + up_fr, :, :])

                # Calculate loss and update network parameters
                loss = loss_fcn(output, target_batch[start_i:start_i + up_fr, :, :])
                loss.backward()
                optim.step()

                # Set the network hidden state, to detach it from the computation graph
                self.detach()
                self.zero_grad()

                # Update the start index for the next iteration and add the loss to the batch_loss total
                start_i += up_fr
                batch_loss += loss

            # Add the average batch loss to the epoch loss and reset the hidden states to zeros
            ep_loss += batch_loss / (k + 1)
            self.reset()
        return ep_loss / (batch_i + 1)
    
    def process_data(self, input_data, target_data, loss_fcn, chunk, grad=False): # from CoreAudioML/networks.py
        with (torch.no_grad() if not grad else nullcontext()):
            output = torch.empty_like(target_data)
            for l in range(int(output.size()[0] / chunk)):
                output[l * chunk:(l + 1) * chunk] = self(input_data[l * chunk:(l + 1) * chunk])
                self.detach()
            # If the data set doesn't divide evenly into the chunk length, process the remainder
            if not (output.size()[0] / chunk).is_integer():
                output[(l + 1) * chunk:-1] = self(input_data[(l + 1) * chunk:-1])
            self.reset()
            loss = loss_fcn(output, target_data)
        return output, loss

if __name__ == "__main__":
    pedaLSTM = pedaLSTM(input_size=args.input_size, output_size=args.output_size, hidden_size=args.hidden_size,
                       num_layers=args.num_layers, bias=args.bias, batch_first=False)

    if not torch.cuda.is_available() or args.cuda == 0:
        cuda = 0
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(0)
        pedaLSTM = pedaLSTM.cuda()
        cuda = 1
    
    optimiser = torch.optim.Adam(pedaLSTM.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5)
    loss_function = training.ESRLoss()

    # Data processing
    total_data = dataset.DataSet(data_dir=args.data_path)

    total_data.create_subset('train', frame_len=44100)
    #total_data.load_file('/train/ts9', 'train')
    total_data.load_file(os.path.join('train', args.file_name), 'train')

    total_data.create_subset('val')
    total_data.load_file(os.path.join('val', args.file_name), 'val')

    total_data.create_subset('test')
    total_data.load_file(os.path.join('test', args.file_name), 'test')

    pedaLSTM.save_state = True
    
    min_val_loss = 1e10 # dummy big value
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        print('>> Training epoch: ' + str(epoch) + ' out of ' + str(args.epochs))
        epoch_loss = pedaLSTM.train_epoch(total_data.subsets['train'].data['input'][0],
                                          total_data.subsets['train'].data['target'][0],
                                          loss_function, optimiser, args.batch_size)
        #print('>>                   Trained epoch:  ' + str(epoch))
        if epoch % args.val_freq == 0:
            val_output, val_loss = pedaLSTM.process_data(total_data.subsets['val'].data['input'][0],
                                                         total_data.subsets['val'].data['target'][0], loss_function, args.chunk)
            scheduler.step(val_loss)
            print(">> Validation loss: " + str(val_loss))
            if val_loss < min_val_loss:
                patience_count = 0
                #pedaLSTM.save_model('best', args.model_save_path)
                pedaLSTM.save_model()
                wavfile.write(os.path.join(args.model_save_path, 'val_out_best.wav'),
                              total_data.subsets['test'].fs, val_output.cpu().numpy()[:, 0, 0])
            else:
                patience_count += 1
            #print('>> Current LR: ' + str(optimiser.param_groups[0]['lr']))
            print('>> Current LR: ' + str(scheduler.get_last_lr()))
            if args.val_patience and patience_count > args.val_patience:
                print('>> Validation patience limit exceded at epoch ' + str(epoch))
                break
        
    lossESR = training.ESRLoss()
    test_output, test_loss = pedaLSTM.process_data(total_data.subsets['test'].data['input'][0],
                                                    total_data.subsets['test'].data['target'][0],
                                                    loss_function, args.chunk)
    print('>> Test loss: ' + str(test_loss))
    wavfile.write(os.path.join(args.model_save_path, "test_out_final.wav"), total_data.subsets['test'].fs, test_output.cpu().numpy()[:, 0, 0])
    print('>> Saved test_out_final.wav to ' + str(args.model_save_path))