# Absolute spaghetti nonsense
import scipy.io.wavfile as wavfile
import numpy as np


if __name__ == "__main__":
    base_dir = './audio32fp_split/'
    exts = ['ts9_in', 
            'ts9_out_drive=00', 'ts9_out_drive=01', 'ts9_out_drive=02', 'ts9_out_drive=03', 
            'ts9_out_drive=04', 'ts9_out_drive=05', 'ts9_out_drive=06', 'ts9_out_drive=07', 
            'ts9_out_drive=08', 'ts9_out_drive=09', 'ts9_out_drive=10']
    # WITHOUT second channel for drive level
    test_audio = []
    train_audio = []
    val_audio = []

    dist_test = []
    dist_train = []
    dist_val = []

    test_par = []
    train_par = []
    val_par = []

    
    for i, dir in enumerate(exts):
        if i == 0:
            test_f, test_data = wavfile.read(base_dir + dir + '/' + dir + '_test.wav')
            test_size = len(test_data)
            #test_audio = np.append(test_audio, test_data) 

            train_f, train_data = wavfile.read(base_dir + dir + '/' + dir + '_train.wav')
            train_size = len(train_data)
            #train_audio = np.append(train_audio, train_data) 

            val_f, val_data = wavfile.read(base_dir + dir + '/' + dir + '_val.wav')
            val_size = len(val_data)
            #val_audio = np.append(val_audio, val_data)

        else:
            test_audio = np.append(test_audio, test_data) 
            train_audio = np.append(train_audio, train_data) 
            val_audio = np.append(val_audio, val_data)

            drive = float((i - 1)/10)
            #print('i: ' + str(i) + '; drive: ' + str(drive))
            test_par = np.append(test_par, np.ones(test_size) * drive)
            train_par = np.append(train_par, np.ones(train_size) * drive)
            val_par = np.append(val_par, np.ones(val_size) * drive)

            test_f, test_data = wavfile.read(base_dir + dir + '/' + dir + '_test.wav')
            dist_test = np.append(dist_test, test_data)

            train_f, train_data = wavfile.read(base_dir + dir + '/' + dir + '_train.wav')
            dist_train =np.append(dist_train, train_data)

            val_f, val_data = wavfile.read(base_dir + dir + '/' + dir + '_val.wav')
            dist_val = np.append(dist_val, val_data)

    # Make clean stereo
    clean_str_test = np.column_stack((test_audio, test_par))
    clean_str_train = np.column_stack((train_audio, train_par))
    clean_str_val = np.column_stack((val_audio, val_par))
    # Write stereo clean wavfiles
    wavfile.write('./audio32fp_param/test/ts9-input.wav', val_f, clean_str_test.astype(np.float32))
    wavfile.write('./audio32fp_param/train/ts9-input.wav', val_f, clean_str_train.astype(np.float32))
    wavfile.write('./audio32fp_param/val/ts9-input.wav', val_f, clean_str_val.astype(np.float32))
    # Write mono distorted wavfiles
    wavfile.write('./audio32fp_param/test/ts9-target.wav', val_f, dist_test.astype(np.float32))
    wavfile.write('./audio32fp_param/train/ts9-target.wav', val_f, dist_train.astype(np.float32))
    wavfile.write('./audio32fp_param/val/ts9-target.wav', val_f, dist_val.astype(np.float32))