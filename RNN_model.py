import subprocess as sp
import os
import silence_tensorflow.auto # pylint: disable=unused-import
import tensorflow as tf

def mask_unused_gpus(leave_unmasked=1):

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

  try:
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

    if len(available_gpus) < leave_unmasked: raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(gpus[available_gpus[0]], 'GPU')
  
  except Exception as e:
    print('"nvidia-smi" is probably not installed. GPUs are not masked', e)

mask_unused_gpus()

import absl.logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Masking
from tensorflow.keras.layers import TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

absl.logging.set_verbosity(absl.logging.ERROR)

# Save graphs --done, refine functions, anamoly dectection --done, better masking --done, Fourier Transform of lc.

os.chdir('/home/ricky/RNNAE')

data_GP = np.array(np.load('data_GP.npy', allow_pickle=True))
data_meta_GP = np.array(np.load('data_meta_GP.npy', allow_pickle=True))

def create_input(split_portion=0.8, num_of_type=1):

    claimedtype = []

    input = [ [] for i in range(num_of_type+1)]
    input_train = [ [] for i in range(num_of_type+1)]
    input_test = [ [] for i in range(num_of_type+1)]
    type_train = [ [] for i in range(num_of_type+1)]
    type_test = [ [] for i in range(num_of_type+1)]

    for i in range(len(data_meta_GP)):

        if 'Ia' in data_meta_GP[i][-2]:
            claimedtype.append(0)
        if 'IIP' in data_meta_GP[i][-2]:
            claimedtype.append(1)

    for i in range(len(claimedtype)):

        input[0].append(list(data_GP[i]))

        if claimedtype[i] == 0:
            input[1].append(list(data_GP[i]))
        if claimedtype[i] == 1:
            input[2].append(list(data_GP[i]))

    for i in range(len(input)):
        input[i] = np.array(input[i])

        input_train[i] = input[i][:int(split_portion*len(input[i]))]
        input_test[i]  = input[i][int(split_portion*len(input[i])):]

        input[i] = input[i].transpose((0, 2, 1))
        input_train[i] = K.cast_to_floatx(input_train[i].transpose((0, 2, 1)))
        input_test[i] = K.cast_to_floatx(input_test[i].transpose((0, 2, 1)))

        type_train[i] = claimedtype[:int(split_portion*len(input[i]))]
        type_test[i]  = claimedtype[int(split_portion*len(input[i])):]

    for i in range(len(input)-1):
        print(f'For type {claimedtype[i]}, total size of data is {input[i].shape}, training size is {input_train[i].shape}, testing size is {input_test[i].shape}')

    return input, input_train, input_test, type_train, type_test

def masking(input_tmp, split):

    weight = []
    input_tmp = input_tmp

    for i in range(input_tmp.shape[0]):
        tmp1 = [1 for j in range(data_GP[i+split][-3])]
        tmp2 = [0 for j in range(input_tmp.shape[1] - data_GP[i+split][-3])]
        tmp = np.hstack((tmp1, tmp2))
        weight.append(tmp)

    weight = np.array(weight)

    mask = [weight for i in range(input_tmp.shape[2])]
    mask = np.transpose(mask, (1, 2, 0))

    return mask

def custom_loss(y_true, y_pred, mask):

    last_band_id = int((data_GP.shape[1] - 3 - 1)/2)+1

    y_true_masked = tf.math.multiply(y_true, mask)
    y_pred_masked = tf.math.multiply(y_pred, mask[:,:,1:last_band_id])
    
    mse = tf.keras.losses.mean_squared_error(y_true = y_true_masked[:,:,1:last_band_id], y_pred = y_pred_masked[:,:,:])

    return mse

def custom_loss2(y_true, y_pred):

    last_band_id = int((data_GP.shape[1] - 3 - 1)/2)+1
    
    mse = tf.keras.losses.mean_squared_error(y_true = y_true[:,:,1:last_band_id], y_pred = y_pred[:,:,:])

    return mse

def rnnae(input):

    input_seq = keras.Input(shape=(input[0].shape[1], input[0].shape[2]))
    input_mask = keras.Input(shape=(input[0].shape[1], input[0].shape[2]))
    #input_conc_phase = keras.Input(shape=(input[0].shape[1], 1))

    x = BatchNormalization()(input_seq)
    x = Bidirectional(GRU(185, activation='tanh', return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(50, activation='tanh', return_sequences=True))(x)
    x = BatchNormalization()(x)
    encoded = Bidirectional(GRU(6, activation='tanh', return_sequences=False))(x)

    x = RepeatVector(input[0].shape[1])(encoded)
    #merged = concatenate([x, input_conc_phase], axis=-1)
    #x = BatchNormalization()(merged)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(50, activation='tanh', return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(185, activation='tanh', return_sequences=True))(x)
    x = BatchNormalization()(x)
    decoded = TimeDistributed(Dense(input[0].shape[2]-3-1))(x)

    #autoencoder = keras.Model([input_seq, input_mask, input_conc_phase], decoded)
    autoencoder = keras.Model([input_seq, input_mask], decoded)
    autoencoder.add_loss(custom_loss(input_seq, decoded, input_mask))
    encoder = keras.Model(input_seq, encoded)

    opt = Adam(learning_rate=0.0001)

    autoencoder.compile(optimizer=opt, loss=None)
    #autoencoder.summary()

    return autoencoder, encoder

def rnnae2(input):

    input_seq = keras.Input(shape=(input[0].shape[1], input[0].shape[2]))

    x = Masking(mask_value=0., input_shape=(input[0].shape[1], input[0].shape[2]))(input_seq)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(input[0].shape[1], activation='tanh', return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(50, activation='tanh', return_sequences=True))(x)
    x = BatchNormalization()(x)
    encoded = Bidirectional(GRU(6, activation='tanh', return_sequences=False))(x)

    x = RepeatVector(input[0].shape[1])(encoded)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(50, activation='tanh', return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(input[0].shape[1], activation='tanh', return_sequences=True))(x)
    x = BatchNormalization()(x)
    decoded = TimeDistributed(Dense(input[0].shape[2]-3-1))(x)

    autoencoder = keras.Model(input_seq, decoded)
    autoencoder.add_loss(custom_loss2(input_seq, decoded))
    encoder = keras.Model(input_seq, encoded)

    opt = Adam(learning_rate=0.0001)

    autoencoder.compile(optimizer=opt, loss=None)
    #autoencoder.summary()

    return autoencoder, encoder

def rnnae_train(autoencoder, input_tmp, mask_tmp, patience=40, epochs=1000):

    callbacks = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                       verbose=0, mode='min', baseline=None,
                       restore_best_weights=True)

    history = autoencoder.fit(x=[input_tmp, mask_tmp], y=None,
                            validation_split = 0.1,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[callbacks])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid()
    plt.ylim(0, 0.05)

    os.chdir('/home/ricky/RNNAE')
    plt.savefig('RNN training history.pdf')

    return

def rnnae_train2(autoencoder, input_tmp, patience=40, epochs=1000):

    callbacks = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                       verbose=0, mode='min', baseline=None,
                       restore_best_weights=True)

    history = autoencoder.fit(x=input_tmp, y=None,
                            validation_split = 0.1,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[callbacks])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid()
    plt.ylim(0, 0.05)
    
    os.chdir('/home/ricky/RNNAE')
    plt.savefig('RNN training history.pdf')

    return

def main():

    input, input_train, input_test, type_train, type_test = create_input()

    os.chdir('/home/ricky/RNNAE/RNN_npy')
    np.save('input.npy', np.array(input, dtype=object))
    np.save('input_train.npy', np.array(input_train, dtype=object))
    np.save('input_test.npy', np.array(input_test, dtype=object))
    np.save('type_train.npy', np.array(type_train, dtype=object))
    np.save('type_test.npy', np.array(type_test, dtype=object))

    autoencoder, encoder = rnnae2(input)
    rnnae_train2(autoencoder, input_train[0], 30, 1000)

    autoencoder.save('/home/ricky/RNNAE/RNN_autoencoder_model')
    encoder.save('/home/ricky/RNNAE/RNN_encoder_model')

    print('end of RNN_model.py')

if __name__ == '__main__':
    main()