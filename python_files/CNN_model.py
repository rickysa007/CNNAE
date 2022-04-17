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
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
absl.logging.set_verbosity(absl.logging.ERROR)


os.chdir('/home/ricky/RNNAE/conv_npy')

print('Loading in data...')
lc = np.array(np.load('lc.npy', allow_pickle=True))
lc_meta = np.array(np.load('lc_meta.npy', allow_pickle=True))
print('Finished loading in data')

def create_input(rep=32, split_portion=0.8, num_of_type=1):

    claimedtype = []

    input = [ [] for i in range(num_of_type+1)]
    input_train = [ [] for i in range(num_of_type+1)]
    input_test = [ [] for i in range(num_of_type+1)]

    input_meta = [ [] for i in range(num_of_type+1)]
    input_meta_train = [ [] for i in range(num_of_type+1)]
    input_meta_test = [ [] for i in range(num_of_type+1)]

    type_train = [ [] for i in range(num_of_type+1)]
    type_test = [ [] for i in range(num_of_type+1)]

    for i in range(len(lc_meta)):

        if 'Ia' in lc_meta[i]['type']:
                claimedtype.append(0)
        if 'IIP' in lc_meta[i]['type']:
                claimedtype.append(1)

    for i in range(len(claimedtype)):

        input[0].append(list(lc[i]))

        #input_meta_tmp = [lc_meta[i]['peak_mag'], lc_meta[i]['delta_m'], lc_meta[i]['t_normalised_noise'], lc_meta[i]['no_near_peak']]
        input_meta_tmp = [lc_meta[i]['peak_mag'], lc_meta[i]['delta_m']]
        input_meta[0].append(input_meta_tmp)

        if claimedtype[i] == 0:
            input[1].append(list(lc[i]))

            #input_meta_tmp = [lc_meta[i]['peak_mag'], lc_meta[i]['delta_m'], lc_meta[i]['t_normalised_noise'], lc_meta[i]['no_near_peak']]
            input_meta_tmp = [lc_meta[i]['peak_mag'], lc_meta[i]['delta_m']]
            input_meta[1].append(input_meta_tmp)

        if claimedtype[i] == 1:
            input[2].append(list(lc[i]))

            #input_meta_tmp = [lc_meta[i]['peak_mag'], lc_meta[i]['delta_m'], lc_meta[i]['t_normalised_noise'], lc_meta[i]['no_near_peak']]
            input_meta_tmp = [lc_meta[i]['peak_mag'], lc_meta[i]['delta_m']]
            input_meta[2].append(input_meta_tmp)

    for i in range(len(input)):

        #set the peak to be day 0 again     
        for j in range(len(input[i])):
            m_max_id = np.argmin(input[i][j][1]) #find maximum by g band
            t_max = input[i][j][0][m_max_id]
            input[i][j][0] -= t_max

        #insert 0 at the end of the lc (actually no need to pad time, do it here just for consistent shape)
        for j in range(len(input[i])):
            for k in range(96-input[i][j][0].shape[0]):
                for l in range(7):
                    input[i][j][l] = np.insert(input[i][j][l], -1, 0)

        #insert 0 in front of the lc and then truncate the end
        for j in range(len(input[i])):
            for k in range(int(25+input[i][j][0][0])):
                for l in range(6):
                    input[i][j][l+1] = np.insert(input[i][j][l+1], 0, 0)[:96]

        input[i] = np.array(input[i])

        input[i]             = K.cast_to_floatx(input[i].transpose((0, 2, 1)))
        input[i]             = np.repeat(input[i][:,:,1:-3], rep, axis=1)
        input[i]             = np.reshape(input[i],(input[i].shape[0], int(input[i].shape[1]/rep), int(input[i].shape[2]*rep), 1))

        input_train[i]       = input[i][:int(split_portion*len(input[i])),:,:,:]
        input_test[i]        = input[i][int(split_portion*len(input[i])):,:,:,:]

        input_meta[i]        = K.cast_to_floatx(input_meta[i]).astype(np.float)
        input_meta_train[i]  = input_meta[i][:int(split_portion*len(input[i]))]
        input_meta_test[i]   = input_meta[i][int(split_portion*len(input[i])):]

        type_train[i]        = claimedtype[:int(split_portion*len(input[i]))]
        type_test[i]         = claimedtype[int(split_portion*len(input[i])):]

    for i in range(len(input)-1):
        print(f'For type {claimedtype[i]}, total size of data is {input[i].shape}, training size is {input_train[i].shape}, testing size is {input_test[i].shape}')

    return input, input_train, input_test, input_meta, input_meta_train, input_meta_test, type_train, type_test

def cnnae(input, input_meta, d=64):

    w = input[0].shape[1]
    h = input[0].shape[2]
    a = input_meta[0].shape[1]

    input_seq = keras.Input(shape=(w, h, 1))
    input_meta = keras.Input(shape=(a))

    # Encoder
    x = layers.BatchNormalization()(input_seq)
    x = layers.Conv2D(d, (3, 3), activation=layers.LeakyReLU(), padding="same")(x) # 96*96*64
    x = layers.MaxPooling2D((2, 2), padding="same")(x) # 48*48*64
    x = layers.Conv2D(d, (3, 3), activation=layers.LeakyReLU(), padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x) # 24*24*64
    x = layers.Conv2D(d, (3, 3), activation=layers.LeakyReLU(), padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x) # 12*12*64
    x = layers.Conv2D(d, (3, 3), activation=layers.LeakyReLU(), padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x) # 6*6*64
    x = layers.Conv2D(d, (3, 3), activation=layers.LeakyReLU(), padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x) # 3*3*64
    x = layers.Conv2D(d, (3, 3), activation=layers.LeakyReLU(), padding="same")(x)
    x = layers.MaxPooling2D((3, 3), padding="same")(x) # 1*1*64
    x = layers.Flatten()(x) # 64
    x = layers.Dense(d, activation=layers.LeakyReLU())(x)
    x = layers.Dense(d//8, activation='linear')(x) #16
    encoded = layers.concatenate([x, input_meta], axis=-1)

    # Decoder
    x = layers.Dense(d, activation=layers.LeakyReLU())(encoded) #64
    x = layers.Reshape((1, 1, d), input_shape=(d,))(x) # 1*1*64
    x = layers.Conv2DTranspose(d, (3, 3), strides=3, activation=layers.LeakyReLU(), padding="same")(x) # 3*3*64
    x = layers.Conv2DTranspose(d, (3, 3), strides=2, activation=layers.LeakyReLU(), padding="same")(x) # 6*6*64
    x = layers.Conv2DTranspose(d, (3, 3), strides=2, activation=layers.LeakyReLU(), padding="same")(x) # 12*12*64
    x = layers.Conv2DTranspose(d, (3, 3), strides=2, activation=layers.LeakyReLU(), padding="same")(x) # 24*24*64
    x = layers.Conv2DTranspose(d, (3, 3), strides=2, activation=layers.LeakyReLU(), padding="same")(x) # 48*48*64
    x = layers.Conv2DTranspose(d, (3, 3), strides=2, activation=layers.LeakyReLU(), padding="same")(x) # 96*96*64
    decoded = layers.Conv2D(1, (3, 3), activation="tanh", padding="same")(x)

    autoencoder = keras.Model([input_seq, input_meta], decoded)
    encoder = keras.Model([input_seq, input_meta], encoded)

    opt = Adam(learning_rate=0.0001)

    autoencoder.compile(optimizer=opt, loss="mse")
    print(autoencoder.summary())

    return autoencoder, encoder

def cnnae_train(autoencoder, input_tmp, input_meta_tmp, patience=10, epochs=100, **kwargs):
    
	if kwargs['callbacks']:
		callbacks = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
								verbose=0, mode='min', baseline=None,
								restore_best_weights=True)
	else:
		callbacks = None

	history = autoencoder.fit(
							x=[input_tmp, input_meta_tmp],
							y=input_tmp,
							validation_split = 0.1,
							epochs=epochs,
							verbose=2,
							callbacks=callbacks)

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.grid()
	plt.ylim(0, 5e-3)

	os.chdir('/home/ricky/RNNAE/CNN_product')
	plt.savefig('CNN training history.pdf')

	return

def main():

    input, input_train, input_test, input_meta, input_meta_train, input_meta_test, type_train, type_test = create_input()

    os.chdir('/home/ricky/RNNAE/CNN_product/CNN_npy')
    np.save('input.npy', np.array(input, dtype=object))
    np.save('input_train.npy', np.array(input_train, dtype=object))
    np.save('input_test.npy', np.array(input_test, dtype=object))
    np.save('input_meta.npy', np.array(input_meta, dtype=object))
    np.save('input_meta_train.npy', np.array(input_meta_train, dtype=object))
    np.save('input_meta_test.npy', np.array(input_meta_test, dtype=object))
    np.save('type_train.npy', np.array(type_train, dtype=object))
    np.save('type_test.npy', np.array(type_test, dtype=object))

    autoencoder, encoder = cnnae(input, input_meta, d=64)
    cnnae_train(autoencoder, input_train[0], input_meta_train[0], epochs=500, callbacks=False)

    autoencoder.save('/home/ricky/RNNAE/CNN_product/CNN_autoencoder_model')
    encoder.save('/home/ricky/RNNAE/CNN_product/CNN_encoder_model')

    print('end of CNN_model.py')

if __name__ == '__main__':
	main()