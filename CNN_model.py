from enum import auto
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
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
absl.logging.set_verbosity(absl.logging.ERROR)


os.chdir('/home/ricky/RNNAE')

print('Loading in data...')
data_GP = np.array(np.load('data_GP.npy', allow_pickle=True))
data_meta_GP = np.array(np.load('data_meta_GP.npy', allow_pickle=True))
print('Finished loading in data')

def create_input(rep=16, split_portion=0.8, num_of_type=1):

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

		input[i]       = K.cast_to_floatx(input[i].transpose((0, 2, 1)))
		input_train[i] = K.cast_to_floatx(input_train[i].transpose((0, 2, 1)))
		input_test[i]  = K.cast_to_floatx(input_test[i].transpose((0, 2, 1)))

		input[i]       = np.repeat(input[i][:,:,1:-3], rep, axis=1)
		input_train[i] = np.repeat(input_train[i][:,:,1:-3], rep, axis=1)
		input_test[i]  = np.repeat(input_test[i][:,:,1:-3], rep, axis=1)

		input[i]       = np.reshape(input[i],(input[i].shape[0], int(input[i].shape[1]/rep), int(input[i].shape[2]*rep), 1))
		input_train[i] = np.reshape(input_train[i],(input_train[i].shape[0], int(input_train[i].shape[1]/rep), int(input_train[i].shape[2]*rep), 1))
		input_test[i]  = np.reshape(input_test[i],(input_test[i].shape[0], int(input_test[i].shape[1]/rep), int(input_test[i].shape[2]*rep), 1))

		type_train[i]  = claimedtype[:int(split_portion*len(input[i]))]
		type_test[i]   = claimedtype[int(split_portion*len(input[i])):]

	for i in range(len(input)-1):
		print(f'For type {claimedtype[i]}, total size of data is {input[i].shape}, training size is {input_train[i].shape}, testing size is {input_test[i].shape}')

	return input, input_train, input_test, type_train, type_test

def cnnae(input):

	input_seq = keras.Input(shape=(input[0].shape[1], input[0].shape[2], 1))

	x = Conv2D(64, (3, 3), activation="relu", padding="same")(input_seq)
	x = MaxPooling2D((2, 2), padding="same")(x)
	x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
	encoded = MaxPooling2D((2, 2), padding="same")(x)

	# Decoder
	x = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(encoded)
	x = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
	decoded = Conv2D(1, (3, 3), activation="tanh", padding="same")(x)

	autoencoder = keras.Model(input_seq, decoded)
	encoder = keras.Model(input_seq, encoded)

	opt = Adam(learning_rate=0.0001)

	autoencoder.compile(optimizer=opt, loss="mse")
	print(autoencoder.summary())

	return autoencoder, encoder

def cnnae_train(autoencoder, input_tmp, patience=10, epochs=200):

	callbacks = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
						verbose=0, mode='min', baseline=None,
						restore_best_weights=True)

	history = autoencoder.fit(
							x=input_tmp,
							y=input_tmp,
							validation_split = 0.1,
							epochs=epochs,
							verbose=1,
							callbacks=[callbacks])

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.grid()
	plt.ylim(0, 1e-3)

	os.chdir('/home/ricky/RNNAE')
	plt.savefig('CNN training history.pdf')

	return

def main():

	input, input_train, input_test, type_train, type_test = create_input()

	os.chdir('/home/ricky/RNNAE/CNN_npy')
	np.save('input.npy', np.array(input, dtype=object))
	np.save('input_train.npy', np.array(input_train, dtype=object))
	np.save('input_test.npy', np.array(input_test, dtype=object))
	np.save('type_train.npy', np.array(type_train, dtype=object))
	np.save('type_test.npy', np.array(type_test, dtype=object))

	autoencoder, encoder = cnnae(input)
	cnnae_train(autoencoder, input_train[0])

	autoencoder.save('/home/ricky/RNNAE/CNN_autoencoder_model')
	encoder.save('/home/ricky/RNNAE/CNN_encoder_model')

	print('end of CNN_model.py')

if __name__ == '__main__':
	main()