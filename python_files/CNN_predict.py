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
import shutil
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

absl.logging.set_verbosity(absl.logging.ERROR)


os.chdir('/home/ricky/RNNAE/conv_npy')

lc = np.array(np.load('lc.npy', allow_pickle=True))
lc_meta = np.array(np.load('lc_meta.npy', allow_pickle=True))

def import_data(x):

    x = np.load(f'{x}.npy', allow_pickle=True)
    x = np.asarray(x).astype('float32')

    return x

def create_clean_directory(d):

    isExist = os.path.exists(d)
    if isExist:
        shutil.rmtree(d)
        os.makedirs(d)
    else:
        os.makedirs(d)

    return

def cnnae_test(autoencoder, input_tmp, input_meta_tmp):

    pred = autoencoder.predict(x=[input_tmp, input_meta_tmp], verbose=1)
    pred_loss = autoencoder.evaluate(x=[input_tmp, input_meta_tmp], y=input_tmp, verbose=1)

    return pred, pred_loss

def latent_space_concatenation(latent_space, meta_data: list, split):
    latent_space_add = []
    for i in range(latent_space.shape[0]):
        latent_space_add.append([lc_meta[i+split][j] for j in meta_data])
    latent_space_add = np.array(latent_space_add)
    
    return np.concatenate((latent_space, latent_space_add), axis=-1)

def isolation_forest(latent_space, n_tree, split):

    clf = IsolationForest(n_estimators=n_tree, warm_start=True)
    clf.fit(latent_space)
    anomaly = clf.score_samples(latent_space)
    anomaly_id = np.argsort(anomaly)

    d = '/home/ricky/RNNAE/CNN_product/CNN_anomaly_graph'

    create_clean_directory(d)

    for i, ano in enumerate(anomaly_id):
        name = lc_meta[ano+split]['SN_name']
        peak_mag = "{:.2f}".format(lc_meta[ano+split]['peak_mag'])
        delta_m = "{:.2f}".format(lc_meta[ano+split]['delta_m'])
        t_normalised_noise = "{:.2f}".format(lc_meta[ano+split]['t_normalised_noise'])
        no_near_peak = "{:.2f}".format(lc_meta[ano+split]['no_near_peak'])

        try:
            shutil.copy(f'/home/ricky/RNNAE/SDSS_GP_graph/{name}.pdf', f'/home/ricky/RNNAE/CNN_product/CNN_anomaly_graph/{i}_{name}_{peak_mag}_{delta_m}_{t_normalised_noise}_{no_near_peak}.pdf')
        except:
            shutil.copy(f'/home/ricky/RNNAE/SDSS_prime_GP_graph/{name}.pdf', f'/home/ricky/RNNAE/CNN_product/CNN_anomaly_graph/{i}_{name}_{peak_mag}_{delta_m}_{t_normalised_noise}_{no_near_peak}.pdf')

    return anomaly_id

def latent_space_graph(latent_space, anomaly_id, split):

    d = '/home/ricky/RNNAE/CNN_product/CNN_latent_space_graph'
    create_clean_directory(d)
    os.chdir(d)

    color = [0 for i in range(latent_space.shape[0])]
    for i, ano in enumerate(anomaly_id):
        color[ano] = i

    print('plotting latent space graphs...')

    for i in tqdm(range(latent_space.shape[1] - 1)):
        for j in range(latent_space.shape[1] - 1 - i):
            fig = plt.figure(figsize=(8, 6))
            plt.grid() 
            plt.scatter(latent_space[:,i], latent_space[:,i+j+1], c=color, cmap='viridis', s=6)
            plt.colorbar()
            plt.title(f'id {i} vs id {i+j+1}')
            plt.savefig(f'id_{i}_vs_id_{i+j+1}.pdf', bbox_inches='tight')
            plt.close()

    return

def reconstruction_graph(input_tmp, pred, split, filters=['g', 'r', 'i']):

    color1 = ['seagreen', 'crimson', 'maroon']
    color2 = ['darkgreen', 'firebrick', 'darkred']

    d = '/home/ricky/RNNAE/CNN_product/CNN_reconstruction_graph'
    create_clean_directory(d)

    for i in tqdm(range(input_tmp.shape[0])):

        os.chdir(d)

        isExist = os.path.exists(f'./{lc_meta[i+split]["SN_name"]}')

        if not isExist:
            os.makedirs(f'./{lc_meta[i+split]["SN_name"]}')
            os.chdir(f'./{lc_meta[i+split]["SN_name"]}')

        fig, axs = plt.subplots(3, figsize=(12, 18))

        fig.suptitle('Images of CNN')
        axs[0].set_title('input test image')
        axs[1].set_title('reconstructed test image')
        axs[2].set_title('difference')

        a0 = axs[0].imshow(input_tmp[i].reshape(96,96).T, interpolation='nearest', aspect='auto', cmap='BrBG')
        a1 = axs[1].imshow(pred[i].reshape(96,96).T, interpolation='nearest', aspect='auto', cmap='BrBG')
        a2 = axs[2].imshow((input_tmp[i] - pred[i]).reshape(96,96).T, interpolation='nearest', aspect='auto', cmap='BrBG')

        fig.colorbar(a0, ax=axs[0])
        fig.colorbar(a1, ax=axs[1])
        fig.colorbar(a2, ax=axs[2])

        fig.savefig(f'./{lc_meta[i+split]["SN_name"]}.pdf', bbox_inches='tight')

        plt.close()

        for j, filter in enumerate(filters):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1)

            plt.gca().invert_yaxis()

            # And a corresponding grid
            ax.grid(which='major', alpha=0.8)
            ax.grid(which='minor', alpha=0.3)

            plt.xlabel('Timestep', fontsize=15)
            plt.ylabel('Normalized Absolute Magnitude', fontsize=15)

            plt.title(f'{lc_meta[i+split]["SN_name"]}, {lc_meta[i+split]["type"]}, {filter}')

            plt.scatter(np.linspace(0, 96, 96), input_tmp[i][:,j,:], s=2, marker='o', color=color1[j], label=f'test data'.format('o'))
            plt.scatter(np.linspace(0, 96, 96), pred[i][:,j,:], s=12, marker='X', color=color2[j], label=f'reconstruction'.format('X'))
            
            plt.legend()

            plt.savefig(f'./{lc_meta[i+split]["SN_name"]}_{filter}_band.pdf')

            plt.close()

    return

def cnn_predict(autoencoder, input_tmp, input_meta_tmp, **kwargs):

    split = int(0.8*(lc.shape[0]))

    if kwargs['training_data']:
        split = 0

    pred, pred_loss = cnnae_test(autoencoder, input_tmp, input_meta_tmp)

    if kwargs['reconstruct_graph']:
        print('Plotting reconstruction graphs...')
        reconstruction_graph(input_tmp, pred, split)

    return pred, pred_loss

def main():
    
    print('Loading in autoencoder model...')
    autoencoder = tf.keras.models.load_model('/home/ricky/RNNAE/CNN_product/CNN_autoencoder_model')
    print('Autoencoder finished loading')

    print('Loading in encoder model...')
    encoder = tf.keras.models.load_model('/home/ricky/RNNAE/CNN_product/CNN_encoder_model')
    print('Encoder finished loading')

    os.chdir('/home/ricky/RNNAE/CNN_product/CNN_npy')
    input = import_data('input')
    input_train = import_data('input_train')
    input_test = import_data('input_test')
    input_meta = import_data('input_meta')
    input_meta_train = import_data('input_meta_train')
    input_meta_test = import_data('input_meta_test')
    type_train = import_data('type_train')
    type_test = import_data('type_test')

    cnn_predict(autoencoder, input_test[0], input_meta_test[0], reconstruct_graph=False, training_data=False)

    latent_space = encoder.predict([input[0], input_meta[0]], verbose=1)
    print(latent_space.shape)

    anomaly_id = isolation_forest(latent_space, 10000, 0)
    latent_space_graph(latent_space, anomaly_id, split=0)

    '''latent_space_concatentate = latent_space_concatenation(latent_space, ['t_normalised_noise', 'no_near_peak'], split=0)

    anomaly_id = isolation_forest(latent_space_concatentate, 10000, 0)
    latent_space_graph(latent_space_concatentate, anomaly_id, split=0)'''

    print('End of CNN_predict.py')

if __name__ == '__main__':
    main()