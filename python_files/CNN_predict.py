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
        #t_normalised_noise = "{:.2f}".format(lc_meta[ano+split]['t_normalised_noise'])
        #no_near_peak = "{:.2f}".format(lc_meta[ano+split]['no_near_peak'])

        # naming problem?
        try:
            shutil.copy(f'/home/ricky/RNNAE/SDSS_GP_graph/{name}.pdf', f'/home/ricky/RNNAE/CNN_product/CNN_anomaly_graph/{i+1}_{name}_{peak_mag}_{delta_m}.pdf')
        except OSError:
            try:
                shutil.copy(f'/home/ricky/RNNAE/SDSS_prime_GP_graph/{name}.pdf', f'/home/ricky/RNNAE/CNN_product/CNN_anomaly_graph/{i+1}_{name}_{peak_mag}_{delta_m}.pdf')
            except OSError:
                name = name.replace('_prime','')
                shutil.copy(f'/home/ricky/RNNAE/SDSS_prime_GP_graph/{name}.pdf', f'/home/ricky/RNNAE/CNN_product/CNN_anomaly_graph/{i+1}_{name}_{peak_mag}_{delta_m}.pdf')
    
    return anomaly_id

def cdf(anomaly_id, split=0):

    os.chdir('/home/ricky/RNNAE/CNN_product')

    rank_normalIa = []
    rank_peculiarIa = []

    for i, ano in enumerate(anomaly_id):
        if lc_meta[ano+split]['type'] == 'Ia':
            rank_normalIa.append(i)
        else:
            rank_peculiarIa.append(i)

    count, bins_count = np.histogram(rank_normalIa, bins=len(anomaly_id))
    pdf = count/sum(count)
    cdf_n = np.cumsum(pdf)

    count, bins_count = np.histogram(rank_peculiarIa, bins=len(anomaly_id))
    pdf = count/sum(count)
    cdf_p = np.cumsum(pdf)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(bins_count[1:], cdf_n, label="Normal SNeIa")
    plt.plot(bins_count[1:], cdf_p, label="Subtype SNeIa")

    plt.xlabel('Anomaly Ranking', fontsize=15)
    plt.ylabel('CDF', fontsize=15)
    plt.title('CDF of anomaly ranking')

    plt.grid()
    plt.legend()

    plt.savefig('cdf.pdf', bbox_inches='tight')
    plt.close()

    return

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
            
            if i == latent_space.shape[1]-2:
                plt.xlabel('\u0394 M_15', fontsize=12)
            else:
                plt.xlabel(f'id {i+1}', fontsize=12) # addition + 1 for readability
            
            if i+j+1 == latent_space.shape[1]-2:
                plt.ylabel('\u0394 M_15', fontsize=12)
            elif i+j+1 == latent_space.shape[1]-1:
                plt.gca().invert_yaxis()
                plt.ylabel('g band peak magnitude', fontsize=12)
            else:
                plt.ylabel(f'id {i+j+1+1}', fontsize=12) # addition + 1 for readability
            
            plt.title(f'latent space id {i+1} vs id {i+j+1+1}') # addition + 1 for readability
            plt.colorbar(label='anomaly ranking')
            
            plt.savefig(f'id_{i+1}_vs_id_{i+j+1+1}.pdf', bbox_inches='tight') # addition + 1 for readability
            plt.close()

    return

def reconstruction_graph(input_tmp, pred, split, filters=['g', 'r', 'i']):

    color1 = ['mediumturquoise', 'crimson', 'maroon']
    color2 = ['lightseagreen', 'firebrick', 'darkred']

    d = '/home/ricky/RNNAE/CNN_product/CNN_reconstruction_graph'
    create_clean_directory(d)

    for i in tqdm(range(input_tmp.shape[0])):

        os.chdir(d)

        isExist = os.path.exists(f'./{lc_meta[i+split]["SN_name"]}')

        if not isExist:
            os.makedirs(f'./{lc_meta[i+split]["SN_name"]}')
            os.chdir(f'./{lc_meta[i+split]["SN_name"]}')

        fig, axs = plt.subplots(3, figsize=(8, 16))

        fig.suptitle('Images of CNN')
        axs[0].set_title('Input Test Image')
        axs[1].set_title('Reconstructed Test Image')
        axs[2].set_title('Difference between Images')

        a = []
        a.append(axs[0].imshow(input_tmp[i].reshape(96,96).T, interpolation='nearest', aspect='auto', cmap='BrBG'))
        a.append(axs[1].imshow(pred[i].reshape(96,96).T, interpolation='nearest', aspect='auto', cmap='BrBG'))
        a.append(axs[2].imshow((input_tmp[i] - pred[i]).reshape(96,96).T, interpolation='nearest', aspect='auto', cmap='BrBG'))

        for j in range(3):
            plt.colorbar(a[j], ax=axs[j]).set_label('Normalized Absolute Magnitude')
            axs[j].set_xlabel('Timestep')

        fig.savefig(f'./{lc_meta[i+split]["SN_name"]}.pdf', bbox_inches='tight')
        plt.close()

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{lc_meta[i+split]["SN_name"]}, type {lc_meta[i+split]["type"]}', fontsize=15)

        for j, filter in enumerate(filters):

            axs[j].set_xlabel('Timestep', fontsize=12)
            axs[j].set_ylabel('Normalized Absolute Magnitude', fontsize=12)

            axs[j].invert_yaxis()

            axs[j].set_title(f'{filter} band', fontsize=12)

            axs[j].scatter(np.linspace(0, 96, 96), input_tmp[i][:,j,:], s=4, marker='o', color=color1[j], label=f'test data'.format('o'))
            axs[j].scatter(np.linspace(0, 96, 96), pred[i][:,j,:], s=16, marker='X', color=color2[j], label=f'reconstruction'.format('X'))
            
            axs[j].grid()
            axs[j].legend()

        plt.savefig(f'./{lc_meta[i+split]["SN_name"]}_lc.pdf', bbox_inches='tight')
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

    anomaly_id = isolation_forest(latent_space, 5000, 0)
    cdf(anomaly_id, 0)
    
    latent_space_graph(latent_space, anomaly_id, split=0)

    '''latent_space_concatentate = latent_space_concatenation(latent_space, ['t_normalised_noise', 'no_near_peak'], split=0)

    anomaly_id = isolation_forest(latent_space_concatentate, 1000, 0)
    latent_space_graph(latent_space_concatentate, anomaly_id, split=0)'''

    print('End of CNN_predict.py')

if __name__ == '__main__':
    main()