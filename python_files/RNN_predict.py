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

def rnnae_test(autoencoder, input_tmp, mask_tmp):

    yhat = autoencoder.predict(x=[input_tmp, mask_tmp], verbose=1)
    yhat_loss = autoencoder.evaluate(x=[input_tmp, mask_tmp], y=None, verbose=1)

    return yhat

def rnnae_test2(autoencoder, input_tmp):

    yhat = autoencoder.predict(x=input_tmp, verbose=1)
    yhat_loss = autoencoder.evaluate(x=input_tmp, y=None, verbose=1)

    return yhat

def latent_space_demo(encoder, input_tmp):

    latent_space = encoder.predict(input_tmp, verbose=1)

    os.chdir('/home/ricky/RNNAE/RNN_product/RNN_latent_space_graph')

    for i in range(latent_space.shape[1] - 1):
        for j in range(latent_space.shape[1] - 1 - i):
            fig = plt.figure(figsize=(6, 6))
            plt.grid()
            plt.scatter(latent_space[:,i], latent_space[:,i+j+1], s=8)
            plt.title(f'id {i} vs id {i+j+1}.pdf')
            plt.savefig(f'id_{i}_vs_id_{i+j+1}.pdf')
            plt.close()

    return latent_space

def isolation_forest(latent_space, n_tree, split):

    clf = IsolationForest(n_estimators=n_tree, warm_start=True)
    clf.fit(latent_space)
    anomaly = clf.score_samples(latent_space)
    anomaly_id = np.argsort(anomaly)

    shutil.rmtree('/home/ricky/RNNAE/RNN_product/RNN_anomaly_graph')
    os.makedirs('/home/ricky/RNNAE/RNN_product/RNN_anomaly_graph')

    for i, ano in enumerate(anomaly_id):
        shutil.copy(f'/home/ricky/RNNAE/GP_graph/{lc_meta[ano+split][-1]}.pdf', f'/home/ricky/RNNAE/RNN_anomaly_graph/{i}_{lc_meta[ano+split][-1]}.pdf')

    return

def reconstruction_graph(input_tmp, yhat, split, filters=['g', 'r', 'i']):

    color1 = ['mediumturquoise', 'crimson', 'maroon']
    color2 = ['lightseagreen', 'firebrick', 'darkred']

    shutil.rmtree('/home/ricky/RNNAE/RNN_product/RNN_reconstruction_graph')
    os.makedirs('/home/ricky/RNNAE/RNN_product/RNN_reconstruction_graph')

    for i in tqdm(range(input_tmp.shape[0])):

        os.chdir('/home/ricky/RNNAE/RNN_product/RNN_reconstruction_graph')

        isExist = os.path.exists(f'./{lc_meta[i+split]["SN_name"]}')

        if not isExist:
            os.makedirs(f'./{lc_meta[i+split]["SN_name"]}')
            os.chdir(f'./{lc_meta[i+split]["SN_name"]}')

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{lc_meta[i+split]["SN_name"]}, type {lc_meta[i+split]["type"]}', fontsize=15)

        input_time = np.linspace(input_tmp[i,0,0], input_tmp[i,-1,0], 96)

        for j, filter in enumerate(filters):

            axs[j].set_xlabel('Timestep', fontsize=12)
            axs[j].set_ylabel('Normalized Absolute Magnitude', fontsize=12)

            axs[j].invert_yaxis()

            axs[j].set_title(f'{filter} band', fontsize=12)

            #plt.errorbar(data_GP[i+split][0], data_GP[i+split][j+1], y_err=data_GP[i+split][j+4], fmt='v')

            axs[j].scatter(input_time, input_tmp[i,:,j+1], s=4, marker='o', color=color1[j], label=f'test data'.format('o'))
            axs[j].scatter(input_time, yhat[i,:,j], s=20, marker='X', color=color2[j], label=f'reconstruction'.format('X'))
            
            axs[j].grid()
            axs[j].legend()

        plt.savefig(f'./{lc_meta[i+split]["SN_name"]}.pdf', bbox_inches='tight')
        plt.close()

    return

def rnn_predict(autoencoder, encoder, input_tmp, **kwargs):

    split = int(0.8*(lc.shape[0]))

    if kwargs['training_data']:
        split = 0

    yhat = rnnae_test2(autoencoder, input_tmp)
    print('Plotting reconstruction graphs...')
    reconstruction_graph(input_tmp, yhat, split)

    return


def main():
    
    print('Loading in autoencoder model...')
    autoencoder = tf.keras.models.load_model('/home/ricky/RNNAE/RNN_product/RNN_autoencoder_model')
    print('Autoencoder finished loading')

    print('Loading in encoder model...')
    encoder = tf.keras.models.load_model('/home/ricky/RNNAE/RNN_product/RNN_encoder_model')
    print('Encoder finished loading')

    os.chdir('/home/ricky/RNNAE/RNN_product/RNN_npy')

    input = import_data('input')
    input_train = import_data('input_train')
    input_test = import_data('input_test')
    type_train = import_data('type_train')
    type_test = import_data('type_test')

    rnn_predict(autoencoder, encoder, input_test[0], training_data=False)

    '''latent_space = latent_space_demo(encoder, input_train[0])
    isolation_forest(latent_space, 1000, 0)'''

    print('End of RNN_predict.py')

if __name__ == '__main__':
    main()