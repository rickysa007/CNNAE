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


os.chdir('/home/ricky/RNNAE')

data_GP = np.array(np.load('data_GP.npy', allow_pickle=True))
data_meta_GP = np.array(np.load('data_meta_GP.npy', allow_pickle=True))

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

    os.chdir('/home/ricky/RNNAE/RNN_latent_space_graph')

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

    shutil.rmtree('/home/ricky/RNNAE/RNN_anomaly_graph')
    os.makedirs('/home/ricky/RNNAE/RNN_anomaly_graph')

    for i, ano in enumerate(anomaly_id):
        shutil.copy(f'/home/ricky/RNNAE/GP_graph/{data_meta_GP[ano+split][-1]}.pdf', f'/home/ricky/RNNAE/RNN_anomaly_graph/{i}_{data_meta_GP[ano+split][-1]}.pdf')

    return

def reconstruction_graph(input_tmp, yhat, split, filters=['u', 'g', 'i']):

    color1 = ['darkviolet', 'seagreen', 'crimson', 'maroon']
    color2 = ['darkmagenta', 'darkgreen', 'firebrick', 'darkred']

    shutil.rmtree('/home/ricky/RNNAE/RNN_reconstruction_graph')
    os.makedirs('/home/ricky/RNNAE/RNN_reconstruction_graph')

    for i in tqdm(range(input_tmp.shape[0])):

        os.chdir('/home/ricky/RNNAE/RNN_reconstruction_graph')

        isExist = os.path.exists(f'./{data_meta_GP[i+split][-1]}')

        if not isExist:
            os.makedirs(f'./{data_meta_GP[i+split][-1]}')
            os.chdir(f'./{data_meta_GP[i+split][-1]}')

        for j, filter in enumerate(filters):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1)

            plt.gca().invert_yaxis()

            # And a corresponding grid
            ax.grid(which='major', alpha=0.8)
            ax.grid(which='minor', alpha=0.3)

            plt.xlabel('Timestep', fontsize=15)
            plt.ylabel('Absolute Magnitude', fontsize=15)

            plt.xlim(-50, 200)

            plt.title(f'{data_meta_GP[i+split][-1]}, {data_meta_GP[i+split][-2]}, {filter}')

            #plt.errorbar(data_GP[i+split][0], data_GP[i+split][j+1], y_err=data_GP[i+split][j+4], fmt='v')

            plt.scatter(input_tmp[i,:data_meta_GP[i+split][0],0], input_tmp[i,:data_meta_GP[i+split][0],j+1], s=2, marker='o', color=color1[j], label=f'test data'.format('o'))
            plt.scatter(input_tmp[i,:data_meta_GP[i+split][0],0], yhat[i,:data_meta_GP[i+split][0],j], s=12, marker='X', color=color2[j], label=f'reconstruction'.format('X'))
            
            plt.legend()

            plt.savefig(f'./{data_meta_GP[i][-1]}_{filter}_band.pdf')

            plt.close()

    return

def rnn_predict(autoencoder, encoder, input_tmp, **kwargs):

    split = int(0.8*(data_GP.shape[0]))

    if kwargs['training_data']:
        split = 0

    yhat = rnnae_test2(autoencoder, input_tmp)
    print('Plotting reconstruction graphs...')
    reconstruction_graph(input_tmp, yhat, split)

    return


def main():
    
    print('Loading in autoencoder model...')
    autoencoder = tf.keras.models.load_model('/home/ricky/RNNAE/RNN_autoencoder_model')
    print('Autoencoder finished loading')

    print('Loading in encoder model...')
    encoder = tf.keras.models.load_model('/home/ricky/RNNAE/RNN_encoder_model')
    print('Encoder finished loading')

    os.chdir('/home/ricky/RNNAE/RNN_npy')

    input = import_data('input')
    input_train = import_data('input_train')
    input_test = import_data('input_test')
    type_train = import_data('type_train')
    type_test = import_data('type_test')

    rnn_predict(autoencoder, encoder, input_test[0], training_data=False)

    latent_space = latent_space_demo(encoder, input_train[0])
    isolation_forest(latent_space, 1000, 0)

    print('End of RNN_predict.py')

if __name__ == '__main__':
    main()