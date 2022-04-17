import os
import subprocess as sp
import silence_tensorflow.auto # pylint: disable=unused-import
import matplotlib.pyplot as plt
import numpy as np
import shutil
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

from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import IsolationForest

pp = Path(__file__).parent.parent

os.chdir(f'{pp}/CNN_product/CNN_npy')
input = np.array(np.load('input.npy', allow_pickle=True))
input = np.asarray(input).astype('float32')
input_meta = np.array(np.load('input_meta.npy', allow_pickle=True))
input_meta = np.asarray(input_meta).astype('float32')

os.chdir(f'{pp}/conv_npy')
lc_meta = np.array(np.load('lc_meta.npy', allow_pickle=True))

def create_clean_directory(d):

    isExist = os.path.exists(d)
    if isExist:
        shutil.rmtree(d)
        os.makedirs(d)
    else:
        os.makedirs(d)

    return

def compare_pop(a: list, b: list, c: np.ndarray):

    #Return c with c[pop_id] removed

    b_set = set(b)

    pop_id = [i for i, item in enumerate(a) if item in b_set]

    return np.delete(c, pop_id, axis=1)

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

    d = '/home/ricky/RNNAE/CNN_product/human_anomaly_graph'

    create_clean_directory(d)

    for i, ano in enumerate(anomaly_id):
        name = lc_meta[ano+split]['SN_name']
        peak_mag = "{:.2f}".format(lc_meta[ano+split]['peak_mag'])
        delta_m = "{:.2f}".format(lc_meta[ano+split]['delta_m'])
        t_normalised_noise = "{:.2f}".format(lc_meta[ano+split]['t_normalised_noise'])
        no_near_peak = "{:.2f}".format(lc_meta[ano+split]['no_near_peak'])

        try:
            shutil.copy(f'/home/ricky/RNNAE/SDSS_GP_graph/{name}.pdf', f'/home/ricky/RNNAE/CNN_product/human_anomaly_graph/{i}_{name}_{peak_mag}_{delta_m}_{t_normalised_noise}_{no_near_peak}.pdf')
        except:
            shutil.copy(f'/home/ricky/RNNAE/SDSS_prime_GP_graph/{name}.pdf', f'/home/ricky/RNNAE/CNN_product/human_anomaly_graph/{i}_{name}_{peak_mag}_{delta_m}_{t_normalised_noise}_{no_near_peak}.pdf')

    return anomaly_id

def latent_space_graph(latent_space, anomaly_id, split):

    d = '/home/ricky/RNNAE/CNN_product/human_latent_space_graph'
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

def main():

    # SOMEHOW ENCDOER SOMETIMES HAS WARNING???
    print('Loading in encoder model...')
    encoder = tf.keras.models.load_model('/home/ricky/RNNAE/CNN_product/CNN_encoder_model')
    print('Encoder finished loading')

    with open(f'{pp}/CNN_product/human_IF/exclusion.txt') as file:
        exclusion = [line.rstrip() for line in file]
    
    SN_name = [lc_meta[i]['SN_name'] for i in range(lc_meta.shape[0])]

    input_pop = compare_pop(exclusion, SN_name, input)
    print(input_pop.shape)

    input_meta_pop = compare_pop(exclusion, SN_name, input_meta)

    latent_space = encoder.predict([input_pop[0], input_meta_pop[0]], verbose=1)

    latent_space_concatenate = latent_space_concatenation(latent_space, ['delta_m'], split=0)
    anomaly_id = isolation_forest(latent_space_concatenate, 10000, 0)
    latent_space_graph(latent_space_concatenate, anomaly_id, split=0)

if __name__ == '__main__':
    main()