import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def pht_conversion(lc_tmp, lc_meta_tmp):

    #Equations from SDSS DR7

    lc_conv_tmp = [ [] for i in range(lc_tmp.shape[0])]

    lc_conv_tmp[0] = lc_tmp[0]
    
    lc_conv_tmp[1] = lc_tmp[1] - (-0.060)*( (lc_tmp[1] - lc_tmp[2]) - 0.53)
    lc_conv_tmp[2] = lc_tmp[2] - (-0.035)*( (lc_tmp[2] - lc_tmp[3]) - 0.21)
    lc_conv_tmp[3] = lc_tmp[3] - (-0.041)*( (lc_tmp[2] - lc_tmp[3]) - 0.21)

    lc_conv_tmp[4] = np.sqrt( (np.square(lc_tmp[4]) - np.square( (-0.060)*np.sqrt( (np.square(lc_tmp[4]) + np.square(lc_tmp[5])).astype(float) ) )).astype(float) )
    lc_conv_tmp[5] = np.sqrt( (np.square(lc_tmp[5]) - np.square( (-0.035)*np.sqrt( (np.square(lc_tmp[5]) + np.square(lc_tmp[6])).astype(float) ) )).astype(float) )
    lc_conv_tmp[6] = np.sqrt( (np.square(lc_tmp[6]) - np.square( (-0.041)*np.sqrt( (np.square(lc_tmp[5]) + np.square(lc_tmp[6])).astype(float) ) )).astype(float) )

    lc_conv_tmp = np.array(lc_conv_tmp)
    lc_conv_tmp = list(lc_conv_tmp[:,:lc_meta_tmp[0]])

    for i in range(len(lc_conv_tmp)):
        for j in range(len(lc_tmp[0]) - lc_meta_tmp[0]):
            lc_conv_tmp[i] = np.append(lc_conv_tmp[i], 0)

    lc_conv_tmp = np.array(lc_conv_tmp)

    return lc_conv_tmp

def comparison_graph(lc_meta_merged, lc_meta_SDSS, lc_meta_SDSS_p, lc_SDSS, lc_SDSS_p, lc_conv):

    seen = set()
    dupe_SN = []

    for i, x in enumerate(lc_meta_merged[:,2]):
        if x in seen:
            dupe_SN.append(x)
        else:
            seen.add(x)

    dupe_SDSS   = [x for x, i in enumerate(lc_meta_SDSS[:,2]) if i in dupe_SN]
    dupe_SDSS_p = [x for x, i in enumerate(lc_meta_SDSS_p[:,2]) if i in dupe_SN]

    # fit peak?
    i=2
    
    diff = np.argmin(lc_conv[dupe_SDSS_p[i]][1]) - np.argmin(lc_SDSS[dupe_SDSS[i]][1])
    print(diff)

    plt.scatter(np.linspace(0, 96, 96)+diff, lc_SDSS[dupe_SDSS[i]][1], s=8, label='SDSS')
    plt.scatter(np.linspace(0, 96, 96), lc_SDSS_p[dupe_SDSS_p[i]][1], s=8, label='SDSS_p')
    plt.scatter(np.linspace(0, 96, 96), lc_conv[dupe_SDSS_p[i]][1], s=8, label='conv')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()

    return

def create_clean_directory(d):

    isExist = os.path.exists(d)
    if isExist:
        shutil.rmtree(d)
        os.makedirs(d)
    else:
        os.makedirs(d)

    return

def main():

    pp = Path(__file__).parent.parent

    os.chdir(f'{pp}/SDSS_GP_npy')

    lc_SDSS = np.array(np.load('lc.npy', allow_pickle=True))
    lc_meta_SDSS = np.array(np.load('lc_meta.npy', allow_pickle=True))

    os.chdir(f'{pp}/SDSS_prime_GP_npy')

    lc_SDSS_p = np.array(np.load('lc.npy', allow_pickle=True))
    lc_meta_SDSS_p = np.array(np.load('lc_meta.npy', allow_pickle=True))

    lc_conv = []

    for i in range(lc_SDSS_p.shape[0]):

        lc_conv_tmp = pht_conversion(lc_SDSS_p[i], lc_meta_SDSS_p[i])

        lc_conv.append(lc_conv_tmp)

    lc_conv = np.array(lc_conv)

    lc_SDSS_merged = np.concatenate((lc_SDSS, lc_conv))
    lc_meta_SDSS_merged = np.concatenate((lc_meta_SDSS, lc_meta_SDSS_p))

    print(f'Shape of light curves after conversion is {lc_SDSS_merged.shape}')

    create_clean_directory(f'{pp}/pht_conv_graph')
    comparison_graph(lc_meta_SDSS_merged, lc_meta_SDSS, lc_meta_SDSS_p, lc_SDSS, lc_SDSS_p, lc_conv)

    create_clean_directory(f'{pp}/conv_npy')
    np.save(f'{pp}/conv_npy/lc.npy', np.array(lc_SDSS_merged, dtype=object))
    np.save(f'{pp}/conv_npy/lc_meta.npy', np.array(lc_meta_SDSS_merged, dtype=object))

    print('End of pht_conv.py')

if __name__ == '__main__':
    main()