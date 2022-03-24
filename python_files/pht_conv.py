import itertools
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def pht_conversion(lc_tmp, lc_meta_tmp):

    #Equations from SDSS DR7

    lc_conv_tmp = [ [] for i in range(lc_tmp.shape[0])]

    lc_conv_tmp[0] = lc_tmp[0]

    #unnormalize lc
    for i in range(lc_tmp.shape[0] - 4):
        lc_conv_tmp[i+1] = lc_tmp[i+1]*lc_meta_tmp['range'] + lc_meta_tmp['mean']
        lc_conv_tmp[i+4] = lc_tmp[i+4]*lc_meta_tmp['range']
    
    lc_conv_tmp[1] = lc_tmp[1] - (-0.060)*( (lc_tmp[1] - lc_tmp[2]) - 0.53)
    lc_conv_tmp[2] = lc_tmp[2] - (-0.035)*( (lc_tmp[2] - lc_tmp[3]) - 0.21)
    lc_conv_tmp[3] = lc_tmp[3] - (-0.041)*( (lc_tmp[2] - lc_tmp[3]) - 0.21)

    lc_conv_tmp[4] = np.sqrt( (np.square(lc_tmp[4]) - np.square( (-0.060)*np.sqrt( (np.square(lc_tmp[4]) + np.square(lc_tmp[5])).astype(float) ) )).astype(float) )
    lc_conv_tmp[5] = np.sqrt( (np.square(lc_tmp[5]) - np.square( (-0.035)*np.sqrt( (np.square(lc_tmp[5]) + np.square(lc_tmp[6])).astype(float) ) )).astype(float) )
    lc_conv_tmp[6] = np.sqrt( (np.square(lc_tmp[6]) - np.square( (-0.041)*np.sqrt( (np.square(lc_tmp[5]) + np.square(lc_tmp[6])).astype(float) ) )).astype(float) )

    #renormalize lc
    mean_tmp = np.mean(list(itertools.chain(lc_conv_tmp[1], lc_conv_tmp[2], lc_conv_tmp[3])))
    range_tmp = np.abs(np.max(list(itertools.chain(lc_conv_tmp[1], lc_conv_tmp[2], lc_conv_tmp[3]))) - np.min(list(itertools.chain(lc_conv_tmp[1], lc_conv_tmp[2], lc_conv_tmp[3]))))

    for i in range(lc_tmp.shape[0] - 4):
        lc_conv_tmp[i+1] = (lc_conv_tmp[i+1] - mean_tmp) / range_tmp
        lc_conv_tmp[i+4] = lc_conv_tmp[i+4]*range_tmp

    lc_conv_tmp = np.array(lc_conv_tmp)
    lc_conv_tmp = list(lc_conv_tmp[:,:lc_meta_tmp['t_len']])

    #reappend time and 0 magnitude at the tail
    for i in range(len(lc_tmp[0]) - lc_meta_tmp['t_len']):
            lc_conv_tmp[0] = np.append(lc_conv_tmp[0], lc_conv_tmp[0][-1] + 1)

    for i in range(len(lc_conv_tmp) - 1):
        for j in range(len(lc_tmp[0]) - lc_meta_tmp['t_len']):
            lc_conv_tmp[i+1] = np.append(lc_conv_tmp[i+1], 0)

    #lc_conv_tmp = np.array(lc_conv_tmp)

    return lc_conv_tmp

def comparison_graph(lc_meta_merged, lc_meta_SDSS, lc_meta_SDSS_p, lc_SDSS, lc_SDSS_p, lc_conv, t, m, m_err):

    SN_name_merged = [lc_meta_merged[i]['SN_name'] for i in range(len(lc_meta_merged))]
    SN_name_SDSS = [lc_meta_SDSS[i]['SN_name'] for i in range(len(lc_meta_SDSS))]
    SN_name_SDSS_p = [lc_meta_SDSS_p[i]['SN_name'] for i in range(len(lc_meta_SDSS_p))]

    seen = set()
    dupe_SN = []

    for i, x in enumerate(SN_name_merged):
        if x in seen:
            dupe_SN.append(x)
        else:
            seen.add(x)

    dupe_SDSS   = [id for id, x in enumerate(SN_name_SDSS) if x in dupe_SN]
    dupe_SDSS_p = [id for id, x in enumerate(SN_name_SDSS_p) if x in dupe_SN]
    print(dupe_SDSS)
    print(dupe_SN)

    for i in range(len(dupe_SN)):

        os.makedirs(f'./{dupe_SN[i]}')
        os.chdir(f'./{dupe_SN[i]}')

        pht_sys = ['g', 'r', 'i']
        colors = ['lightseagreen', 'crimson', 'darkred']

        for j in range((lc_SDSS.shape[1]-1)//2):
            fig = plt.figure(figsize=(12, 9))

            plt.scatter(lc_SDSS[dupe_SDSS[i]][0], lc_SDSS[dupe_SDSS[i]][j+1], marker='x', s=40, label='SDSS', color=colors[j])
            plt.scatter(lc_SDSS_p[dupe_SDSS_p[i]][0], lc_SDSS_p[dupe_SDSS_p[i]][j+1], s=20, label='SDSS_p', color=colors[j])
            plt.scatter(lc_conv[dupe_SDSS_p[i]][0], lc_conv[dupe_SDSS_p[i]][j+1], marker='v', s=40, label='converted', color=colors[j])
            #plt.errorbar(t[dupe_SDSS[i]][j], m[dupe_SDSS[i]][j], m_err[dupe_SDSS[i]][j], label='SDSS actual observation', color=colors[j])

            plt.legend()
            plt.gca().invert_yaxis()
            plt.grid()
            plt.title(f'{dupe_SN[i]}, {pht_sys[j]}')

            plt.show()
            plt.savefig(f'{dupe_SN[i]}_{pht_sys[j]}_band.pdf')
            plt.close()

        os.chdir('..')

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

    print(lc_SDSS.shape, lc_SDSS_p.shape)

    os.chdir(f'{pp}/SDSS_import_npy')
    t_SDSS = np.load('Time_all.npy', allow_pickle=True)
    m_SDSS = np.load('Magnitude_Abs_all.npy', allow_pickle=True)
    m_err_SDSS = np.load('Magnitude_Abs_err_all.npy', allow_pickle=True)

    lc_conv = []

    for i in range(lc_SDSS_p.shape[0]):

        lc_conv_tmp = pht_conversion(lc_SDSS_p[i], lc_meta_SDSS_p[i])

        lc_conv.append(lc_conv_tmp)

    lc_conv = np.array(lc_conv)

    for i in range(lc_SDSS.shape[0]):
        if len(lc_SDSS[i][0]) != 96:
            print(f'{i}, {len(lc_SDSS[i][0])}')

    lc_SDSS_merged = np.concatenate((lc_SDSS, lc_conv))
    lc_meta_SDSS_merged = np.concatenate((lc_meta_SDSS, lc_meta_SDSS_p))

    print(f'Shape of light curves after conversion is {lc_SDSS_merged.shape}')

    create_clean_directory(f'{pp}/pht_conv_graph')
    os.chdir(f'{pp}/pht_conv_graph')
    comparison_graph(lc_meta_SDSS_merged, lc_meta_SDSS, lc_meta_SDSS_p, lc_SDSS, lc_SDSS_p, lc_conv, t_SDSS, m_SDSS, m_err_SDSS)

    create_clean_directory(f'{pp}/conv_npy')
    np.save(f'{pp}/conv_npy/lc.npy', np.array(lc_SDSS_merged, dtype=object))
    np.save(f'{pp}/conv_npy/lc_meta.npy', np.array(lc_meta_SDSS_merged, dtype=object))

    print('End of pht_conv.py')

if __name__ == '__main__':
    main()