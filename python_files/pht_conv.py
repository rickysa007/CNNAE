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

def avoid_duplicate_name(lc_meta_SDSS, lc_meta_SDSS_p):

    lc_meta_merged_tmp = np.concatenate((lc_meta_SDSS, lc_meta_SDSS_p))

    SN_name_merged = [lc_meta_merged_tmp[i]['SN_name'] for i in range(len(lc_meta_merged_tmp))]
    SN_name_SDSS_p = [lc_meta_SDSS_p[i]['SN_name'] for i in range(len(lc_meta_SDSS_p))]

    seen = set()
    dupe_SN = []

    for i, x in enumerate(SN_name_merged):
        if x in seen:
            dupe_SN.append(x)
        else:
            seen.add(x)

    for i, elemi in enumerate(SN_name_SDSS_p):
        for j, elemj in enumerate(dupe_SN):
            if elemi == elemj:
                lc_meta_SDSS_p[i]['SN_name'] = f'{elemi}_prime'

    return lc_meta_SDSS_p, dupe_SN

def comparison_graph(dupe_SN, lc_meta_SDSS, lc_meta_SDSS_p, lc_SDSS, lc_SDSS_p, lc_conv):

    SN_name_SDSS   = [lc_meta_SDSS[i]['SN_name'] for i in range(len(lc_meta_SDSS))]
    SN_name_SDSS_p = [lc_meta_SDSS_p[i]['SN_name'] for i in range(len(lc_meta_SDSS_p))]

    dupe_SN_p = []
    for i, elem in enumerate(dupe_SN):
        dupe_SN_p.append(f'{elem}_prime')

    dupe_SDSS   = [id for id, x in enumerate(SN_name_SDSS) if x in dupe_SN]
    dupe_SDSS_p = [id for id, x in enumerate(SN_name_SDSS_p) if x in dupe_SN_p]

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
            plt.grid()
            plt.legend()
            plt.title(f'{dupe_SN[i]}, {pht_sys[j]}')
            plt.gca().invert_yaxis()
            #plt.show()
            plt.savefig(f'{dupe_SN[i]}_{pht_sys[j]}_band.pdf')
            plt.close()

        os.chdir('..')

    return

def qc_graph(qc):

    diff = []
    var = []
    noise = []
    score = []

    for i in range(len(qc)):
        diff.append(qc[i]['diff'])
        var.append(qc[i]['var'])
        noise.append(qc[i]['noise'])
        score.append(qc[i]['score'])

    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('Histograms of quality indicators', fontsize=15)
    xlabel=[['sum of difference', 'sum of variance'], ['sum of uncertainty', 'peak quality score']]
    
    axs[0][0].hist(diff, bins=2*round(max(diff)))
    axs[0][1].hist(var, bins=40*round(max(var)))
    axs[1][0].hist(noise, bins=round(max(noise)))
    axs[1][1].hist(score, bins=round(max(score)))

    axs[0][0].axvspan(0, 1, alpha=0.2, color='red')
    axs[0][1].axvspan(0, 0.05, alpha=0.2, color='red')
    axs[1][0].axvspan(20, max(noise), alpha=0.2, color='red')
    axs[1][1].axvspan(0, 5, alpha=0.2, color='red')

    axs[0][0].set_xlim(0, 60)
    axs[0][1].set_xlim(0, 1)
    axs[1][0].set_xlim(0, 80)
    axs[1][1].set_xlim(0, 60)

    for i in range(2):
        for j in range(2):
            axs[i][j].set_xlabel(xlabel[i][j], fontsize=12)
            axs[i][j].set_ylabel('Count', fontsize=12)
            axs[i][j].grid()

    plt.savefig('./qc.pdf', bbox_inches='tight')
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
    lc_SDSS      = np.array(np.load('lc.npy', allow_pickle=True))
    lc_meta_SDSS = np.array(np.load('lc_meta.npy', allow_pickle=True))
    lc_qc_SDSS   = np.array(np.load('lc_qc.npy', allow_pickle=True))

    os.chdir(f'{pp}/SDSS_prime_GP_npy')
    lc_SDSS_p      = np.array(np.load('lc.npy', allow_pickle=True))
    lc_meta_SDSS_p = np.array(np.load('lc_meta.npy', allow_pickle=True))
    lc_qc_SDSS_p = np.array(np.load('lc_qc.npy', allow_pickle=True))

    print(lc_SDSS.shape, lc_SDSS_p.shape)

    lc_conv = []

    for i in range(lc_SDSS_p.shape[0]):

        lc_conv_tmp = pht_conversion(lc_SDSS_p[i], lc_meta_SDSS_p[i])

        lc_conv.append(lc_conv_tmp)

    lc_conv = np.array(lc_conv, dtype=object)

    '''for i in range(lc_SDSS.shape[0]):
        if len(lc_SDSS[i][0]) != 96:
            print(f'{i}, {len(lc_SDSS[i][0])}')'''

    lc_SDSS_merged = np.concatenate((lc_SDSS, lc_conv))

    lc_meta_SDSS_p, dupe_SN = avoid_duplicate_name(lc_meta_SDSS, lc_meta_SDSS_p)
    lc_meta_SDSS_merged = np.concatenate((lc_meta_SDSS, lc_meta_SDSS_p))

    lc_qc_SDSS_merged = np.concatenate((lc_qc_SDSS, lc_qc_SDSS_p))
    os.chdir(f'{pp}')
    qc_graph(lc_qc_SDSS_merged)

    print(f'Shape of light curves after conversion is {lc_SDSS_merged.shape}')

    create_clean_directory(f'{pp}/pht_conv_graph')
    os.chdir(f'{pp}/pht_conv_graph')
    comparison_graph(dupe_SN, lc_meta_SDSS, lc_meta_SDSS_p, lc_SDSS, lc_SDSS_p, lc_conv)

    create_clean_directory(f'{pp}/conv_npy')
    np.save(f'{pp}/conv_npy/lc.npy', np.array(lc_SDSS_merged, dtype=object))
    np.save(f'{pp}/conv_npy/lc_meta.npy', np.array(lc_meta_SDSS_merged, dtype=object))

    print('End of pht_conv.py')

if __name__ == '__main__':
    main()