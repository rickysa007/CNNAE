import os
import numpy as np
from pathlib import Path

def main():

    pp = Path(__file__).parent.parent

    os.chdir(f'{pp}/conv_npy')
    lc = np.array(np.load('lc.npy', allow_pickle=True))
    lc_meta = np.array(np.load('lc_meta.npy', allow_pickle=True))

    #print(lc_meta[0])

    for i in range(lc_meta.shape[0]):

        lc_meta[i]['peak_mag'] = np.min(lc[i,2])*lc_meta[i]['range']+lc_meta[i]['mean'] # find peak mag by r band

        max_id = np.argmin(lc[i,2]) # find peak by r band
        try:
            lc_meta[i]['delta_m'] = lc[i,1][max_id] - lc[i,1][max_id+15] # calculate delta_m by g band
        except:
            lc_meta[i]['delta_m'] = 0

        lc_meta[i]['t_normalised_noise'] = np.sum(lc[i,4] + lc[i,5] + lc[i,6]) / lc_meta[i]['t_len']

    #print(lc_meta[0])

    np.save(f'{pp}/conv_npy/lc.npy', np.array(lc, dtype=object))
    np.save(f'{pp}/conv_npy/lc_meta.npy', np.array(lc_meta, dtype=object))
    #kjhkjh
    print('End of metadata.py')

if __name__ == '__main__':
    main()