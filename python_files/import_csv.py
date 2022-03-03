import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.contrib import tenumerate

def avoid_non_SNIa(ii, csv_data_meta):
    
    SN_type = csv_data_meta.true_target[ii]

    if SN_type == 90:
        return True
    else:
        return False

def avoid_empty_SN(ii, oid, oid_count_elem, filters, 
    csv_data, csv_data_meta,
    num=10, lc_length_prepeak=-250, lc_length_postpeak=250):

    t = [ [] for i in filters]
    m_app = [ [] for i in filters]

    band = csv_data.passband[ii:ii+oid_count_elem]
    
    N_specified_bands = 0

    #print('index', ii)
    
    for i, f in enumerate(filters):

        band_id = np.where(band == f)

        t[i] = np.array(csv_data.mjd.iloc[band_id[0]])
        m_app[i] = np.array(csv_data.flux.iloc[band_id[0]])

    t_max = csv_data_meta.true_peakmjd[ii]
    #print(t_max)

    for i in range(len(filters)):
        t_duration = np.where((t[i] > (t_max + lc_length_prepeak)) & (t[i] < (t_max + lc_length_postpeak)))
        if t_duration[0].shape[0] == 0:
                return False
        else:
            N_specified_bands += t_duration[0].shape[0]

    #print(N_specified_bands)
    if N_specified_bands > num:
        try:
            csv_data_meta.true_distmod[ii]
            csv_data_meta.true_z[ii]
            return True
        except Exception:
            return False
    else:
        #print('tiny', ii)
        return False

class LC_Preprocess:

    def __init__(self, ii, oid, oid_count_elem, filters, 
        csv_data, csv_data_meta):
        
        self.ii = ii
        self.oid = oid
        self.oid_count_elem = oid_count_elem

        self.filters = filters

        self.data = csv_data
        self.data_meta = csv_data_meta

        self.band = self.data.passband[ii:ii+oid_count_elem]
        self.claimedtype = 0

        self.t = [ [] for filter in self.filters]
        self.flux = [ [] for filter in self.filters]
        self.flux_err = [ [] for filter in self.filters]
        self.m = [ [] for filter in self.filters]
        self.m_err = [ [] for filter in self.filters]

    def peak_alignment(self, lc_length_prepeak=-200, lc_length_postpeak=200):

        self.t_peak = self.data_meta.true_peakmjd[self.ii]

        for i, f in enumerate(self.filters):

            self.t[i] = np.array(self.t[i]) - self.t_peak

            self.t[i]     = np.delete(self.t[i], np.where(self.t[i] > lc_length_postpeak))
            self.m[i]     = self.m[i][0:len(self.t[i])]
            self.m_err[i] = self.m_err[i][0:len(self.t[i])]

            self.t[i]     = np.delete(self.t[i], np.where(self.t[i] < lc_length_prepeak))
            self.m[i]     = self.m[i][len(self.m[i]) - len(self.t[i]):]
            self.m_err[i] = self.m_err[i][len(self.m_err[i]) - len(self.t[i]):]

            if (len(self.t[i]) - len(self.m[i])) != 0:
                print('bruh')

        return self.t, self.m, self.m_err, self.claimedtype

    def lc_graph(self, colors = ['darkcyan', 'limegreen', 'crimson'], lc_length_prepeak=-200, lc_length_postpeak=200):
        
        colors = ['indigo','darkcyan', 'limegreen', 'darkorange', 'crimson', 'maroon']

        plt.plot(figsize=(16,12))

        for i, filter in enumerate(self.filters):
            plt.errorbar(self.t[i], self.m[i], self.m_err[i], label=filter, color=colors[i], fmt='.')
        
        plt.title(f'{self.oid}, {self.claimedtype}')
        plt.xlim(lc_length_prepeak, lc_length_postpeak)
        #plt.ylim(-23, -14)
        plt.xlabel('time (day)')
        #plt.ylabel('absolute magnitude')
        plt.ylabel('flux')
        plt.legend()
        plt.grid()
        #plt.gca().invert_yaxis()
        plt.savefig(f'/home/ricky/RNNAE/import_graph/{self.oid}.pdf')
        #plt.savefig(fr'C:\\Users\\ricky\\FYP\\RNNAE_public\\import_graph\\{self.SN_name}.pdf')
        plt.close()

    def lc_extractor(self, **kwargs):
        
        dist_mod = float(self.data_meta.true_distmod[self.ii])

        z = float(self.data_meta.true_z[self.ii])

        self.claimedtype = self.data_meta.true_target[self.ii]

        f_min = 0

        for i, f in enumerate(self.filters):

            band_id = np.where(self.band == f)

            self.t[i] = np.array(self.data.mjd.iloc[band_id[0]])
            self.flux[i] = np.array(self.data.flux.iloc[band_id[0]])
            self.flux_err[i] = np.array(self.data.flux_err.iloc[band_id[0]])
            
            if np.min(self.flux[i]) < f_min:
                f_min = np.min(self.flux[i])

        for i, f in enumerate(self.filters):

            self.flux[i] -= f_min
            self.flux[i] = np.where(self.flux[i] == 0, 1e-6, self.flux[i])
            self.m[i] = -2.5*np.log10(self.flux[i]) - dist_mod + 2.5*np.log10(1+z) + 27.5
            self.flux[i] += f_min
            self.m_err[i] = 2.5*0.434*(np.absolute(self.flux_err[i]/(self.flux[i])))

        if kwargs['peak_alignment']:
            LC_Preprocess.peak_alignment(self)

        if kwargs['LC_graph']:
            LC_Preprocess.lc_graph(self)

        return self.t, self.m, self.m_err, self.claimedtype, self.oid

def main():

    os.chdir('/home/ricky/RNNAE/plasticc_csv')
    print(os.getcwd())

    req_cols_meta = ['object_id', 'ddf_bool','target',
       'true_target', 'true_submodel', 'true_z', 'true_distmod',
       'true_lensdmu', 'true_peakmjd',
        ]

    req_cols = ['object_id', 'mjd', 'passband', 'flux', 'flux_err']

    print('Loading in csv ...')
    data_meta = pd.read_csv('plasticc_test_metadata.csv', usecols=req_cols_meta, low_memory=False)
    data = pd.read_csv('plasticc_test_lightcurves_02.csv', usecols=req_cols, low_memory=False)
    print('Finished loading csv')

    min = np.min(data.object_id)
    max = np.max(data.object_id)

    min_id = list(data_meta[data_meta.object_id == min].object_id.index)[0]
    max_id = list(data_meta[data_meta.object_id == max].object_id.index)[0]

    value, counts = np.unique(data.object_id, return_counts=True)

    t_all = []
    m_all = []
    m_err_all = []
    claimedtype_all = []
    SN_name_all = []

    filters_all = [0, 1, 2, 3, 4, 5]
    num_extracted_SN = 0

    print('Screening and extracting SNe ...')
    
    for ii, oid in tenumerate(data_meta.object_id[min_id:max_id]):

        csv_QC1 = avoid_non_SNIa(ii, data_meta)
        csv_QC2 = avoid_empty_SN(ii, oid, counts[ii], filters_all, 
                    data, data_meta,
                    num=10, lc_length_prepeak=-200, lc_length_postpeak=200)
            
        if (csv_QC1 and csv_QC2):
            
            LC_graph_bool = np.random.rand(1) > 0.99

            LC_result = LC_Preprocess(ii, oid, counts[ii], filters_all, data, data_meta).lc_extractor(peak_alignment=True, LC_graph=LC_graph_bool)
            
            t_all.append(LC_result[0])
            m_all.append(LC_result[1])
            m_err_all.append(LC_result[2])
            claimedtype_all.append(LC_result[3])
            SN_name_all.append(LC_result[4])
            
            num_extracted_SN += 1

    os.chdir('/home/ricky/RNNAE/import_npy')
    #os.chdir(r'C:\\Users\\ricky\\FYP\\RNNAE_public\\import_npy')
    print('The current working directory is', os.getcwd())

    np.save('Time_all.npy', np.array(t_all, dtype=object))
    np.save('Magnitude_Abs_all.npy', np.array(m_all, dtype=object))
    np.save('Magnitude_Abs_err_all.npy', np.array(m_err_all, dtype=object))
    np.save('Type_all.npy', np.array(claimedtype_all))
    np.save('SN_name.npy', np.array(SN_name_all))

    print(f'There are {num_extracted_SN} extracted SNe')
    print('End of import.py')

if __name__ == '__main__':
    main()