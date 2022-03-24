import os
import shutil
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import sqrt, log10
from tqdm.contrib import tenumerate

class QC:

    '''
    Class to screens out the SNe, notice functions in this class are designed to handle one SN
    
    Input
        filename: string
            The filename of the .json of the SNe
        
        filters: 1D list of string
            Name of each band of a particular filters system in use
        
        json_data_temp: .json file
            .json file of the current SN deciding to be screened or not
    '''


    def __init__(self, filename, filters, json_data_temp):

        self.filename = filename

        self.SN_name = self.filename.replace('.json', '')
        self.SN_name = self.SN_name.replace('_', ':')
        
        self.filters = filters
        self.json_data_temp = json_data_temp

        self.band = [] # The band found in the .json file

        self.t = [ [] for filter in self.filters]
        self.m_app = [ [] for filter in self.filters]


    def avoid_non_SNIa(self):

        '''
        Avoid SN that is not SNIa

        Return
            self.filename_QC: Boolean
                Status of the SN, True means to be selected and False means not to be selected
        '''
        
        try:
            self.SN_type = self.json_data_temp[self.SN_name]['claimedtype'][0]['value'] # The claimed type of the SN if any
        except Exception:
            self.SN_type = ''

        if 'Ia' in self.SN_type: # Select for SN containing the string 'Ia'
            return True 
        else:
            return False
    

    def avoid_empty_SN(self, num=20, lc_len_prepeak=-24, lc_len_postpeak=72):

        '''
        Avoid SN that does not contain more than 40 data points at the specified band

        Return
            self.filename_QC: Boolean
                Status of the SN, True means to be selected and False means not to be selected
        '''
        
        try:
            self.N = len(self.json_data_temp[self.SN_name]['photometry']) # The no. of data point of photometry in each SN
        except Exception:
            self.N = 1

        for i in range(self.N): # Loop through all photemetry data points in one SN

            # Avoid any data point without band data
            try:
                self.band.append(self.json_data_temp[self.SN_name]['photometry'][i]['band'])
            except Exception:
                self.band.append(0)

        self.N_specified_bands = 0 # Number of data points at the specified bands

        if all(elem in self.band for elem in self.filters):

            for i in range(self.N):
                for j, filter in enumerate(self.filters):
                    if self.band[i] == filter:

                        self.m_app[j].append(float(self.json_data_temp[self.SN_name]['photometry'][i]['magnitude']))

                        self.t[j].append(float(self.json_data_temp[self.SN_name]['photometry'][i]['time']))

            self.m_max_id = np.argmin(self.m_app[1]) # Choosing r band
            self.t_max = self.t[1][self.m_max_id]

            for i in range(len(self.filters)):
                self.t_duration = [j for j in self.t[i] if (j > (self.t_max + lc_len_prepeak)) and (j < (self.t_max + lc_len_postpeak))]
                if len(self.t_duration) == 0:
                    return False
                else:
                    self.N_specified_bands += len(self.t_duration)
        
        if self.N_specified_bands > num: # Select for SN with more than num=40 data points
            try:
                float(self.json_data_temp[self.SN_name]['lumdist'][0]['value'])
                float(self.json_data_temp[self.SN_name]['redshift'][0]['value'])
                return True
            except Exception:
                return False
        else:
            return False


class LC_Preprocess:


    def __init__(self, filename, filters, json_data, lc_len_prepeak=-24, lc_len_postpeak=72):

        self.filename = filename

        self.SN_name = self.filename.replace('.json', '')
        self.SN_name = self.SN_name.replace('_', ':')
        
        self.filters = filters
        self.json_data = json_data

        self.lc_len_prepeak = lc_len_prepeak
        self.lc_len_postpeak = lc_len_postpeak

        self.band = []
        self.claimedtype = 0

        self.t = [ [] for filter in self.filters]
        self.m = [ [] for filter in self.filters]
        self.m_err = [ [] for filter in self.filters]


    def lc_truncation(self, peak_alignment):

        self.m_max_id = np.argmin(self.m[1]) # Finding maximum by r band
        self.t_max = self.t[1][self.m_max_id]

        if peak_alignment is True:

            for i, filter in enumerate(self.filters):
                self.t[i] = np.array(self.t[i]) - self.t_max

            self.t_max = 0

        else:
            for i, filter in enumerate(self.filters):
                self.t[i] = np.array(self.t[i])

        self.lc_len_prepeak  += self.t_max
        self.lc_len_postpeak += self.t_max

        for i, filter in enumerate(self.filters):

            self.t[i]     = np.delete(self.t[i], np.where(self.t[i] > self.lc_len_postpeak))
            self.m[i]     = self.m[i][0:len(self.t[i])]
            self.m_err[i] = self.m_err[i][0:len(self.t[i])]

            self.t[i]     = np.delete(self.t[i], np.where(self.t[i] < self.lc_len_prepeak))
            self.m[i]     = self.m[i][len(self.m[i]) - len(self.t[i]):]
            self.m_err[i] = self.m_err[i][len(self.m_err[i]) - len(self.t[i]):]

            if (len(self.t[i]) - len(self.m[i])) != 0:
                print('unmatch length between time and magnitude!!!')

        return self.t, self.m, self.m_err, self.claimedtype


    def lc_graph(self, colors = ['lightseagreen', 'crimson', 'darkred']):

        #colors = ['blue', 'green', 'red', 'maroon']

        plt.plot(figsize=(16,12))

        for i, filter in enumerate(self.filters):
            plt.errorbar(np.array(self.t[i]), np.array(self.m[i]), np.array(self.m_err[i]), label=filter, color=colors[i], fmt='.')
        
        plt.title(f'{self.SN_name}, {self.claimedtype}')
        plt.xlim(self.lc_len_prepeak, self.lc_len_postpeak)
        plt.xlabel('time (day)')
        plt.ylabel('absolute magnitude')
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()
        plt.savefig(f'./{self.SN_name}.pdf')
        plt.clf()


    def lc_extractor(self, **kwargs):

        self.lum_dist = float(self.json_data[self.SN_name]['lumdist'][0]['value'])

        try:
            self.lum_dist_err = float(self.json_data[self.SN_name]['lumdist'][0]['e_value'])
        except Exception:
            self.lum_dist_err = 0

        self.z = float(self.json_data[self.SN_name]['redshift'][0]['value'])

        self.N = len(self.json_data[self.SN_name]['photometry'])

        self.claimedtype = self.json_data[self.SN_name]['claimedtype'][0]['value']

        for i in range(self.N):

            try:
                self.band.append(self.json_data[self.SN_name]['photometry'][i]['band'])
            except:
                self.band.append(0)

            for j, filter in enumerate(self.filters):

                if self.band[i] == filter:

                    self.t[j].append(float(self.json_data[self.SN_name]['photometry'][i]['time']))

                    self.m_app = float(self.json_data[self.SN_name]['photometry'][i]['magnitude'])

                    self.m[j].append(self.m_app - 5*log10(self.lum_dist*1e5) + 2.5*log10(1+self.z))

                    try:
                        self.m_app_err = float(self.json_data[self.SN_name]['photometry'][j]['e_magnitude'])
                        self.m_err[j].append(sqrt(self.m_app_err**2 + (5*0.434*self.lum_dist_err/self.lum_dist)**2))
                    except:
                        self.m_err[j].append(0.3)

        if kwargs['peak_alignment']:
            LC_Preprocess.lc_truncation(self, peak_alignment=True)
        else:
            LC_Preprocess.lc_truncation(self, peak_alignment=False)

        if kwargs['LC_graph']:
            LC_Preprocess.lc_graph(self)
        
        return self.t, self.m, self.m_err, self.claimedtype, self.SN_name

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

    os.chdir(f'{pp}/OSC_json') # Where the .json files are stored
    print('The current working directory is', os.getcwd())

    filenames = glob.glob('*.json')
    np.random.seed(1) 
    np.random.shuffle(filenames) # Shuffle for later ML training purpose
    print('The number of SNe is', len(filenames))

    t_all = []
    m_all = []
    m_err_all = []
    claimedtype_all = []
    SN_name_all = []

    filter_SDSS = ['g', 'r', 'i'] # Selected band system
    filter_SDSS_prime = ["g'", "r'", "i'"]
    filter_Johnson = ['B', 'V', 'R', 'I']

    filter_all = filter_SDSS_prime

    if filter_all == filter_SDSS:
        phtmet_sys_name = 'SDSS'
    if filter_all == filter_SDSS_prime:
        phtmet_sys_name = 'SDSS_prime'

    create_clean_directory(f'{pp}/{phtmet_sys_name}_import_graph')
    os.chdir(f'{pp}/{phtmet_sys_name}_import_graph')

    lc_len_prepeak = -24
    lc_len_postpeak = 72

    num_extracted_SN = 0 # Initialize the number of screened

    # Import .json of SNe and also screening
    print('Screening and extracting SNe ...')
    for i, filename in tenumerate(filenames):
        with open(f'{pp}/OSC_json/{filename}', encoding="utf-8") as f:

            json_data = json.load(f)

            filename_QC1 = QC(filename, filter_all, json_data).avoid_non_SNIa()
            filename_QC2 = QC(filename, filter_all, json_data).avoid_empty_SN(num=20, lc_len_prepeak=lc_len_prepeak, lc_len_postpeak=lc_len_postpeak)

            if (filename_QC1 and filename_QC2) is True: # Choosing SNe that are both SNIa and not empty

                LC_result = LC_Preprocess(
                                        filename, filter_all, json_data, 
                                        lc_len_prepeak=lc_len_prepeak, lc_len_postpeak=lc_len_postpeak
                                        ).lc_extractor(
                                            peak_alignment=False, LC_graph=True
                                            )

                t_all.append(LC_result[0])
                m_all.append(LC_result[1])
                m_err_all.append(LC_result[2])
                claimedtype_all.append(LC_result[3])
                SN_name_all.append(LC_result[4])

                num_extracted_SN += 1

    create_clean_directory(f'{pp}/{phtmet_sys_name}_import_npy')

    os.chdir(f'{pp}/{phtmet_sys_name}_import_npy')
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