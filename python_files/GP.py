import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import george
from george import kernels
from scipy.optimize import minimize
from pathlib import Path
from tqdm import tqdm

class GP:


    def __init__(self, t, m, m_err, type, SN_name, filters, filters_EffWM , lc_len_prepeak=-24, lc_len_postpeak=72):

        self.t = t
        self.m = m
        self.m_err = m_err

        self.type = type
        self.SN_name = SN_name

        self.filters = filters
        self.filters_EffWM = filters_EffWM

        self.lc_len_prepeak = lc_len_prepeak
        self.lc_len_postpeak = lc_len_postpeak

        self.x = []
        self.y = []
        self.y_err = []
        self.filters_num = []

        self.data_t = []
        self.data_m = [ [] for filter in self.filters]
        self.data_m_err = [ [] for filter in self.filters]

        self.data = [ [] for i in range(1 + len(self.filters)*2)]
        self.data_plot = [ [] for i in range(1 + len(self.filters)*2)]
        self.data_meta =   {'t_len': None,
                            'type': None, 'SN_name': None,
                            'mean': 0, 'range': 1,
                            'peak_mag': 0, 'delta_m': 0, 'no_near_peak': 0}
        self.data_qc =  {'diff': 0, 'var': 0, 'noise': 0, 'score': 0}

        self.y_mean = 0
        self.y_range = 0

        for i, filter in enumerate(self.filters):
            for j in range(len(self.t[i])):

                self.x.append(self.t[i][j])
                self.y.append(self.m[i][j])
                self.y_err.append(self.m_err[i][j])
                self.filters_num.append(self.filters_EffWM[i])

    def x_GP_pred_generator(self):

        self.t_first = min([self.t[i][0] for i in range(len(self.filters))])
        self.t_last = max([self.t[i][-1] for i in range(len(self.filters))])

        self.x_tmp = []
        self.filters_num_tmp = []

        for i, filter in enumerate(self.filters):
            self.t_tmp = np.linspace(round(self.t_first), round(self.t_last), round(self.t_last) - round(self.t_first) + 1, endpoint=True)
            if len(self.t_tmp) == 97:
                self.t_tmp = self.t_tmp[:-1]
            for j in range(len(self.t_tmp)):
                self.x_tmp.append(self.t_tmp[j])
                self.filters_num_tmp.append(self.filters_EffWM[i])

        self.x_GP_pred = np.vstack([self.x_tmp, self.filters_num_tmp]).T

        return self.x_GP_pred


    def lc_padding(self):

        self.lc_len = self.lc_len_postpeak - self.lc_len_prepeak
        self.data_t = np.linspace(round(self.t_first), round(self.t_last), round(self.t_last) - round(self.t_first) + 1, endpoint=True)
        self.data_t_len = len(self.data_t)

        if self.data_t_len == 97:
                self.data_t = self.data_t[:-1]

        for i in range(len(self.filters)):

            self.data_m[i] = self.GP_pred[int((i)*len(self.GP_pred)/len(self.filters)):int((i+1)*len(self.GP_pred)/len(self.filters))]
            self.data_m_err[i] = np.sqrt(self.GP_var[int((i)*len(self.GP_var)/len(self.filters)):int((i+1)*len(self.GP_var)/len(self.filters))])

            '''for j in range(self.lc_len - len(self.data_m[i])):
                
                self.data_m[i] = np.append(self.data_m[i], 0)
                self.data_m_err[i] = np.append(self.data_m_err[i], 0)

                self.data_m[i] = np.append(self.data_m[i], self.data_m[i][-1])
                self.data_m_err[i] = np.append(self.data_m_err[i], self.data_m_err[i][-1])'''

        '''for i in range(self.lc_len - self.data_t_len):
            self.data_t = np.append(self.data_t, self.data_t[-1] + 1)'''

        '''if len(self.data_t) != self.lc_len:
            print('incorrect length')'''

        self.data[0] = self.data_t
        for i in range(len(self.filters)):
            self.data[i+1] = self.data_m[i]
        for i in range(len(self.filters)):
            self.data[i+len(self.filters)+1] = self.data_m_err[i]

        return self.data

    def lc_meta(self):

        self.data_meta['t_len'] = self.data_t_len
        self.data_meta['type'] = self.type
        self.data_meta['SN_name'] = self.SN_name

        self.peak_id = np.argmin(self.data[2]) #finding peak by r band
        self.data_t_peak = self.data[0][self.peak_id]
        self.score = 0
        for i in range(len(self.filters)):
            self.score += np.sum(np.cosh((np.array(self.t[i]) - self.data_t_peak)/10)**(-2))

        self.data_qc['score'] = self.score

        return self.data_meta, self.data_qc


    def normalization(self):

        self.y_mean = np.mean(self.y)
        self.y_range = np.abs(np.max(self.y) - np.min(self.y))

        self.y_err = self.y_err / self.y_range
        self.y = (self.y - self.y_mean) / self.y_range

        self.data_meta['mean'] = self.y_mean
        self.data_meta['range'] = self.y_range

        return self.y, self.y_err, self.y_mean, self.y_range


    def lc_graph(self, colors = ['lightseagreen', 'crimson', 'darkred']):

        #colors = ['indigo','darkcyan', 'limegreen', 'darkorange', 'crimson', 'maroon']

        plt.plot(figsize=(16,12))

        self.data_plot[0] = np.linspace(round(self.t_first), round(self.t_last), round(self.t_last) - round(self.t_first) + 1, endpoint=True)
        if len(self.data_plot[0]) == 97:
            self.data_plot[0] = self.data_plot[0][:-1]

        for i, filter in enumerate(self.filters):
            
            if self.y_range != 0:
                self.m_tmp = (self.m[i] - self.y_mean) / self.y_range
                self.m_err_tmp = self.m_err[i] / self.y_range
            else:
                self.m_tmp = self.m[i]
                self.m_err_tmp = self.m_err[i]

            self.data_plot[i+1] = self.data[i+1][:self.data_meta['t_len']]
            self.data_plot[i+len(self.filters)+1] = self.data[i+len(self.filters)+1][:self.data_meta['t_len']]

            plt.errorbar(np.array(self.t[i]), np.array(self.m_tmp), np.array(self.m_err_tmp), label=filter, color=colors[i], fmt='.')
            plt.plot(self.data_plot[0], self.data_plot[i+1], label=filter, color=colors[i], alpha=0.8)
            plt.fill_between(self.data_plot[0], self.data_plot[i+1] - self.data_plot[i+len(self.filters)+1], self.data_plot[i+1] + self.data_plot[i+len(self.filters)+1], color=colors[i], alpha=0.2)

        plt.title(f'{self.SN_name}, {self.type}')
        #plt.xlim(self.lc_len_prepeak, self.lc_len_postpeak)  
        plt.xlabel('time (MJD)')
        plt.ylabel('normalized absolute magnitude')
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()
        #plt.show()
        plt.savefig(f'./{self.SN_name}.pdf')
        plt.clf()

        return

    def qc(self, **kwargs):

        diff = 0
        var = 0
        err = 0

        for i in range(len(self.filters) - 1):
            diff += np.sum(self.data[i+2] - self.data[i+1])

        for i in range(len(self.filters)):
            var += np.var(self.data[i+1])
            err += np.sum(self.data[i+4])

        diff_norm = abs(diff*self.lc_len/self.data_t_len)
        var_norm = var*self.lc_len/self.data_t_len
        err_norm = abs(err*self.lc_len/self.data_t_len)

        return diff_norm, var_norm, err_norm

    def GP_interpolate(self, **kwargs):

        if kwargs['normalization']:
            self.normalization()

        self.x_GP = np.vstack([self.x, self.filters_num]).T

        self.x_GP_pred = self.x_GP_pred_generator()

        #print(self.x_GP, self.x_GP_pred)

        self.k1 = np.var(self.y)*kernels.ExpSquaredKernel(metric=[50, 5], ndim=2)
        self.k2 = np.var(self.y)*kernels.ExpKernel(metric=[50, 5], ndim=2)
        self.k = self.k1 + self.k2
        #self.k = np.var(self.y)*kernels.ExpSquaredKernel(metric=[50, 5], ndim=2)
        self.gp = george.GP(self.k, white_noise=np.log(np.var(self.y)), fit_white_noise=True)

        self.gp.compute(self.x_GP, self.y_err)

        def neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            return -self.gp.log_likelihood(np.array(self.y))

        def grad_neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            return -self.gp.grad_log_likelihood(np.array(self.y))

        try:
            minimize(neg_ln_like, self.gp.get_parameter_vector(), jac=grad_neg_ln_like, method='L-BFGS-B')
            self.gp.recompute()
            self.GP_pred, self.GP_var = self.gp.predict(self.y, self.x_GP_pred, return_var=True)
        except Exception:
            #print('failed to converge')
            return None, None, None

        self.data = self.lc_padding()
        self.data_meta, self.data_qc = self.lc_meta()

        self.data_qc['diff'], self.data_qc['var'], self.data_qc['noise'] = self.qc()

        if self.data_qc['diff'] < 1 or self.data_qc['var'] < 0.05 or self.data_qc['noise'] > 20 or self.data_qc['score'] < 5:
            return None, None, self.data_qc

        if kwargs['LC_graph']:
            self.lc_graph()

        return self.data, self.data_meta, self.data_qc

def create_clean_directory(d):

    isExist = os.path.exists(d)
    if isExist:
        shutil.rmtree(d)
        os.makedirs(d)
    else:
        os.makedirs(d)

    return

def main():

    SDSS_dict       = {'sys': 'SDSS', 
                       'filter': ['g', 'r', 'i'],
                       'EffWM': [4.672, 6.141, 7.458]}

    SDSS_prime_dict = {'sys': 'SDSS_prime',
                       'filter': ["g'", "r'", "i'"],
                       'EffWM': [4.725, 6.203, 7.673]}

    phtmet_sys = SDSS_dict

    filters_all = phtmet_sys['filter']
    filters_EffWM = phtmet_sys['EffWM']
    phtmet_sys_name = phtmet_sys['sys']

    pp = Path(__file__).parent.parent

    os.chdir(f'{pp}/{phtmet_sys_name}_import_npy')
    print('Loading in import.npy ...')

    t_all = np.load('Time_all.npy', allow_pickle=True)
    m_all = np.load('Magnitude_Abs_all.npy', allow_pickle=True)
    m_err_all = np.load('Magnitude_Abs_err_all.npy', allow_pickle=True)
    claimedtype_all = np.load('Type_all.npy', allow_pickle=True)
    SN_name_all = np.load('SN_name.npy', allow_pickle=True)

    print('Finished loading in import.npy')

    create_clean_directory(f'{pp}/{phtmet_sys_name}_GP_graph')

    os.chdir(f'{pp}/{phtmet_sys_name}_GP_graph')

    print('Working on GP interpolaiton')

    data_all      = [ [] for i in t_all]
    data_meta_all = [ [] for i in t_all]
    data_qc_all   = [ [] for i in t_all]

    for i in tqdm(range(len(t_all))):

        #LC_graph_bool = np.random.rand(1) > 0.99
        data_all[i], data_meta_all[i], data_qc_all[i] = GP(
            t_all[i], m_all[i], m_err_all[i], 
            claimedtype_all[i], SN_name_all[i], 
            filters_all, filters_EffWM,
            lc_len_prepeak=-24, lc_len_postpeak=72
            ).GP_interpolate(
                normalization=True, LC_graph=True
                )
    
    data_all      = list(filter(None, data_all))
    data_meta_all = list(filter(None, data_meta_all))
    data_qc_all   = list(filter(None, data_qc_all))

    print(f'There are in total successful {len(data_all)} GP interpolated SNe, and {len(t_all) - len(data_all)} SNe not successful')

    os.chdir(f'{pp}')

    create_clean_directory(f'{pp}/{phtmet_sys_name}_GP_npy')

    np.save(f'{pp}/{phtmet_sys_name}_GP_npy/lc.npy', np.array(data_all, dtype=object))
    np.save(f'{pp}/{phtmet_sys_name}_GP_npy/lc_meta.npy', np.array(data_meta_all, dtype=object))
    np.save(f'{pp}/{phtmet_sys_name}_GP_npy/lc_qc.npy', np.array(data_qc_all, dtype=object))

    print('End of GP.py')


if __name__ == '__main__':
    main()