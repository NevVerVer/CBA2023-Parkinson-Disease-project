import numpy as np
import pickle
import sys
import os
from docopt import docopt
from tabulate import tabulate

from utils import progbar
from swift import aswift

from dbs import cDBS, pDBS


from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def calc_envelope(x):
    ht = hilbert(x)
    return np.abs(ht)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_beta_envelope(list_of_pathes, list_of_titles, ts, te,
                       area='STN', plot_signal=False):
    # default beta, Hz
    lowcut = 13.5
    highcut = 30.0

    fig, ax = plt.subplots(1, 1, figsize=(18, 4))

    for n, name in enumerate(list_of_pathes):
        # load data
        mfm_class = MFM()
        mfm_class.load(name)
        x = mfm_class.S[ts:te, mfm_class.struct[area]]

        T = mfm_class.params['dt'] * (te-ts)
        fs = 1/mfm_class.params['dt']
        t = np.linspace(0, T, x.shape[0], endpoint=False)

        # ax.plot(t, x, label='orig_{}'.format(list_of_titles[n]))
        y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
        y_env = calc_envelope(y)
        if plot_signal:
            ax.plot(t, y, label=list_of_titles[n], alpha=0.8)
        ax.plot(t, y_env, alpha=0.6, label='env_{}'.format(list_of_titles[n]))
    ax.set_xlabel('time (seconds)')
    ax.legend()
    plt.grid(True)
    plt.show()
#=====================================================

def plot_psd(list_of_pathes, list_of_titles, beta_peak=True):
    from scipy import signal
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle('PSD of all populations', fontsize=13)
    j = 2
    log = {}
    if beta_peak:
        for l in list_of_titles:
            log[l] = {}
    for i in range(3):
        for k in range(3):
            for n, name in enumerate(list_of_pathes):
                # load data
                mfm_class = MFM()
                mfm_class.load(name)
                xx = mfm_class.S

                f, Pxx = signal.welch(xx[:, j], 1/mfm_class.params['dt'], nperseg=2048)
                Pxx = 10*np.log10(Pxx)

                lab = mfm_class.idx_dict[j]

                if beta_peak:
                    # print(lab)
                    amp_max = np.max(Pxx[(f<33.) & (f>20.)])
                    amp_max_idx = np.where(Pxx == amp_max)
                    # print('Ampl: {}, at freq: {}'.format(amp_max,
                    #                                      f[amp_max_idx]))
                    log[list_of_titles[n]][lab] = [round(amp_max, 4),
                                                   round(f[amp_max_idx][0], 4)]

                axs[i, k].plot(f[f<100],Pxx[f<100], label=lab+list_of_titles[n])

            # Freq of intrest
            x1, x2 = 13.5, 30.0
            lw = 0.7
            axs[i, k].axvline(x=x1, c='red', alpha=0.3, label='beta, (13.5–30 Hz)', lw=lw)
            axs[i, k].axvline(x=x2, c='red', alpha=0.3, lw=lw)
            axs[i, k].fill_betweenx([-50, 10], x1, x2, color='red', alpha=0.08)

            x1, x2 = 6.5, 8.0
            axs[i, k].axvline(x=x1, c='green', alpha=0.3, label='theta, (6.5–8 Hz)', lw=lw)
            axs[i, k].axvline(x=x2, c='green', alpha=0.3, lw=lw)
            axs[i, k].fill_betweenx([-50, 10], x1, x2, color='green', alpha=0.08)

            x1, x2 = 1.5, 6.0
            axs[i, k].axvline(x=x1, c='blue', alpha=0.3, label='delta, (1.5–6 Hz)', lw=lw)
            axs[i, k].axvline(x=x2, c='blue', alpha=0.3, lw=lw)
            axs[i, k].fill_betweenx([-50, 10], x1, x2, color='blue', alpha=0.08)

            axs[i, k].set_ylabel('PSD (dB/Hz)')
            axs[i, k].set_xlabel('Frequency (Hz)')
            axs[i, k].set_xlim([-3, 105])
            axs[i, k].set_ylim([-50, 10])
            axs[i, k].legend()
            j += 2
   
    plt.tight_layout()
    plt.show()
    return log

#=====================================================
def plot_activity(ts, te, list_of_pathes, list_of_titles):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(9, 1, figsize=(9, int(9*1.5)))
        fig.suptitle('LFP activity, [{}, {}]'.format(ts, te), fontsize=13)
        j = 2
        for i in range(9):
            for n, name in enumerate(list_of_pathes):
                # load data
                mfm_class = MFM()
                mfm_class.load(name)
                xx = mfm_class.S

                t = np.arange(mfm_class.params['N'])[ts:te] * mfm_class.params['dt']
                axs[i].plot(t, xx[ts:te, j], label=mfm_class.idx_dict[j] + list_of_titles[n],
                            alpha=0.5)
                axs[i].legend(loc=1)
                axs[i].set_xlim([t[0], t[-1]])
                if mfm_class.params['cDBS'] or mfm_class.params['pDBS']:
                    axs[i].axvline(mfm_class.params['DBS_start']* mfm_class.params['dt'],
                                c='red', alpha=0.4)
                # axs[i].axvline(self.params['stim_start'], c='red', alpha=0.4)
                
            j += 2
        plt.tight_layout()
        plt.show()

#=====Variance=======================================================================
def calc_scalar_var(x):
    return np.var(x, axis=0)

def print_var(list_of_pathes, list_of_titles):
    import matplotlib.pyplot as plt
    # scalar variance - all signal
    for n, name in enumerate(list_of_pathes):
        # load data
        mfm_class = MFM()
        mfm_class.load(name)
        print('For ', list_of_titles[n], '-'*20)
        scal_var = mfm_class.calc_scalar_var(mfm_class.S)
        j = 2
        for _ in range(9):
            print('Variance of {}: {}'.format(mfm_class.idx_dict[j],
                                                round(scal_var[j], 4)))
            j += 2

def plot_slidingw_var(list_of_pathes, list_of_titles, size, ss=0):
    import matplotlib.pyplot as plt
    # 2nd: sliding window variance
    fig, axs = plt.subplots(9, 1, figsize=(15, 11))
    fig.suptitle('Variance, sliding window, size={}'.format(size), fontsize=13)

    for n, name in enumerate(list_of_pathes):
        # load data
        mfm_class = MFM()
        mfm_class.load(name)

        steps = int(mfm_class.params['N']/size)
        vals = np.empty((steps, 9))
        ts, te = ss, ss+size
        for idx in range(steps):
            vals[idx, :] = np.var(mfm_class.S[ts:te, 2::2], axis=0)
            ts = te
            te += size

        j = 2
        for i in range(9):
            axs[i].plot(vals[:, i], label=mfm_class.idx_dict[j]+list_of_titles[n])
            # if mfm_class.params['cDBS'] or mfm_class.params['pDBS']:
            #     axs[i].axvline(mfm_class.params['DBS_start']* mfm_class.params['dt'],
            #                     c='red', alpha=0.4)
            j += 2
            axs[i].legend()
    plt.tight_layout()
    plt.show()

def plot_accum_var(list_of_pathes, list_of_titles, size, ss=0):
    import matplotlib.pyplot as plt
    # 3rd: accumulation of variance
    fig, axs = plt.subplots(9, 1, figsize=(15, 11))
    fig.suptitle('Accumulated variance, size={}'.format(size), fontsize=13)

    for n, name in enumerate(list_of_pathes):
        # load data
        mfm_class = MFM()
        mfm_class.load(name)

        steps = int(mfm_class.params['N']/size)
        vals = np.empty((steps, 9))  # self.params['N'], 
        ts, te = ss, ss+size
        for idx in range(steps):
            vals[idx, :] = np.var(mfm_class.S[ts:te, 2::2], axis=0)
            te += size

        j = 2
        for i in range(9):
            axs[i].plot(vals[:, i], label=mfm_class.idx_dict[j]+list_of_titles[n])
            # if mfm_class.params['cDBS'] or mfm_class.params['pDBS']:
            #     axs[i].axvline(mfm_class.params['DBS_start']* mfm_class.params['dt'],
            #                     c='red', alpha=0.4)
            j += 2
            axs[i].legend()
    plt.tight_layout()
    plt.show()
#=====================================================
#=======Phase plot======================================
def plot_limit_cycles(list_of_pathes, list_of_titles, ts, te):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(9, 9, figsize=(20, 18))
    for z, i in enumerate(range(2, 20, 2)):
        for k, j in enumerate(range(2, 20, 2)):

            for n, name in enumerate(list_of_pathes):
                # load data
                mfm_class = MFM()
                mfm_class.load(name)

                if z == k:
                    axs[z, k].text(0.25, 0.4, mfm_class.idx_dict[i], fontsize = 32)
                else:
                    axs[z, k].plot(mfm_class.S[ts:te, i], mfm_class.S[ts:te, j],
                                    lw=0.5, alpha=0.6, label=list_of_titles[n]) # c='black',
                    # axs[z, k].legend()
            axs[z, k].set_xticks([])
            axs[z, k].set_yticks([])
    plt.show()

#=====================================================


#=====================================================
class MFM(object):
    def __init__(self,**kwargs):
        self.t = 0                  #internal time counter
        self.i = 0                  #internal index

        self._load_params(kwargs)
        self._set_MFM_params()
        self._set_DBS()
#-------------------------------------------------------------------
        self.S = np.zeros((self.params['N'], 20))
        self.S[0,:] = [ 43.74102506,  -1.15197439,    6.96276347,  -22.25852135,    7.19671392,
                       -28.57548512,  17.26916297,  132.89911127,    9.71319243,   67.69191101,
                        9.57769785,  -21.11198645,    5.33943222,  -22.62375016,    0.54172422,
                       -22.23467637,   6.76173506,   143.5386694,    7.49756915,   23.59983148]

        self.swift = aswift(tau_s = 1./self.params['swift_f'] * self.params['swift_c'],
                            tau_f = 1./self.params['swift_f'] * self.params['swift_c'] / self.params['swift_s2f'],
                            f = self.params['swift_f'],
                            fs = self.params['fs'])

        self.memory = {
            'amp'   : np.zeros(self.params['N']),
            'phase' : np.zeros(self.params['N']),
            'stim'  : np.zeros(self.params['N']),
        }

    def __str__(self):
        general = ('Run Info\n'+
                   '--------\n'+
                   'length   : {0} s\n'
                   'DD       : {1}\n'
                   'cDBS     : {2}\n'
                   'pDBS     : {3}\n')\
                   .format(self.params['tstop'],self.params['DD'],self.params['cDBS'],self.params['pDBS'])

        SWIFT = ('\nSWIFT Parameters\n'
                 '----------------\n'
                 'f     : {:0.1f} Hz\n'
                 'tau_s : {:0.4f} s\n'
                 'tau_f : {:0.4f} s\n')\
                 .format(*[self.params[key] for key in ['swift_f','swift_tau_s','swift_tau_f']])

        if self.params['cDBS']:
            cDBS =('\ncDBS Parameters\n'
                   '---------------\n'
                   'frequency : {} Hz\n'
                   'amplitude : {} mA\n'
                   'pulse width : {} us')\
                   .format(*[self.params[key] for key in ['cDBS_f','cDBS_amp','cDBS_width']])
        else: cDBS=''

        if self.params['pDBS']:
            pDBS=('\npDBS Parameters\n'
                  '---------------\n'
                  'phase thr  : {} rad\n'
                  'stim amp   : {} mA\n'
                  'power thr  : {} dB\n'
                  'ref period : {}\n')\
                  .format(*[self.params[key] for key in ['pDBS_phase','pDBS_amp','pDBS_power_thr','pDBS_ref_period']])
        else: pDBS=''
            
        return general+SWIFT+cDBS+pDBS
        
    def _load_params(self,kwargs):
        def process_kwargs(kwargs):
            for key,value in kwargs.items():
                if key not in self.params:
                    print('Invalid keyword argument {}'.format(key))
                else:
                    value = type(self.params[key])(value) #cast value to that of params[key]
                    self.params[key] = value

        self.params = {}

        self.params['verbose'] = True
        self.params['fname'] = 'data/000.mfm'

        #General parameters
        self.params['dt']         = 1e-3        # s
        self.params['stim_start'] = 0.0         # s
        self.params['tstop']      = 50.0        # s
        self.params['RunID']      = -1          
                
        #DD parameters
        self.params['DD'] = True 
        self.params['DDmode'] = 5

        #Stimulation parameters
        self.params['stim_target'] = 'STN'
        self.params['Cm']          = 1e-4       # F
        
        #cDBS parameters
        self.params['cDBS']        = False
        self.params['cDBS_f']      = 130.       # (Hz)
        self.params['cDBS_amp']    = 3.0        # (mV)
        self.params['cDBS_width']  = 60.        # (us)

        #pDBS parameters
        self.params['pDBS']       = False
        self.params['pDBS_phase'] = 2.24        # rad
        self.params['pDBS_amp']   = 2.38        # rad
        self.params['pDBS_width'] = 60          # us
        self.params['pDBS_ref_period'] = 0.3    # s
        self.params['pDBS_power_thr'] = -28.57  # dB
        
        #SWIFT params
        self.params['state_target'] = 'p1'
        self.params['swift_f']      = 29        # Hz
        self.params['swift_tau_s']  = 0.2397    # s - Overrides swift_c when set
        self.params['swift_c']      = 10        # unitless - number of cycles per tau
        self.params['swift_s2f']    = 5         # unitless - ratio of tau_s to tau_f

        self.params['noise'] = False  # my parameter

        self._options = self.params.copy()
        process_kwargs(kwargs)
        
        #Set parameters dependent on other parameters
        # np.ceil - round to the top int value
        self.params['N'] = int(np.ceil(self.params['tstop']/self.params['dt']))
        self.params['fs'] = 1./self.params['dt']
        
        # added, idx of DBS start
        self.params['precent_DBS'] = 5
        self.params['DBS_start'] = int(self.params['N']*self.params['precent_DBS']/100) # added, idx of DBS start

        if self.params['swift_tau_s'] is None:
            self.params['swift_tau_s'] = 1./self.params['swift_f'] * self.params['swift_c']
        self.params['swift_tau_f'] = self.params['swift_tau_s'] / self.params['swift_s2f']
                         
    def _set_MFM_params(self):
        self.phin = 15
        self.noiseAmp = 0.03 # 0.01
        self.re = 80 #mm  # Corticocortical axonal range
        self.gammae = 125 #s^-1  # Cortical damping rate
        self.alpha = 160 #s^-1  # Synaptodendritic Decay rate
        self.beta = 640 #s^-1  # Rise rate
        self.gammasq=self.gammae^2
        self.alphabeta=self.alpha*self.beta
        self.aPb = (self.alpha+self.beta)#/alphabeta
        self.alphagamma=self.alpha*self.gammae # ?

        #Axonal Delays: from second to first (1st population type is postsynaptic)
        #(ms)  # kuda-otkuda: taud1e = in d1 from e
        self.taues   = int(0.035/self.params['dt'])
        self.tauis   = int(0.035/self.params['dt'])
        self.taud1e  = int(0.002/self.params['dt'])
        self.taud2e  = int(0.002/self.params['dt'])
        self.taud1s  = int(0.002/self.params['dt'])
        self.taud2s  = int(0.002/self.params['dt'])
        self.taup1d1 = int(0.001/self.params['dt'])
        self.taup1p2 = int(0.001/self.params['dt'])
        self.taup1ST = int(0.001/self.params['dt'])
        self.taup2d2 = int(0.001/self.params['dt'])
        self.taup2ST = int(0.001/self.params['dt'])
        self.tauSTe  = int(0.001/self.params['dt'])
        self.tauSTp2 = int(0.001/self.params['dt'])
        self.tause   = int(0.050/self.params['dt'])
        self.taure   = int(0.050/self.params['dt'])
        self.tausp1  = int(0.003/self.params['dt'])
        self.tausr   = int(0.002/self.params['dt'])
        self.taurs   = int(0.002/self.params['dt'])
        self.taud2d1 = int(0.001/self.params['dt'])

        #Threshold spread (mV)
        self.sigmaprime = 3.8

        #Connection strength (mVs)
        self.vee   =  1.6   #1.6 1.4 is DD
        self.vie   =  1.6   #1.6 1.4 is DD
        self.vei   = -1.9   #-1.9-1.6 is DD
        self.vii   = -1.9   #-1.9 -1.6 is DD
        self.ves   =  0.4 # match
        self.vis   =  0.4
        self.vd1e  =  1.0   #1 #1 is normal, .5 is DD # match
        self.vd1d1 = -0.3
        self.vd1s  =  0.1 # match
        self.vd2e  =  0.7   #0.7 1.4 is DD # match
        self.vd2d2 = -0.3
        self.vd2s  =  0.05 # match
        self.vp1d1 = -0.1
        self.vp1p2 = -0.03
        self.vp1ST =  0.3
        self.vp2d2 = -0.3   #-0.3; -0.5 is DD
        self.vp2p2 = -0.1   #-0.1; -0.07 is DD
        self.vp2ST =  0.3
        self.vSTe  =  0.1
        self.vSTp2 = -0.04
        self.vse   =  0.8 # match
        self.vsp1  = -0.03 # match 
        self.vsr   = -0.4 # match
        self.vsn   =  0.5
        self.vre   =  0.15  # match
        self.vrs   =  0.03
        self.vd2d1 =  0

        #Sigmoids
        #Maximum Firing Rates (s^-1)
        self.Qe = 300
        self.Qi = 300
        self.Qd1 = 65
        self.Qd2 = 65
        self.Qp1 = 250
        self.Qp2 = 300
        self.QST = 500
        self.Qs = 300
        self.Qr = 500

        #Firing Thresholds (mV)
        self.thetae = 14
        self.thetai = 14
        self.thetad1 = 19
        self.thetad2 = 19
        self.thetap1 = 10
        self.thetap2 = 9     #9 8 is DD
        self.thetaST = 10    #10 9 is DD
        self.thetas = 13
        self.thetar = 13

        self.phie     = 0
        self.phie_dot = 1
        self.Ve       = 2
        self.Ve_dot   = 3
        self.Vi       = 4
        self.Vi_dot   = 5
        self.Vd1      = 6
        self.Vd1_dot  = 7
        self.Vd2      = 8
        self.Vd2_dot  = 9
        self.Vp1      = 10
        self.Vp1_dot  = 11
        self.Vp2      = 12
        self.Vp2_dot  = 13
        self.VST      = 14
        self.VST_dot  = 15
        self.Vs       = 16
        self.Vs_dot   = 17
        self.Vr       = 18
        self.Vr_dot   = 19
        self.struct={'e'   :2,
                     'i'   :4,
                     'd1'  :6,
                     'd2'  :8,
                     'p1'  :10,
                     'p2'  :12,
                     'STN' :14,
                     's'   :16,
                     'r'   :18}
        self.idx_dict = {0: 'phie', 1: 'phie_dot',
                         2: 'Ve', 3: 'Ve_dot',
                         4: 'Vi', 5: 'Vi_dot',
                         6: 'Vd1', 7: 'Vd1_dot',
                         8: 'Vd2', 9: 'Vd2_dot',
                         10: 'Vp1', 11: 'Vp1_dot',
                         12: 'Vp2', 13: 'Vp2_dot',
                         14: 'VST', 15: 'VST_dot',
                         16: 'Vs', 17: 'Vs_dot',
                         18: 'Vr', 19: 'Vr_dot',}
        
        #Set "full parkinsonian state" parameters
        if self.params['DD']:
            if self.params['DDmode']==1:
                h = 15 #[5, 10, 15]
                xi = 0.6 # [0, 0,6] # in range
                self.thetad1 = self.thetad1 - h*xi
                self.thetad2 = self.thetad2 - h*xi
                self.vd1e = self.vd1e - xi  
                self.vd2e = self.vd2e - xi  

            if self.params['DDmode']==89:
                h = 24 #24
                xi = 0.6 # 0.6 
                self.thetad1 = self.thetad1 - h*xi
                self.thetad2 = self.thetad2 - h*xi
                self.vd1e = self.vd1e - xi  # 1 -> 0.5
                self.vd2e = self.vd2e + xi  # 0.7 -> 1.4

            elif self.params['DDmode']==2:
                eps = 1
                self.vd1e = self.vd1e - eps # 
                self.vd2e = self.vd2e + eps #

            elif self.params['DDmode']==3:
                self.vp2p2 = -0.05

            elif self.params['DDmode']==4:
                self.vee = 1.2
                self.vie = 1.2
                self.vei = -1.4
                self.vii = -1.4

            elif self.params['DDmode']==5:
                self.vee = 1.4
                self.vie = 1.4
                self.vei = -1.6
                self.vii = -1.6
                self.vd1e = 0.5 # 
                self.vd2e = 1.4 #
                self.vp2d2 = -0.5 #
                self.vp2p2 = -0.07 #
                self.thetap2 = 8 #
                self.thetaST = 9 #
                
            elif self.params['DDmode']==15:
                h = 15 # 15
                xi = 0.6 # 0.6
                self.thetad1 = self.thetad1 - h*xi
                self.thetad2 = self.thetad2 - h*xi
                
                self.vee = 1.4
                self.vie = 1.4
                self.vei = -1.6
                self.vii = -1.6
                self.vd1e = 0.5 # 
                self.vd2e = 1.4 #
                self.vp2d2 = -0.5 #
                self.vp2p2 = -0.07 #
                self.thetap2 = 8 #
                self.thetaST = 9 #
                
            elif self.params['DDmode']==150:
                h = 15 #[5, 10, 15]
                xi = 0.6 # [0, 0,6] # in range
                self.thetad1 = self.thetad1 - h*xi
                self.thetad2 = self.thetad2 - h*xi
                self.vd1e = self.vd1e - xi
                self.vd2e = self.vd2e - xi

                self.vee = 1.4
                self.vie = 1.4
                self.vei = -1.6
                self.vii = -1.6
 
                self.vp2d2 = -0.5 #
                self.vp2p2 = -0.07 #
                self.thetap2 = 8 #
                self.thetaST = 9 #

    def _set_DBS(self):
        if self.params['cDBS']:
            self.cDBS = cDBS(dt       = self.params['dt'],
                             f        = self.params['cDBS_f'],
                             stim_amp = self.params['cDBS_amp'],
                             width    = self.params['cDBS_width'],
                             tstart   = self.params['stim_start'])
        else:
            self.cDBS = None
            
        self.pDBS = pDBS(dt         = self.params['dt'],
                         f          = self.params['swift_f'],
                         tau_s      = self.params['swift_tau_s'],
                         tau_f      = self.params['swift_tau_f'],
                         phase_thr  = self.params['pDBS_phase'],
                         ref_period = self.params['pDBS_ref_period'],
                         stim_amp   = self.params['pDBS_amp'],
                         width      = self.params['pDBS_width'],
                         power_thr  = self.params['pDBS_power_thr'])

    def advance(self):
        def sigmoid(V,Q,theta):
            # sigm = np.sqrt(3)/np.pi
            sigm = 3.8 # YES = [2,9; 3.4]; NO = [0.1]
            return Q/(1+np.exp(-(V-theta)/sigm))

        i = self.i
        
        dSdt = np.zeros(20)

        dSdt[self.phie]= self.S[i, self.phie_dot]
        dSdt[self.phie_dot] = self.gammasq*(sigmoid(self.S[i,self.Ve], self.Qe, self.thetae)-\
                                            self.S[i,self.phie])-2*self.gammae*self.S[i,self.phie_dot]

        dSdt[self.Ve] = self.S[i,self.Ve_dot]
        dSdt[self.Ve_dot] = self.alphagamma*(self.vee*self.S[i,self.phie]+\
                                             self.vei*sigmoid(self.S[i,self.Vi],self.Qi, self.thetai)+\
                                             self.ves*sigmoid(self.S[i-self.taues,self.Vs], self.Qs, self.thetas)-\
                                             self.S[i,self.Ve])-\
                            self.aPb*self.S[i,self.Ve_dot]

        dSdt[self.Vi] = self.S[i,self.Vi_dot]
        dSdt[self.Vi_dot] = self.alphagamma*(self.vii*sigmoid(self.S[i,self.Vi], self.Qi, self.thetai)+\
                                             self.vie*self.S[i,self.phie]+\
                                             self.vis*sigmoid(self.S[i-self.tauis,self.Vs], self.Qs, self.thetas)-\
                                             self.S[i,self.Vi])-\
                            self.aPb*self.S[i,self.Vi_dot]

        dSdt[self.Vd1] = self.S[i,self.Vd1_dot]
        dSdt[self.Vd1_dot] = self.alphabeta*(self.vd1e*self.S[i-self.taud1e,self.phie]+\
                                             self.vd1s*sigmoid(self.S[i-self.taud1s,self.Vs], self.Qs, self.thetas)+\
                                             self.vd1d1*sigmoid(self.S[i,self.Vd1], self.Qd1, self.thetad1)-\
                                             self.S[i,self.Vd1])-\
                            self.aPb*self.S[i,self.Vd1_dot] #Add in SNc

        dSdt[self.Vd2] = self.S[i,self.Vd2_dot]
        dSdt[self.Vd2_dot] = self.alphabeta*(self.vd2e*self.S[i-self.taud2e, self.Ve]+\
                                             self.vd2d1*sigmoid(self.S[i-self.taud2d1,self.Vd1], self.Qd1, self.thetad1)+\
                                             self.vd2s*sigmoid(self.S[i-self.taud2s,self.Vs], self.Qs, self.thetas)+\
                                             self.vd2d2*sigmoid(self.S[i,self.Vd2], self.Qd2, self.thetad2)-\
                                             self.S[i,self.Vd2])-\
                            self.aPb*self.S[i,self.Vd2_dot] #Add in the SNc

        dSdt[self.Vp1] = self.S[i,self.Vp1_dot]
        dSdt[self.Vp1_dot] = self.alphabeta*(self.vp1d1*sigmoid(self.S[i-self.taup1d1,self.Vd1], self.Qd1, self.thetad1)+\
                                             self.vp1p2*sigmoid(self.S[i-self.taup1p2,self.Vp2], self.Qp2, self.thetap2)+\
                                             self.vp1ST*sigmoid(self.S[i-self.taup1ST,self.VST], self.QST, self.thetaST)-\
                                             self.S[i,self.Vp1])-\
                            self.aPb*self.S[i,self.Vp1_dot]

        dSdt[self.Vp2] = self.S[i,self.Vp2_dot]
        dSdt[self.Vp2_dot] = self.alphabeta*(self.vp2d2*sigmoid(self.S[i-self.taup2d2,self.Vd2], self.Qd2, self.thetad2)+\
                                             self.vp2p2*sigmoid(self.S[i,self.Vp2], self.Qp2, self.thetap2)+\
                                             self.vp2ST*sigmoid(self.S[i-self.taup2ST,self.VST], self.QST, self.thetaST)-\
                                             self.S[i,self.Vp2])-\
                            self.aPb*self.S[i,self.Vp2_dot]

        dSdt[self.VST] = self.S[i,self.VST_dot]
        dSdt[self.VST_dot] = self.alphabeta*(self.vSTp2*sigmoid(self.S[i-self.tauSTp2,self.Vp2], self.Qp2, self.thetap2)+\
                                             self.vSTe*self.S[i-self.tauSTe,self.phie]-\
                                             self.S[i,self.VST])-\
                            self.aPb*self.S[i,self.VST_dot]

        dSdt[self.Vs] = self.S[i,self.Vs_dot]
        dSdt[self.Vs_dot] = self.alphabeta*(self.vsp1*sigmoid(self.S[i-self.tausp1,self.Vp1], self.Qp1, self.thetap1)+\
                                            self.vse*self.S[i-self.tause,self.phie]+\
                                            self.vsr*sigmoid(self.S[i-self.tausr,self.Vr], self.Qr, self.thetar)+\
                                            self.phin-\
                                            self.S[i,self.Vs])-\
                            self.aPb*self.S[i,self.Vs_dot]

        dSdt[self.Vr] = self.S[i,self.Vr_dot]
        dSdt[self.Vr_dot] = self.alphabeta*(self.vre*self.S[i-self.taure,self.phie]+\
                                            self.vrs*sigmoid(self.S[i-self.taurs,self.Vs], self.Qs, self.thetas)-\
                                            self.S[i,self.Vr])-\
                            self.aPb*self.S[i,self.Vr_dot]

        #DBS
        #====================================================================================
        if self.params['cDBS'] and self.i > self.params['DBS_start']:
        # if self.params['cDBS']:
            cDBS_C = self.cDBS.advance()
            self.S[i,self.struct[self.params['stim_target']]] += cDBS_C/self.params['Cm']
            if cDBS_C != 0: self.memory['stim'][i+1] = cDBS_C
            
        else:
            pDBS_C = self.pDBS.advance(self.S[i,self.struct[self.params['state_target']]])
            if self.params['pDBS'] and self.i > self.params['DBS_start']:
            # if self.params['pDBS']:
                self.S[i,self.struct[self.params['stim_target']]] += pDBS_C/self.params['Cm']
                if pDBS_C != 0: self.memory['stim'][i+1] = pDBS_C
            
            self.memory['amp'][i+1]   = self.pDBS.amp
            self.memory['phase'][i+1] = self.pDBS.phase
            
        #Advance
        #====================================================================================
        self.S[i+1,:] = self.S[i,:] + self.params['dt']*dSdt

        #Noise
        #====================================================================================
        if self.params['noise']:

            self.S[i+1,self.Ve] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qe*\
                (1-sigmoid(self.S[i+1,self.Ve], 1, self.thetae))*sigmoid(self.S[i,self.Ve], 1, self.thetae)

            self.S[i+1,self.Vi] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qi*\
                (1-sigmoid(self.S[i+1,self.Vi], 1, self.thetai))*sigmoid(self.S[i,self.Vi], 1, self.thetai)

            self.S[i+1,self.Vd1] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qd1*\
                (1-sigmoid(self.S[i+1,self.Vd1], 1, self.thetad1))*sigmoid(self.S[i,self.Vd1], 1, self.thetad1)
            
            self.S[i+1,self.Vd2] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qd2*\
                (1-sigmoid(self.S[i+1,self.Vd2], 1, self.thetad2))*sigmoid(self.S[i,self.Vd2], 1, self.thetad2)
            
            self.S[i+1,self.Vp1] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qp1*\
                (1-sigmoid(self.S[i+1,self.Vp1], 1, self.thetap1))*sigmoid(self.S[i,self.Vp1], 1, self.thetap1)
            
            self.S[i+1,self.Vp2] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qp2*\
                (1-sigmoid(self.S[i+1,self.Vp2], 1, self.thetap2))*sigmoid(self.S[i,self.Vp2], 1, self.thetap2)
            
            self.S[i+1,self.VST] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.QST*\
                (1-sigmoid(self.S[i+1,self.VST], 1, self.thetaST))*sigmoid(self.S[i,self.VST], 1, self.thetaST)
            
            self.S[i+1,self.Vs] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qs*\
                (1-sigmoid(self.S[i+1,self.Vs], 1, self.thetas))*sigmoid(self.S[i,self.Vs], 1, self.thetas)
            
            self.S[i+1,self.Vr] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qr*\
                (1-sigmoid(self.S[i+1,self.Vr], 1, self.thetar))*sigmoid(self.S[i,self.Vr], 1, self.thetar)

        self.i += 1

    def run(self):
        #if self.params['verbose']: self.progbar = ProgressBar()
        if self.params['verbose']: self.progbar = progbar()

        while self.i < self.params['N'] - 1:
            self.advance()

            #if self.params['verbose']: self.progbar.display(float(self.i)/(self.params['N']-2))
            if self.params['verbose']: self.progbar.update(float(self.i)/(self.params['N']-2))
        if self.params['verbose']: print()
        
    def save(self,fname=None):
        if fname == None:
            if not os.path.isdir('data'):
                os.makedirs('data')
            if self.params['RunID'] == -1:
                try:
                    ls = os.listdir('data')
                    lowest_empty = 0
                    found_empty = False
                    while not found_empty:
                        found_empty = True
                        for fname in ls:
                            try: fnum = int(fname[:3])
                            except: fnum = -1
                            if fnum == lowest_empty:
                                lowest_empty += 1
                                found_empty = False
                    self.params['RunID'] = lowest_empty
                except:
                    self.params['RunID'] = 0
            print('\nSaving data...\n  RunID: {0:03d}'.format(self.params['RunID']))
            # fname = 'data/noDD_noise_nocDBS.mfm'.format(self.params['RunID'])
            fname = self.params['fname']
            # fname = 'data/{0:03d}.mfm'.format(self.params['RunID'])
        else:
            print('\nSaving data...\n  {}'.format(fname))
        pickle.dump(self.__dict__,open(fname,'wb'))
        
    def load(self,fname):
        self.__dict__.update(pickle.load(open(fname,'rb')))
    
    #===================Plotting================================

    def plot_limit_cycles(self, ts, te):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(9, 9, figsize=(20, 18))
        for z, i in enumerate(range(2, 20, 2)):
            for k, j in enumerate(range(2, 20, 2)):
                if z == k:
                    axs[z, k].text(0.25, 0.4, self.idx_dict[i], fontsize = 32)
                else:
                    axs[z, k].plot(self.S[ts:te, i], self.S[ts:te, j],
                                    lw=0.5, c='black', alpha=0.6)
                axs[z, k].set_xticks([])
                axs[z, k].set_yticks([])
        plt.show()

    def plot_activity(self, ts, te):
        import matplotlib.pyplot as plt

        t = np.arange(self.params['N'])[ts:te] * self.params['dt']
        fig, axs = plt.subplots(9, 1, figsize=(18, int(9*1.5)))
        # plt.title('LFP activity, [{}, {}]'.format(ts, te))
        fig.suptitle('LFP activity, [{}, {}]'.format(ts, te), fontsize=13)
        j = 2
        for i in range(9):
            axs[i].plot(t, self.S[ts:te, j], label=self.idx_dict[j])
            axs[i].legend()
            axs[i].set_xlim([t[0], t[-1]])
            if self.params['cDBS'] or self.params['pDBS']:
                axs[i].axvline(self.params['DBS_start']* self.params['dt'],
                               c='red', alpha=0.4)
                # axs[i].axvline(self.params['stim_start'], c='red', alpha=0.4)
            j += 2
        plt.tight_layout()
        plt.show()

    def plot_psd(self):
        from scipy import signal
        import matplotlib.pyplot as plt

        if self.params['cDBS'] or self.params['pDBS']:
            sig = self.S[self.params['DBS_start']+1:, :]
        else:
            sig = self.S
        fig, axs = plt.subplots(3, 3, figsize=(15, 11))
        fig.suptitle('PSD of all populations', fontsize=13)
        j = 2
        for i in range(3):
            for k in range(3):
                f, Pxx = signal.welch(sig[:, j], 1/self.params['dt'], nperseg=2048)
                Pxx = 10*np.log10(Pxx)

                lab = self.idx_dict[j]
                col = 'black'
                if j == self.struct[self.params['state_target']]:
                    lab = lab + ' (recorded)'
                    col = 'red'
                elif j == self.struct[self.params['stim_target']]:
                    lab = lab + ' (stimulated)'
                    col = 'red'

                axs[i, k].plot(f[f<100],Pxx[f<100], label=lab, c=col)
                # Freq of intrest
                x1, x2 = 13.5, 30.0
                axs[i, k].axvline(x=x1, c='red', alpha=0.3, label='beta, (13.5–30 Hz)')
                axs[i, k].axvline(x=x2, c='red', alpha=0.3)
                axs[i, k].fill_betweenx([-50, 10], x1, x2, color='red', alpha=0.08)

                x1, x2 = 6.5, 8.0
                axs[i, k].axvline(x=x1, c='green', alpha=0.3, label='theta, (6.5–8 Hz)')
                axs[i, k].axvline(x=x2, c='green', alpha=0.3)
                axs[i, k].fill_betweenx([-50, 10], x1, x2, color='green', alpha=0.08)

                x1, x2 = 1.5, 6.0
                axs[i, k].axvline(x=x1, c='blue', alpha=0.3, label='delta, (1.5–6 Hz)')
                axs[i, k].axvline(x=x2, c='blue', alpha=0.3)
                axs[i, k].fill_betweenx([-50, 10], x1, x2, color='blue', alpha=0.08)

                axs[i, k].set_ylabel('PSD (dB/Hz)')
                axs[i, k].set_xlabel('Frequency (Hz)')
                axs[i, k].set_xlim([-3, 105])
                # axs[i, k].set_ylim([-50, 10])
                axs[i, k].legend()
                j += 2
        plt.tight_layout()
        plt.show()

    #=====Variance=======================================================================
    def calc_scalar_var(self, x):
        return np.var(x, axis=0)

    def print_var(self):
        import matplotlib.pyplot as plt
        # scalar variance - all signal
        scal_var = self.calc_scalar_var(self.S)
        j = 2
        for _ in range(9):
            print('Variance of {}: {}'.format(self.idx_dict[j],
                                              round(scal_var[j], 4)))
            j += 2

    def plot_slidingw_var(self, size):
        import matplotlib.pyplot as plt
        # 2nd: sliding window variance
        steps = int(self.params['N']/size)
        vals = np.empty((steps, 9))
        ts, te = 0, size
        for idx in range(steps):
            vals[idx, :] = np.var(self.S[ts:te, 2::2], axis=0)
            ts = te
            te += size
        fig, axs = plt.subplots(9, 1, figsize=(15, 11))
        fig.suptitle('Variance, sliding window, size={}'.format(size), fontsize=13)
        j = 2
        for i in range(9):
            axs[i].plot(vals[:, i], label=self.idx_dict[j])
            if self.params['cDBS'] or self.params['pDBS']:
                axs[i].axvline(self.params['DBS_start']* self.params['dt'],
                               c='red', alpha=0.4)
            j += 2
        plt.tight_layout()
        plt.show()

    def plot_accum_var(self, size):
        import matplotlib.pyplot as plt
        # 3rd: accumulation of variance
        steps = int(self.params['N']/size)
        vals = np.empty((steps, 9))  # self.params['N'], 
        te = size
        for idx in range(steps):
            vals[idx, :] = np.var(self.S[:te, 2::2], axis=0)
            te += size
        fig, axs = plt.subplots(9, 1, figsize=(15, 11))
        fig.suptitle('Accumulated variance, size={}'.format(size), fontsize=13)
        j = 2
        for i in range(9):
            axs[i].plot(vals[:, i], label=self.idx_dict[j])
            if self.params['cDBS'] or self.params['pDBS']:
                axs[i].axvline(self.params['DBS_start']* self.params['dt'],
                                c='red', alpha=0.4)
            j += 2
        plt.tight_layout()
        plt.show()

    #=====CCF==================================================================
    def ccf(self, x, y):
        Rxx2 = np.correlate(x-np.mean(x), y-np.mean(y), mode='full')
        Rxx2un = np.divide(Rxx2, np.bartlett(len(Rxx2)))
        return Rxx2un[int(Rxx2un.size/2):]

    def plot_autocorr(self, N=500):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 3, figsize=(12, 7))
        fig.suptitle('Autocorrelation', fontsize=13)
        j = 2
        for i in range(3):
            for k in range(3):
                axs[i, k].plot(self.ccf(self.S[:N, j], self.S[:N, j]),
                               label=self.idx_dict[j])
                axs[i, k].legend()
                axs[i, k].axhline(y=0., color='r', linestyle='--',
                                  alpha=0.4, lw=0.8)
        plt.tight_layout()
        plt.show()
        
    def plot_crosscorr(self, N=500):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(9, 9, figsize=(20, 18))
        for z, i in enumerate(range(2, 20, 2)):
            for k, j in enumerate(range(2, 20, 2)):
                if z == k:
                    axs[z, k].text(0.25, 0.4, self.idx_dict[i], fontsize = 32)
                else:
                    axs[z, k].plot(self.ccf(self.S[:N, i], self.S[:N, j]),
                                    lw=0.8, c='black')
                    axs[z, k].axhline(y=0., color='r', linestyle='--',
                                      alpha=0.4, lw=0.8)
                axs[z, k].set_xticks([])
                axs[z, k].set_yticks([])
        plt.show()

    @property
    def options(self):
        return self._options
    
# def main():
#     kwargs = {'DD': True,
#               'noise': True,
#               'cDBS': False,
#               'pDBS': False,

#               'DDmode': 15,
#               'fname': 'data/dd15.mfm'}
    
#     mfm = MFM(**kwargs) 
#     # print(mfm)
#     mfm.run()
#     # mfm.save()
#     return mfm.S, mfm

# ff, mfm_class = main()