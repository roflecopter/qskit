import pandas as pd
import numpy as np
import datetime
import os
import sys
import logging
import neurokit2 as nk
import matplotlib.pyplot as plt
import mne

logging.basicConfig(handlers=[
                        # logging.FileHandler(logging_file),
                        logging.StreamHandler(sys.stdout)
                    ],
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
# put repository dir here
working_dir = '/path/to/qs-kit'
os.chdir(working_dir)
from qskit.hrv.hrv_process import hrv_process
from qskit.hrv.sqi import hrv_quality
cache_dir = os.path.join(working_dir, 'qskit','examples','cache')

# Neurokit example with good data
data = nk.data("bio_resting_5min_100hz")
hrv = hrv_process(data['ECG'], sf = 100, window = 30, slide = 5, metrics = ['time','freq','bsi','ans','r_rr', 'nl','pwr'])
hrv = hrv_quality(hrv, r3_th = 2.2, r4_cor_th = .75)
hrv_good = hrv[hrv['q']]
print(f'ECG good segments: {round(100*len(hrv_good)/len(hrv),1)}%')
plt.plot(hrv_good['dt'],hrv_good['rmssd_30s'])

signal = nk.data("ecg_1000hz")
hrv = hrv_process(signal, sf = 1000, window = 30, slide = 5, metrics = ['time','freq','bsi','ans','r_rr', 'nl','pwr'])
hrv = hrv_quality(hrv, r3_th = 2.2, r4_cor_th = .75)
hrv_good = hrv[hrv['q']]
print(f'ECG good segments: {round(100*len(hrv_good)/len(hrv),1)}%')
plt.plot(hrv_good['dt'],hrv_good['rmssd_30s'])

ecg_signal = nk.data("ecg_3000hz")
hrv = hrv_process(ecg_signal, sf = 3000, window = 10, slide = 3, metrics = ['time','freq','bsi','ans','r_rr', 'nl','pwr'])
hrv = hrv_quality(hrv, r3_th = 2.2, r4_cor_th = .75)
hrv_good = hrv[hrv['q']]
print(f'ECG good segments: {round(100*len(hrv_good)/len(hrv),1)}%')
plt.plot(hrv_good['dt'],hrv_good['hr_10s'])

ecg_signal = nk.ecg_simulate(duration=3600, sampling_rate=512)
hrv = hrv_process(ecg_signal, sf = 512, window = 60, slide = 20, metrics = ['time','freq','bsi','ans','r_rr', 'nl','pwr'])
hrv = hrv_quality(hrv, r3_th = 2.2, r4_cor_th = .75)
hrv_good = hrv[hrv['q']]
print(f'ECG good segments: {round(100*len(hrv_good)/len(hrv),1)}%')
plt.plot(hrv_good['dt'],hrv_good['rmssd_60s'])

# Movesense MD ECG CSV 512Hz
ecg_data = pd.read_csv(os.path.join('qskit','data','ecg','MovesenseMD_EcgActivityGraphView.csv'))
ecg_signal = ecg_data['Count']
hrv = hrv_process(ecg_signal, sf = 512, window = 30, slide = 5, metrics = None)
hrv = hrv_quality(hrv, r3_th = 2.2, r4_cor_th = .75)
hrv_good = hrv[hrv['q']]
print(f'ECG good segments: {round(100*len(hrv_good)/len(hrv),1)}%')
plt.plot(hrv_good['dt'],hrv_good['hr_30s'])

# Shimmer3 ECG 512Hz real data
ecg_data = pd.read_csv(os.path.join('qskit','data','ecg','Shimmer3_ECG_Calibrated.csv'))
ecg_signal = ecg_data['EXG_ADS1292R_1_CH2_24BIT']
hrv = hrv_process(ecg_signal, sf = 512, window = 60, slide = 20, metrics = ['time','freq','bsi','ans','r_rr', 'nl','pwr'])
hrv = hrv_quality(hrv, r3_th = 2.2, r4_cor_th = .75)
hrv_good = hrv[hrv['q']]
print(f'ECG good segments: {round(100*len(hrv_good)/len(hrv),1)}%')
plt.plot(hrv_good['dt'],hrv_good['rmssd_60s'])