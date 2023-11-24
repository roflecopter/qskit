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
working_dir = '/Volumes/Data/Storage/Dev/qs-kit'
os.chdir(working_dir)
from qskit.hrv.hrv_process import hrv_process
from qskit.hrv.sqi import hrv_quality
from qskit.signal import butter_bandpass_filter, sc_interp, median_filter
cache_dir = os.path.join(working_dir, 'qskit','examples','cache')

# Hypnodyne ZMax PPG 256Hz real data
sf = 256
raw = mne.io.read_raw_edf(os.path.join('qskit','data','ppg','ZMAX_OXY_IR_AC.edf'))
raw.pick(['OXY_IR_AC'])
ppg_signal = raw.get_data().ravel()
metrics = ['time', 'freq', 'ans', 'r_rr', 'nl', 'pwr']
hrv = hrv_process(ppg_signal, cache_dir=cache_dir, sf = sf, type = 'PPG', window = 60, slide = 20, metrics = metrics)
hrv = hrv_quality(hrv, r3_th = 2.2, r4_cor_th = .86)
hrv_good = hrv[hrv['q']]
print(f'PPG good segments: {round(100*len(hrv_good)/len(hrv),1)}%')
plt.plot(hrv_good['dt'],hrv_good['hr_60s'])
plt.plot(hrv_good['dt'],hrv_good['rmssd_60s'])
print(f"{np.mean(hrv_good['hr_60s'])} / {np.mean(hrv_good['rmssd_60s'])}")
