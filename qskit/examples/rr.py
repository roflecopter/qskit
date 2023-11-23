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

# Polar Flow Session export to CSV
rr_data = pd.read_csv(os.path.join('qskit','data','RR','PolarFlowExport_RR.CSV'))
metrics = ['time','freq','bsi','ans','r_rr', 'nl','pwr']
hrv = hrv_process(rr_data['duration'], sf = 1000, type = 'RR', window = 30, slide = 5, metrics = metrics)
hrv = hrv_quality(hrv, r3_th = 2.2, r4_cor_th = .75)
hrv_good = hrv[hrv['artifacts_rate'] < 0.1]
print(f'RR good segments: {round(100*len(hrv_good)/len(hrv),1)}%')
plt.plot(hrv_good['dt'],hrv_good['hr_30s'])
