import numpy as np
import pandas as pd
import neurokit2 as nk
import logging
from  ..signal import sc_interp1d, signal_detrend_tarvainen2002
from .metrics import ans, bsi, rRR

logger = logging.getLogger("qskit")
logger.setLevel(logging.INFO)


import warnings
warnings.filterwarnings("ignore")

def hrv_segment(rpeaks, sf, hf_ex = [9/60,1.5], window = 60, metrics = ['time','freq','bsi','ans','r_rr', 'nl','pwr'], debug_file = None):
    sf_interp = 1000
    rr = np.diff(rpeaks)
    if debug_file is not None: 
        rr_ms = pd.DataFrame(rr * sf_interp / sf)
        rr_ms.to_csv(debug_file, header = False, index = False)
    if sf == sf_interp:
        rr_up = rr
        rpeaks_up = rpeaks
    else:
        rr_up = rr * sf_interp / sf 
        rpeaks_up = rpeaks * sf_interp / sf
    # upsample (interpolate) rr to continuous signal with sf_interp sample frequency
    interpolation_method = 'pchip'; desired_len = int(rpeaks_up[1:][-1]-rpeaks_up[0])
    rpeaks_up_interp, rr_up_interp = sc_interp1d(rpeaks_up[1:]-rpeaks_up[0], rr_up, desired_len = desired_len, m = interpolation_method)
    rpeaks_up_interp += rpeaks_up[0]

    # downsample (interpolate) rr signal to 4hz
    interpolation_method = 'pchip'
    # https://www.kubios.com/downloads/HRV-Scientific-Users-Guide.pdf
    # . In addition, the interpolation rate 
    # (by default a 4 Hz cubic spline interpolation is applied to form equidistantly sampled 
    # time series from the IBI data) and detrending method 
    # (by default smoothness priors method is used to remove very low frequency 
    # trend components) can be adjusted here. The default detrending settings 
    # will remove most of the very low frequency components (frequencies below 0.04 Hz) 
    # from the RR interval series prior to analysis
    f_detrend = 1/4; detrend_len = int(round(len(rpeaks_up_interp) / (sf_interp * f_detrend)))
    rpeaks_down_interp, rr_down_interp = sc_interp1d(rpeaks_up_interp, rr_up_interp, desired_len = detrend_len, m = interpolation_method)
    # detrend 4 Hz RR intervals
    rr_detrended = signal_detrend_tarvainen2002(rr_down_interp, 500)
    interpolation_method = 'pchip'
    # interpolate detrended signal back to sf_interp
    rpeaks_detrended_up, rr_detrended_up_interp = sc_interp1d(rpeaks_down_interp, rr_detrended, desired_len = desired_len, m = interpolation_method)
    # calculate detrended signal based on interpolated RR
    # trend_up_interp = rr_up_interp - rr_detrended_up_interp
    # find nearest trend points for each peak   
    nearest_trend_indices = np.abs(rpeaks[1:,np.newaxis] * sf_interp/sf - rpeaks_detrended_up).argmin(axis=1)
    # select RR peaks at nearest trend points and shift min detrended RR at same values as trended minimum
    rr_detrended_up = min(rr_up) - min(rr_detrended_up_interp[nearest_trend_indices]) + rr_detrended_up_interp[nearest_trend_indices]    

    hrv_time_cols = ['rmssd','sdnn']
    hrv_freq_cols = ['hf','lf','lfn','hfn']
    hrv_nl_cols = ['sd1','sd2','dfa_alpha1']
    
    # https://sci-hub.ru/10.1123/pes.19.2.192
    #  Because each spectrogram window was made of 256 successive R-R periods, the time between R-R periods after resampling was 0.25 s. 
    # HFp range was extended from resting recordings ( >0.15–0.5 Hz to >0.15–1.8 Hz) in order to remain in the high BF ranges reached at high exercise intensity. 
    # hf_ex = [11/60,1.8] #ex HF range starts from min is 11 br per minute
    
    # https://ieeexplore.ieee.org/document/1442912
    # In each window the power spectrum was estimated by the FFT technique and integrated over frequency bands defined as Low Frequency (LF: 0.05- 0.15 Hz) and High Frequency (0.20-1.5 Hz)(Figure 1, panel c).
    # hf_ex = [9/60,1.5]

    hrv = {'hr': len(rpeaks) * 60 / window}
    hrv['meannn'] = np.mean(rr_up)
    if 'time' in metrics:
      hrv_time = nk.hrv_time(np.cumsum(rr_detrended_up), sf_interp)
      hrv_time.rename(columns=lambda x: x.replace('HRV_', '').lower(), inplace=True)
      hrv.update(hrv_time[hrv_time_cols].iloc[0].to_dict())
    if 'freq' in metrics:
        hrv_freq = nk.hrv_frequency(np.cumsum(rr_detrended_up), sampling_rate=sf_interp, psd_method='welch', interpolation_rate = 100, normalize = False)
        hrv_freq.rename(columns=lambda x: x.replace('HRV_', '').lower(), inplace=True)
        hrv.update(hrv_freq[hrv_freq_cols].iloc[0].to_dict())
    if 'nl' in metrics:
        hrv_nl = nk.hrv_nonlinear(np.cumsum(rr_detrended_up), sampling_rate=sf_interp)
        hrv_nl.rename(columns=lambda x: x.replace('HRV_', '').lower(), inplace=True)
        hrv.update(hrv_nl[hrv_nl_cols].iloc[0].to_dict())
    if 'pwr' in metrics:
        # power in extended high frequency band
        pwr_hf_ex = nk.signal_power(rr_detrended_up_interp, frequency_band=hf_ex,sampling_rate=sf_interp,show=False,min_frequency=0,method="welch",max_frequency=max(hf_ex),order_criteria=None,normalize=False)
        psd = nk.signal_psd(rr_detrended_up_interp,sampling_rate=sf_interp,show=False,min_frequency=0,method="welch",max_frequency=max(hf_ex),order_criteria=None,normalize=False)
        hf_ex_psd = psd[psd['Frequency'].between(min(hf_ex), max(hf_ex))]
        # peaks frequency & power in extended high frequency band, which is related to respiration
        hrv['ex_hf_peak_freq'] = hf_ex_psd['Frequency'].iloc[np.argmax(hf_ex_psd['Power'])]
        hrv['resp'] = 1/hrv['ex_hf_peak_freq']
        hrv['ex_hf_peak_power'] = hf_ex_psd['Frequency'].iloc[np.argmax(hf_ex_psd['Power'])]
        hrv['ex_hf_power'] = pwr_hf_ex.iloc[0].iloc[0]
    if ('ans' in metrics) and (window >= 30):
        hrv['bsi'] = bsi(rr_detrended_up, sf_interp)
        hrv_ans = ans(hrv['hr'], hrv['meannn'], hrv['rmssd'], hrv['sd1'], hrv['sd2'], hrv['bsi'])
        hrv.update(hrv_ans)
    elif 'bsi' in metrics:
        hrv['bsi'] = bsi(rr_detrended_up, sf_interp)
    if ('r_rr' in metrics) and (window >= 60):
        hrv['r_rr'] = rRR(rr_detrended_up)
    return {f'{key}_{window}s': value for key, value in hrv.items()}
