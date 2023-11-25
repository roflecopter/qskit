import numpy as np
import pandas as pd
import math
import os
import datetime
import logging
import neurokit2 as nk
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.preprocess.preprocess_signal import smooth_signal,taper_signal
from vital_sqi.common.band_filter import BandpassFilter
from hrvanalysis import remove_ectopic_beats
from ..misc import now, spd
from ..signal import sc_interp1d_nan, butter_bandpass_filter
from .sqi import peaks_sqi, beats_cor_sqi
from .hrv_segment import hrv_segment

logger = logging.getLogger("qskit")
logger.setLevel(logging.INFO)

def hrv_process(
        signal, 
        sf, 
        type = 'ECG', 
        window = 60, 
        slide = 30, 
        min_hr = 30, 
        max_hr = 220, 
        metrics = None,
        dts = None,
        user = 'user',
        device = 'device',
        cache_dir = None,
        verbose = False,
        debug = False
):
    accepted = ['ECG','PPG','RPeaks','RR']
    if type not in accepted :
        logger.warning(f'wrong type selected, must be one of {accepted}')
        return None
    if dts is None: dts = datetime.datetime.now()
    hrv_cache_tag = f'hrv_p{window}_s{slide}'; 
    # sliding window in samples
    s_slide = slide * sf

    # if input is rpeaks or rr, then creaty empty signal
    if type == 'RPeaks':
        rpeaks_all = signal
        signal = np.arange(0, rpeaks_all[-1] + 1)
    elif type == 'RR':
        rpeaks_all = np.cumsum(np.append(0, signal))
        signal = np.arange(0, rpeaks_all[-1] + 1)
    segment_len = (window * sf - 1)
    signal_start = 0; signal_end = int(sf) * math.floor(len(signal) / sf) - segment_len;

    # define at each progress percentage to append results into cache file and print
    progress_percent_step = 5; progress_step = s_slide*round(signal_end/((100/progress_percent_step)*s_slide))
    if progress_step == 0: progress_step = window * sf * 60

    # clean signal
    if type == 'ECG':
        signal_clean = nk.ecg_clean(signal, sf, method = 'neurokit')
    elif type == 'PPG':
        # https://www.mdpi.com/2073-8994/14/6/1139
        # The ECG and the PPG bandpass filters were set to 
        # 0.5 to 35 Hz [62,63] and 0.4 to 4 Hz, respectively
        signal_clean = butter_bandpass_filter(signal, .4, 4, sf, 4)
    elif type in ['RPeaks','RR']:
        signal_clean = signal
    hrv_neurokit = None; hrv_nk = None
    
    # load cache
    if cache_dir is not None: 
        hrv_progress_cache_file = os.path.join(cache_dir, f'{hrv_cache_tag}_progress-{user}-{dts.strftime("%Y_%m_%d-%H_%M_%S")}.csv')
        if os.path.isfile(hrv_progress_cache_file): 
            try: hrv_neurokit = pd.read_csv(hrv_progress_cache_file)
            except: hrv_neurokit = None
        if hrv_neurokit is not None: 
            last_ss = max(hrv_neurokit['ss'])
            if  last_ss is not None:
                signal_start = int(last_ss)
            else:
                hrv_neurokit = None
        else: 
            hrv_neurokit = pd.DataFrame()

    slided = 0; slided_past = 0
    ss_started = now(); ss_first = now()
    if signal_start < signal_end:
        for i in range(signal_start, signal_end):
            ss = i; se = ss + segment_len
            if ((i % s_slide == 0) | debug) and (se < signal_end):    
                slided = slided + 1
                se_dt = dts + datetime.timedelta(seconds = se/sf)
                if i % progress_step == 0:
                    ss_past = (now() - ss_started).total_seconds()
                    s_past = (slided - slided_past)*window/60
                    logger.info(f'{device} {hrv_cache_tag} {round((ss/sf)/60)}m {round(100*i/signal_end)}% {round(s_past/ss_past,1)} s-m/s {spd(ss_first, ss_started)} at {dts.strftime("%Y-%m-%d %H:%M")}')
                    ss_started = now(); slided_past = slided
                segment_clean = signal_clean[ss:se]
                try:
                    if type == 'ECG':
                        # https://www.samproell.io/posts/signal/ecg-library-comparison/
                        rpeaks_res = nk.ecg_findpeaks(segment_clean, sampling_rate=sf, method='neurokit')
                        if rpeaks_res is not None:
                            rpeaks = rpeaks_res[f'{type}_R_Peaks']
                            peaks_n = len(rpeaks)
                        else:
                            logger.info(f'no peaks found: {rpeaks_res}')
                            peaks_n = 0
                    elif type == 'PPG':
                        detector = PeakDetector(wave_type='ppg',fs=sf)
                        rpeaks, trough_list = detector.ppg_detector(segment_clean, detector_type=1)
                        if rpeaks is not None:
                            rpeaks = np.array(rpeaks)
                            peaks_n = len(rpeaks)
                        else:
                            logger.info(f'no peaks found: {rpeaks_res}')
                            peaks_n = 0
                    elif type in ['RPeaks','RR']:
                        rpeaks = rpeaks_all[(rpeaks_all >= ss) & (rpeaks_all < se)]
                        peaks_n = len(rpeaks)
                except Exception as error:
                    # handling neurokit no peaks found issue https://github.com/neuropsychology/NeuroKit/issues/580
                    logger.warning(error)
                    peaks_n = 0
                if peaks_n > 2:
                    r1, r2, r3_v = peaks_sqi(rpeaks, window, min_hr, max_hr)
                    if not r1:
                        if type in ['ECG','PPG']:
                            r4_cor = beats_cor_sqi(segment_clean, rpeaks, sf)
                        elif type in ['RPeaks','RR']:
                            r4_cor = np.nan
                        # 1st round of R-peaks correction: Kubios method
                        info, rpeaks_corrected = nk.signal_fixpeaks(rpeaks, sampling_rate=sf, method = 'Kubios', iterative=True, show=False)
                        sf_interp = 1000; 
                        if sf == sf_interp:
                            rpeaks_corrected_ms = rpeaks_corrected
                        else:
                            rpeaks_corrected_ms = rpeaks_corrected * sf_interp/sf # change sf to 1000 hz for future interpolation
                        rr_corrected_ms = np.diff(rpeaks_corrected_ms)
                        # 2nd round of R-peaks correction: removing ectopic beats with hrvanalysis library
                        rr_corrected_ectopic_ms = remove_ectopic_beats(rr_intervals=rr_corrected_ms, method="malik", verbose = False)
                        rr_final_ms = sc_interp1d_nan(rr_corrected_ectopic_ms, m = "pchip")
                        # interpolation by default does not extrapolate, to avoid non-increasing values
                        # checking if there are bounds with nans, replace with mean rr
                        if np.isnan(rr_final_ms[0]):
                            rr_final_ms[0] = np.nanmean(rr_final_ms)
                        if np.isnan(rr_final_ms[-1]):
                            rr_final_ms[-1] = np.nanmean(rr_final_ms)                            
                        rpeaks_final = np.cumsum(np.append(rpeaks_corrected_ms[0], rr_final_ms))*sf/sf_interp
                        rpeaks_final_n = len(rpeaks_final)
                        # sometimes interpolation results in start peaks being negative, 
                        # ignore these segments as this due to removed corner beats
                        if(min(rpeaks_final) > 0):
                            corrected = np.where(~np.isin(rpeaks, rpeaks_final))[0].astype(int)
                            ectopic = np.unique(np.concatenate((info['ectopic'], rpeaks_corrected[np.where(np.isnan(rr_corrected_ectopic_ms))[0]]))).astype(int)
                            missed = info['missed']
                            longshort = info['longshort']
                            extra = info['extra']
                            artifacts = np.unique(np.concatenate((ectopic, missed, longshort, extra, corrected))).astype(int)
                            artifacts_n = len(artifacts); rpeaks_final_n = rpeaks_final_n
                            r1, r2, r3_v = peaks_sqi(rpeaks_final, window, min_hr, max_hr)
                            hrv_nk = None
                            if metrics is None:
                                hrv_nk = {f'hr_{window}s': rpeaks_final_n*60/(window)}
                            else:
                                try:
                                    hrv_nk = hrv_segment(rpeaks_final, sf, hf_ex = [9/60,1.5], window = window, metrics = metrics)
                                except Exception as error:
                                    logger.warning(error)
                                    hrv_nk = None
                            if hrv_nk is not None:
                                hrv_ext = {'ss':ss,'n':rpeaks_final_n,'artifacts_n':artifacts_n,
                                           'artifacts_rate':artifacts_n/rpeaks_final_n,'dt':se_dt,
                                           'ectopic':len(ectopic),'missed':len(missed),'extra':len(extra),
                                           'longshort':len(longshort),'corrected':len(corrected),
                                           'r1':r1,'r2':r2,'r3_v':r3_v,'r4_cor':r4_cor}
                                hrv_nk.update(hrv_ext)
                                hrv_neurokit = pd.concat([hrv_neurokit,pd.DataFrame.from_dict([hrv_nk])])
                if verbose: 
                    if hrv_nk is not None:
                        if f'rmssd_{window}s' in hrv_nk.keys():
                            logger.info(f'{device} {hrv_cache_tag} {round((ss/sf)/60)}m {round(100*i/signal_end)}% | hr: {round(np.nanmean(hrv_nk[f"hr_{window}s"]))}, rmssd: {round(np.nanmean(hrv_nk[f"rmssd_{window}s"]))} | {spd(ss_first, ss_started)} at {dts.strftime("%Y-%m-%d %H:%M")}')
                        elif f'hr_{window}s' in hrv_nk.keys():
                            logger.info(f'{device} {hrv_cache_tag} {round((ss/sf)/60)}m {round(100*i/signal_end)}% | hr: {round(np.nanmean(hrv_nk[f"hr_{window}s"]))} | {spd(ss_first, ss_started)} at {dts.strftime("%Y-%m-%d %H:%M")}')
                    else:
                        logger.info(f'{device} {hrv_cache_tag} {round((ss/sf)/60)}m {round(100*i/signal_end)}% | {spd(ss_first, ss_started)} at {dts.strftime("%Y-%m-%d %H:%M")}')                        
            if debug:
                logger.info(hrv_neurokit)
                break
            if i % progress_step == 0 and cache_dir is not None:
                hrv_neurokit.to_csv(hrv_progress_cache_file, header = True, index = False)
        if (i >= signal_end-1) and not debug:
            # cache final result
            hrv_neurokit_final = hrv_neurokit
            if cache_dir is not None:
                if len(hrv_neurokit_final) > 0:
                    hrv_cache_file = os.path.join(cache_dir, f'{hrv_cache_tag}-{user}-{dts.strftime("%Y_%m_%d-%H_%M_%S")}.csv')
                    hrv_neurokit_final.to_csv(hrv_cache_file, header = True, index = False)
                # remove temporary cache
                os.remove(hrv_progress_cache_file)
            return hrv_neurokit_final
    else:
        return hrv_neurokit
    return None
