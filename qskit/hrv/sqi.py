import numpy as np
import neurokit2 as nk

# ECG and PPG signal quality indices (SQI)
# https://pubmed.ncbi.nlm.nih.gov/25069129/

def peaks_sqi(peaks, duration, min_hr = 40, max_hr = 180):
    # 1) Rule 1: The HR extrapolated from the 10-s sample must
    #   be between 40 and 180 beats per minute (bpm). (Though
    #   it is theoretically possible to have HRs outside of these
    #   values, this is the physiologically probable range of HR
    #   for the adult population likely to use wearable sensors.)
    
    # 2) Rule 2: The maximum acceptable gap between successive
    #   R-peaks/PPG pulse-peaks is 3 s. (This rule ensures no
    #   more than one beat is missed.)
    # 3) Rule 3: The ratio of the maximum beat-to-beat interval
    #   to the minimum beat-to-beat interval within the sample
    #   should be less than 2.2. (This is a conservative limit since
    #   we would not expect the HR to change by more than 10%
    #   in a 10-s sample. We use a limit of 2.2 to allow for a single
    #   missed beat.)
    
    peaks_n = len(peaks)
    r1 = (peaks_n < min_hr * duration / 60) or (peaks_n > max_hr * duration / 60)
    rr_peaks = np.diff(peaks)
    r2 = sum(rr_peaks > 3000) > 0
    r3_v = max(rr_peaks) / min(rr_peaks)
    return [r1, r2, r3_v]

def beats_cor_sqi(ecg_signal, peaks, sf):
    # b) Adaptive template matching: Template-matching approaches have been used in the past for identifying ventricular
    #   ectopic beats [18] and heartbeats [19] in the ECG and for signal
    #   quality assessment of the PPG [20]. Regardless of the actual
    #   morphology of the QRS complexes or PPG pulse waveforms
    #   in a given ECG or PPG sample, template matching searches
    #   for regularity in a segment, which is an indicator of reliability
    #   (since a segment contaminated by artifact would be irregular in
    #     morphology). Our approach is as follows:
    #   1) Using all the detected R-peaks/PPG-pulse peaks of each
    #     ECG/PPG sample, the median beat-to-beat interval is calculated.
    #   2) Individual QRS complexes/PPG pulse waves are then extracted by taking a window, the width of which is the
    #     median beat-to-beat interval, centered on each detected
    #     R-peak/PPG-pulse peak.
    #   3) The average QRS template is then obtained by taking
    #     the mean of all QRS complexes in the sample. Similarly,
    #     the mean PPG pulse-wave template is obtained by taking
    #     the mean of all PPG pulse waves in the sample. The correlation coefficient of each individual QRS complex with
    #     the average QRS template is then calculated. Similarly, the
    #     correlation coefficient of each individual PPG pulse wave
    #     with the average PPG pulse-wave template is calculated.
    #   4) The average correlation coefficient is finally obtained
    #     by averaging all correlation coefficients over the whole
    #     ECG/PPG sample.

    segment_beats = nk.ecg_segment(ecg_signal, rpeaks=peaks, sampling_rate=sf, show=False)
    beats = []
    for beat_index, beat in segment_beats.items():            
        if np.count_nonzero(np.isnan(np.array(beat['Signal']))) == 0:
            beats.append(np.array(beat['Signal']))
    mean_beat = np.mean(beats, axis=0)
    correlations = [np.corrcoef(mean_beat, sublist)[0, 1] for sublist in beats]        
    return np.mean(correlations)

# We use a limit of 2.2 to allow for a single missed beat.)
# The optimum threshold for the average correlation coefficient was found to be 
# 0.66 for the ECG SQI and 0.86 for the PPG SQI.
# segments with good quality marked with 'q' = True
def hrv_quality(hrv, r3_th = 2.2, r4_cor_th = .66):
    hrv['q'] = False
    hrv['r3'] = hrv['r3_v'] >= r3_th
    hrv['r4'] = hrv['r4_cor'] <= r4_cor_th
    hrv.loc[~hrv['r1'] & ~hrv['r2'] & ~hrv['r3'] & ~hrv['r4'],'q'] = True
    return hrv
