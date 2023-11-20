import numpy as np

# A quantitative systematic review of normal values for short-term heart rate variability in healthy adults.
# https://sci-hub.ru/10.1111/j.1540-8159.2010.02841.x
# RMSSD: 42 ± 15 / lnRMSSD 3.49 ± 0.26
# SDNN: 50 ± 16 / lnSDNN 3.82 ± 0.23
# SD1: 29.69 ± 10.6 sqrt(1/2)*RMSSD: sqrt(0.5*42^2) ± sqrt(0.5*15^2)
# SD2: 64.17 ± 19.98 sqrt(2*SDNN^2 -1/2*RMSSD^2): sqrt(2*50^2-0.5*42^2) ± sqrt(2*16^2-0.5*15^2)
# SD1n: 0.31 : 0.098 29.69/(29.69+64.17) ± 0.31*29.69/(29.69+64.17)
# SD2n: 0.31 : 0.098 29.69/(29.69+64.17) ± 0.31*29.69/(29.69+64.17)
# mRR: 926 ± 90

# mHR 65.5 + 7.7 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7001906/#:~:text=Variability%20in%20individual%20resting%20heart,women%2C%202.90%20for%20men).

# https://www.academia.edu/35296847/Methodical_recommendations_USE_KARDiVAR_SYSTEM_FOR_DETERMINATION_OF_THE_STRESS_LEVEL_AND_ESTIMATION_OF_THE_BODY_ADAPTABILITY_Standards_of_measurements_and_physiological_interpretation_Moscow_Prague_2008
# The value of SI is normally in limits from 50 up to 150 conditional units (c.u.).
# bsi 100 ± 50

# https://pubmed.ncbi.nlm.nih.gov/11686633/
# SD2^2 = sqrt(2*SDRR^2 -1/2*RMSSD^2)
# SD1^2 = 1/2 * RMSSD^2
# SD1^2 + SD2^2 = 2*SDRR^2

nRMSSD = 42; nRMSSD_SD = 15
nSDNN = 50; nSDNN_SD = 16
nSD1 = (.5*nRMSSD**2)**.5; nSD1_SD = (.5*nRMSSD_SD**2)**.5
nSD2 = (2*nSDNN**2-.5*nRMSSD**2)**.5; nSD2_SD = (2*nSDNN_SD**2-.5*nRMSSD_SD**2)**.5
nSD1n = nSD1/(nSD1+nSD2); nSD1n_SD = nSD1n * nSD1_SD / nSD1
nSD2n = nSD2/(nSD1+nSD2); nSD2n_SD = nSD2n * nSD2_SD / nSD2
nRR = 926; nRR_SD = 90
nHR = 65.5; nHR_SD = 7.7
nBSI = 100; nBSI_SD = 50

def ans(hr, meannn, rmssd, sd1, sd2, bsi):
    # z-scored distance from normal resting values
    ans = {'hr_zn': (hr - nHR) / nHR_SD}
    ans['meannn_zn'] = (meannn - nRR) / nRR_SD
    ans['rmssd_zn'] = (rmssd - nRMSSD) / nRR_SD
    ans['sd1n'] = sd1 / (sd1 + sd2)
    ans['sd2n'] = sd2 / (sd1 + sd2)
    ans['sd1n_zn'] = (ans['sd1n'] - nSD1n) / nSD1n_SD
    ans['sd2n_zn'] = (ans['sd2n'] - nSD2n) / nSD2n_SD 
    ans['sd2n_zn'] = (ans['sd2n'] - nSD2n) / nSD2n_SD 
    ans['bsi_zn'] = (bsi - nBSI) / nBSI_SD
    ans['sns'] = np.mean([ans['hr_zn'],ans['bsi_zn'],ans['sd2n_zn']])
    ans['pns'] = np.mean([ans['meannn_zn'],ans['rmssd_zn'],ans['sd1n_zn']])
    ans['ans'] = ans['sns'] -ans['pns']
    return ans

# Bayevsky Stress Index
# https://blog.cardiomood.com/post/112068958936/methods-of-hrv-analysis 
# https://www.frontiersin.org/articles/10.3389/fphys.2021.619722/full
def bsi(rr, sf):
    if sf != 1000: rr = rr * 1000 / sf
    bin_width_ms = 50; bin_breaks = max(3,round((max(rr) - min(rr)) / bin_width_ms))
    hist = np.histogram(rr, bin_breaks)
    mo = hist[1][np.argmax(hist[0])]/1000
    Amo = hist[0][np.argmax(hist[0])] / len(rr)
    Var = (np.max(rr) - np.min(rr))/1000
    return Amo / (2 * mo * Var) * 100

# experimental unbinned bsi
from statistics import mode
def bsi_unbinned(rr, sf):
    if sf != 1000: rr = rr * 1000 / sf
    mode_v = mode(rr)/1000
    vr = (np.max(rr) - np.min(rr))/1000
    num_mode = np.count_nonzero(rr == mode_v)
    amo = (num_mode/len(rr))*100
    return amo / (2 * mode_v * vr) * 100

def rRR(rr):
    rr_mean = np.mean(rr)
    return np.mean((rr[:-1] - rr_mean) * (rr[1:] - rr_mean)) / (np.sqrt(np.mean((rr[:-1] - rr_mean) ** 2) * np.mean((rr[1:] - rr_mean) ** 2)))
