import numpy as np
import scipy as sp

from scipy.signal import butter, freqz, lfilter, medfilt
def median_filter(signal, window):
  filtered_signal = medfilt(signal, kernel_size=int(window))
  return(filtered_signal)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    return b, a

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
  
def butter_lowpass_filter(data, highcut, fs, order=3):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
  
def butter_highpass_filter(data, lowcut, fs, order=3):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

import matplotlib.pyplot as plt
def butter_bandpass_plot(lowcut, highcut, fs, order_f):
    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
    b, a = butter_bandpass(lowcut, highcut, fs, order=order_f)
    return([b, a])

from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator, Akima1DInterpolator
def sc_interp1d(x, y, desired_len, m = 'pchip'):
  new_x = np.linspace(x[0], x[-1], desired_len)
  if m == 'akima':
    akima_interp = Akima1DInterpolator(x, y)
    new_y = akima_interp(new_x)
  elif m == 'pchip':
    pchip_interp = PchipInterpolator(x, y)
    new_y = pchip_interp(new_x)
  elif m == 'cubic':
    cubic_interp = CubicSpline(x, y)
    new_y = cubic_interp(new_x)
  elif m == 'natural':
    cubic_interp = CubicSpline(x, y, bc_type='natural')
    new_y = cubic_interp(new_x)
  else:
    new_y = sp.interpolate.interp1d(x, y, kind=m)(new_x)
  return([new_x, new_y])

def sc_interp(y, desired_len, m = 'pchip'):
  y = np.array(y)
  x = np.arange(len(y))
  new_x = np.linspace(x[0], x[-1], desired_len)
  if m == 'akima':
    akima_interp = Akima1DInterpolator(x, y)
    new_y = akima_interp(new_x)
  elif m == 'pchip':
    pchip_interp = PchipInterpolator(x, y)
    new_y = pchip_interp(new_x)
  elif m == 'cubic':
    cubic_interp = CubicSpline(x, y)
    new_y = cubic_interp(new_x)
  elif m == 'natural':
    cubic_interp = CubicSpline(x, y, bc_type='natural')
    new_y = cubic_interp(new_x)
  else:
    new_y = sp.interpolate.interp1d(x, y, kind=m)(new_x)
  return(new_y)

def sc_interp1d_nan(y, m = 'pchip', extrapolate = False):
  y = np.array(y)
  x = np.arange(len(y))
  nan_indices = np.isnan(y); y_interp = []
  if m == 'akima':
    akima_interp = Akima1DInterpolator(x[~nan_indices], y[~nan_indices])
    y_interp = akima_interp(x, extrapolate = extrapolate)
  elif m == 'pchip':
    pchip_interp = PchipInterpolator(x[~nan_indices], y[~nan_indices])
    y_interp = pchip_interp(x, extrapolate = extrapolate)
  elif m == 'cubic':
    cubic_interp = CubicSpline(x[~nan_indices], y[~nan_indices])
    y_interp = cubic_interp(x, extrapolate = extrapolate)
  elif m == 'natural':
    cubic_interp = CubicSpline(x[~nan_indices], y[~nan_indices], bc_type = 'natural')
    y_interp = cubic_interp(x, extrapolate = extrapolate)
  elif m == 'np_linear':
    y_interp = np.interp(x, x[~nan_indices], y[~nan_indices])
  elif m == 'sc_linear':
    f = interp1d(x[~nan_indices], y[~nan_indices], bounds_error=False, kind='linear',assume_sorted=True,copy=False)
    y_interp = f(np.arange(y.shape[0]))
  else:
    f = interp1d(x[~nan_indices], y[~nan_indices], bounds_error=False, kind=m,assume_sorted=True,copy=False)
    y_interp = f(np.arange(y.shape[0]))
  return(y_interp)
