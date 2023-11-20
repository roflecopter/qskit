import numpy as np
import scipy.sparse
def signal_detrend_tarvainen2002(signal, regularization=500):
  # https://github.com/neuropsychology/NeuroKit/pull/569
  # https://stackoverflow.com/questions/69810723/issues-with-signal-detending-using-smoothness-priors-on-the-last-detended-elemen
  # The paper says a second-order difference matrix (N-3)x(N-1). To do this you have to do:
  # B = np.dot(np.ones((N, 1)), np.array([[1, -2, 1]]))
  # D_2 = sp.sparse.dia_matrix((B.T, [0, 1, 2]), shape=(N - 3, N-1))
  N = len(signal)
  identity = np.eye(N)
  B = np.dot(np.ones((N, 1)), np.array([[1, -2, 1]]))
  D_2 = scipy.sparse.dia_matrix((B.T, [0, 1, 2]), shape=(N - 2, N))  # pylint: disable=E1101
  inv = np.linalg.inv(identity + regularization ** 2 * D_2.T @ D_2)
  z_stat = ((identity - inv)) @ signal
  trend = np.squeeze(np.asarray(signal - z_stat))
  # detrend
  detrended = np.array(signal) - trend
  return detrended
