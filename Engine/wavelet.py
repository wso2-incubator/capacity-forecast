from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_csv('/home/sandun/Desktop/2013/21.csv')


def wavelet(df):
    data = df['AWS/EC2 CPUUtilization'].values

    # t = np.linspace(-1, 1, 200, endpoint=False)
    # sig = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
    widths = np.arange(1, 11)
    cwt_matrix = signal.cwt(data, signal.ricker, widths)
    #print(cwtmatr.shape)
    # plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    # plt.show()

    abs_val = []
    for i in range(cwt_matrix.shape[1]):
        temp = [cwt_matrix[k, i] for k in range(cwt_matrix.shape[0])]
        abs_val.append(abs(sum(temp)))
    # abs_val.append(max(cwtmatr[0][i]))
    abs_val = np.array(abs_val)
    return pd.Series(abs_val)
