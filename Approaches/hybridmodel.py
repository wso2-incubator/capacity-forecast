
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Input, Model
from keras.layers import Dropout, Dense
from tcn import TCN
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

dfn = pd.read_csv('./../Engine/data/280.csv')
dfn.head()

WINDOW_SIZE = 5
FEED_LEN = 50
PREDICT_LEN = 6
print('processing data...')
def get_train_data_uni(df):
    global df_min, df_max
    df_min = df.min()
    df -= df_min
    df_max = df.max()
    df /= df_max
    cpu_values = df['AWS/EC2 CPUUtilization'].values
    features = []
    labels = []
    for i in range(FEED_LEN, cpu_values.shape[0]-PREDICT_LEN):
        features.append(cpu_values[i-FEED_LEN:i])
        labels.append(cpu_values[i:i+WINDOW_SIZE])
    features = np.array(features)
    labels = np.array(labels)
    features = features.reshape(features.shape[0], features.shape[1], 1)
    return features, labels

f_tr, f_lb = get_train_data_uni(dfn)

i = Input(batch_shape=(None, None, 1))
o = TCN(return_sequences=False)(i)
o = Dense(WINDOW_SIZE)(o)
m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mae', metrics=['mse'])
m.summary()

history = m.fit(f_tr, f_lb, epochs=5, batch_size=64)
m.save('Models/tcn.h5')

next_steps = m.predict(f_tr[-1].reshape(1, FEED_LEN, 1))
cpu_values = dfn['AWS/EC2 CPUUtilization'].values
cpu = np.concatenate((cpu_values, next_steps.flatten()))
print(cpu_values.shape, cpu.shape)

def smooth(arr):
    arr2 = np.zeros(arr.shape[0]-WINDOW_SIZE)
    for i in range(WINDOW_SIZE, arr.shape[0]-WINDOW_SIZE):
        arr2[i] = np.max(arr[i-WINDOW_SIZE:i+WINDOW_SIZE])
    arr2 = gaussian_filter1d(arr2, sigma=2)
    return arr2

plt.figure(figsize = (18, 14), dpi=80)
cpu_smoothed = smooth(cpu)
plt.plot(cpu_smoothed, color='blue')
plt.plot(cpu_values, color='green')
plt.xlim(25000,cpu.shape[0])
plt.savefig('Results/two_model_after_smoothing.jpg')
plt.show()

cpur = cpu_values.reshape(-1, 1)
smr = cpu_smoothed.reshape(-1, 1)
features = np.concatenate((cpur, smr), axis=1)

feature_set = np.array(features[:FEED_LEN, :])
feature_set = feature_set.reshape(1, feature_set.shape[0], feature_set.shape[1])

for i in range(FEED_LEN+1, features.shape[0]-PREDICT_LEN):
  temp = features[i-FEED_LEN:i,:]
  temp = temp.reshape(1, FEED_LEN, features.shape[1])
  feature_set = np.concatenate((feature_set, temp), axis=0)

label_set = np.array(cpu_smoothed[FEED_LEN:FEED_LEN+PREDICT_LEN])
label_set = label_set.reshape(1, PREDICT_LEN)

for i in range(FEED_LEN+PREDICT_LEN+1, features.shape[0]):
  temp = cpu_smoothed[i-PREDICT_LEN:i]
  temp = temp.reshape(1, PREDICT_LEN)
  label_set = np.concatenate((label_set, temp), axis=0)

mdl = Sequential()
mdl.add(LSTM(units=FEED_LEN, return_sequences=True, input_shape=(FEED_LEN, 2)))
mdl.add(LSTM(units=FEED_LEN, return_sequences=True))
mdl.add(LSTM(units=32, return_sequences=False))
mdl.add(Dense(units=PREDICT_LEN))
mdl.compile(optimizer='adam', loss='mae', metrics=['mae', 'mape', 'mse'])
mdl.summary()

X_train, x_test, Y_train, y_test = train_test_split(feature_set, label_set, train_size=0.8, shuffle=False)
print(X_train.shape)
print(Y_train.shape)
print(x_test.shape)
print(y_test.shape)

historyLstm = mdl.fit(X_train, Y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
mdl.save('Models/two_model_lstm.h5')

plt.plot(historyLstm.history['loss'])
plt.plot(historyLstm.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Results/two_model_model_loss.jpg')
plt.show()

plt.plot(historyLstm.history['mean_absolute_percentage_error'])
plt.plot(historyLstm.history['val_mean_absolute_percentage_error'])
plt.title('mape')
plt.ylabel('mape')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Results/two_model_mape.jpg')
plt.show()

true = x_test[-1].reshape(1, x_test.shape[1], x_test.shape[2])
for i in range(x_test.shape[0]-2, 0, -PREDICT_LEN):
  temp = x_test[i].reshape(1, x_test.shape[1], x_test.shape[2])
  true = np.concatenate((temp, true), axis=0)

pred = mdl.predict(true).flatten()

_, actual = train_test_split(cpu_values, train_size=0.8, shuffle=False)
nans = np.empty((PREDICT_LEN))
nans[:] = np.nan
actual = np.concatenate((actual, nans), axis=0)

diff = actual.shape[0] - pred.shape[0]
nans = np.empty((diff))
nans[:] = np.nan
pred = np.concatenate((nans, pred), axis=0)

plt.figure(figsize=(18, 14), dpi=80)
plt.plot(actual, color='blue', label='Actual')
plt.plot(pred, color='red', label='Predicted')
plt.legend()
plt.xlim(pred.shape[0] - 1000, pred.shape[0])
plt.title('actual vs predicted')
plt.savefig('Results/two_model_actual_vs_predicted.jpg')
plt.show()

def error_calc(act, prd):
  actNans = np.isnan(act)
  act[actNans] = 0
  prdNans = np.isnan(prd)
  prd[prdNans] = 0
  exceedScore = 0
  optimizeScore = 0
  for i in range(0, act.shape[0], PREDICT_LEN):
    actTemp = act[i: PREDICT_LEN+i]
    prdTemp = prd[i:PREDICT_LEN+i]
    optimizeScore += np.sum(np.abs(actTemp - prdTemp))
    if np.max(actTemp) > np.max(prdTemp):
      exceedScore += np.abs(np.max(actTemp) - np.max(prdTemp)) * PREDICT_LEN
  return (exceedScore/act.shape[0], optimizeScore/act.shape[0])

exceedValue, optimScore = error_calc(actual, pred)
print('Exceeding Score:', exceedValue)
print('Optimize score', optimScore)