from keras.models import load_model
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json
from sklearn.model_selection import train_test_split

with open('config.json', 'r+') as f:
    f = json.loads(f.read())
    FEED_LEN = f['FEED_LEN']
    PREDICT_LEN = f['PREDICT_LEN']
    INPUT_DIM = f['INPUT_DIM']
    WINDOW_LEN = f['WINDOW_LEN']


def get_train_data_uni(df):
    df_min = df.min()
    df -= df_min
    df_max = df.max()
    df /= df_max
    cpu_values = df['AWS/EC2 CPUUtilization'].values
    features = []
    labels = []
    for i in range(FEED_LEN, cpu_values.shape[0]-PREDICT_LEN):
        features.append(cpu_values[i-FEED_LEN:i])
        labels.append(cpu_values[i:i+WINDOW_LEN])
    features = np.array(features)
    labels = np.array(labels)
    features = features.reshape(features.shape[0], features.shape[1], 1)
    return features, labels


def hybrid_data(dfn):
    print('Generating input data ....')

    def smooth(arr):
        arr2 = np.zeros(arr.shape[0]-WINDOW_LEN)
        for k in range(WINDOW_LEN, arr.shape[0]-WINDOW_LEN):
            arr2[k] = np.max(arr[k-WINDOW_LEN:k+WINDOW_LEN])
        arr2 = gaussian_filter1d(arr2, sigma=2)
        return arr2

    f_tr, f_lb = get_train_data_uni(dfn)
    m = load_model('tcn.h5')
    next_steps = m.predict(f_tr[-1].reshape(1, FEED_LEN, 1))
    print('Processing dataset....')
    cpu_values = dfn['AWS/EC2 CPUUtilization'].values
    cpu = np.concatenate((cpu_values, next_steps.flatten()))

    cpu_smoothed = smooth(cpu)
    cpur = cpu_values.reshape(-1, 1)
    smr = cpu_smoothed.reshape(-1, 1)
    features = np.concatenate((cpur, smr), axis=1)
    feature_set = np.array(features[:FEED_LEN, :])
    feature_set = feature_set.reshape(1, feature_set.shape[0], feature_set.shape[1])

    for i in range(FEED_LEN+1, features.shape[0]-PREDICT_LEN):
        temp = features[i-FEED_LEN:i, :]
        temp = temp.reshape(1, FEED_LEN, features.shape[1])
        feature_set = np.concatenate((feature_set, temp), axis=0)

    label_set = np.array(cpu_smoothed[FEED_LEN:FEED_LEN+PREDICT_LEN])
    label_set = label_set.reshape(1, PREDICT_LEN)
    for i in range(FEED_LEN+PREDICT_LEN+1, features.shape[0]):
        temp = cpu_smoothed[i-PREDICT_LEN:i]
        temp = temp.reshape(1, PREDICT_LEN)
        label_set = np.concatenate((label_set, temp), axis=0)
    print('Data generated ...')
    return feature_set, label_set
