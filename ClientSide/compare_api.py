import matplotlib.pyplot as plt
import json
import requests
import numpy as np


url_compare = 'http://52.15.69.21:80/compare'
req_compare = requests.get(url_compare)

true, predict = req_compare.json()['true'], req_compare.json()['predict']
true = (np.array(true)-0.5)*100
predict = (np.array(predict)-0.5)*100
plt.figure()
plt.plot(true, color='blue', label='true')
plt.plot(predict, color='red', label='predict')
plt.title('True vs Predict')
plt.xlabel('Timesteps')
plt.ylabel('CPU utilization')
plt.savefig('Plots_TCN/True vs Predict')
plt.show()
