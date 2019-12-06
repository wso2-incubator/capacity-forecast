import matplotlib.pyplot as plt
import requests


def show_metrics(history):
    plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('Plots_LSTM/loss.png')
    # plt.savefig('Plots_TCN/loss.png')
    plt.show()
    plt.plot(history['mean_absolute_percentage_error'])
    # plt.plot(history['val_mean_absolute_percentage_error'])
    plt.title('mape_train')
    plt.ylabel('mape')
    plt.xlabel('epoch')
    # plt.ylim(0,50)
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('Plots_TCN/mape_train.png')
    # plt.savefig('Plots_LSTM/mape_train.png')
    plt.show()


url_graph = 'http://127.0.0.1:1111/history'
req_graph = requests.get(url_graph)
# show_metrics(req_graph.json())
print(req_graph.json())