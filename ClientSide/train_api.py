import requests

url_train = 'http://127.0.0.1:1111/train'

req = {
        'file': open('2.csv', 'rb')
        }

# Training on the data-set
res_tr = requests.post(url_train, files=req)
print("Training accuracy: ", res_tr.json())
