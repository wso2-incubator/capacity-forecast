import requests

url_val = 'http://127.0.0.1:1111/evaluate'

req_ts = {
        'file': open('/home/sandun/Desktop/CPU/RND/280.csv', 'rb')
        }
# Testing on the data-set
res_ts = requests.post(url_val, files=req_ts)
print(res_ts.json())




