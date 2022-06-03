import pickle

with open('meta_data.pickle', 'rb') as f:
    datas = pickle.load(f)

for data in datas:
    print(data)