import sys 
import os 
import tomli
import json
import pickle
import numpy as np
import pandas as pd

path = sys.argv[-1]

with open(os.path.join(path, 'config.toml'), 'rb') as f: 
    ddpmconfig = tomli.load(f)
    
with open(os.path.join(ddpmconfig['real_data_path'], '../config.json')) as f:
    dataconfig = json.load(f)

cat = None 
if 'X_cat_train.npy' in os.listdir(path):
    cat = np.load(os.path.join(path, 'X_cat_train.npy'), allow_pickle=True)

num = None 
if 'X_num_train.npy' in os.listdir(path):
    num = np.load(os.path.join(path, 'X_num_train.npy'))
    
y = np.load(os.path.join(path, 'y_train.npy'))
if 'label_encoder.pkl' in os.listdir(ddpmconfig['real_data_path']):
    with open(os.path.join(ddpmconfig['real_data_path'], 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    y = label_encoder.inverse_transform(y)
y = np.expand_dims(y, axis=1)
print(y)

if cat is not None and num is not None:
    data = np.concatenate([cat, num, y], axis=1)
    cols = dataconfig['ords']+dataconfig['nums']+dataconfig['labs']
elif cat is not None:
    data = np.concatenate([cat, y], axis=1)
    cols = dataconfig['ords']+dataconfig['labs']
elif num is not None:
    data = np.concatenate([num, y], axis=1)
    cols = dataconfig['nums']+dataconfig['labs']

print(num.shape)
print(data, cols)
df = pd.DataFrame(data, columns=cols)
df = df.sample(n=10000)
print(df)
df.to_csv(os.path.join(path, 'samplesclean.csv'), index=False)
