import matplotlib.pyplot as plt
import json
import numpy as np

k = {
    'test': {},
    'val': {}
}

with open("results/result_train_hyper_k.json") as f:
    k['data'] = json.load(f)

for key, data in k['data'].items():
    k['test'][float(key)] = data['newTrain_all'][-1][0][0]
    k['val'][float(key)] = data['newTrain_all'][-1][1][0]

alpha = {
    'test': {},
    'val': {}
}

# Tukey
with open("results/result_train_alpha.json", 'r') as f:
    alpha['data'] = json.load(f)

for key, data in alpha['data'].items():
    alpha['test'][float(key)] = data['newTrain_all'][-1][0][0]
    alpha['val'][float(key)] = data['newTrain_all'][-1][1][0]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax[0].set_title("Hyperparameter k")
ax[0].plot(k['test'].keys(), k['test'].values(), label="Testing set")
ax[0].plot(k['val'].keys(), k['val'].values(), label="Validation set")
ax[0].set_xlabel("Number of retrieved base class statistics k")
ax[0].set_ylabel("Accuracy (5way-1shot)")
ax[0].legend()

ax[1].set_title("Hyperparameter α")
ax[1].plot(alpha['test'].keys(), alpha['test'].values(), label="Testing set")
ax[1].plot(alpha['val'].keys(), alpha['val'].values(), label="Validation Set")
ax[1].legend()
ax[1].set_xlabel("Values of α added on covariance matrix")
ax[1].set_ylabel("Accuracy (5way-1shot)")

plt.tight_layout()
plt.show()
