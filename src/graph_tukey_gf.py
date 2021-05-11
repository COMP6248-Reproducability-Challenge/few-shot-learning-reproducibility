import matplotlib.pyplot as plt
import json
import numpy as np

generated_features = {
    'tukey': {},
    'notukey': {}
}

with open("results/result_train_hyper_gen_features.json") as f:
    generated_features['tukey_data'] = json.load(f)

with open("results/result_train_hyper_gen_features_notukey.json") as f:
    generated_features['notukey_data'] = json.load(f)

for key, data in generated_features['tukey_data'].items():
    generated_features['tukey'][float(key)] = data['newTrain_all'][-1][0][0]

for key, data in generated_features['notukey_data'].items():
    generated_features['notukey'][float(key)] = data['newTrain_all'][-1][0][0]

tukey = {
    'with_gf': {},
    'without_gf': {}
}
# Tukey
with open("results/result_train_hyper_tukey.json", 'r') as f:
    tukey['data'] = json.load(f)

for key, data in tukey['data'].items():
    tukey['with_gf'][float(key)] = data['newTrain_all'][-1][0][0]
    tukey['without_gf'][float(key)] = data['allTrain_all'][-1][0][0]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax[1].set_title("Generated Features")
ax[1].plot(generated_features['tukey'].keys(), generated_features['tukey'].values(), label="w/ Tukey")
ax[1].plot(generated_features['notukey'].keys(), generated_features['notukey'].values(), label="w/o Tukey")
ax[1].set_xlabel("Number of generated features per class")
ax[1].set_ylabel("Test Accuracy (5way-1shot)")
ax[1].legend()

ax[0].set_title("Tukey Transformation")
ax[0].plot(tukey['with_gf'].keys(), tukey['with_gf'].values(), label="w/ generated features")
ax[0].plot(tukey['without_gf'].keys(), tukey['without_gf'].values(), label="w/o generated features")
ax[0].legend()
ax[0].set_xlabel("Values of power λ in Tukey transformation")
ax[0].set_ylabel("Test Accuracy (5way-1shot)")

plt.tight_layout()
plt.show()
