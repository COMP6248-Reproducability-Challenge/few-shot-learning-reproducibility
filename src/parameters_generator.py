import json
import sys

def create_file(loc, paras):
    with open(loc, 'w') as f:
        json.dump(paras, f, indent=4)
    return paras

name = "miniImageNet2"

paras = {
    "allTrain": "data/train/all", # all train data location
    "allValid": "data/valid/all", # all valid data location
    "allTest": "data/test/all", # all test data location
    "all_novel_indices": [3, 5, 6, 7, 12], #novel indices
    "do_tukey": True,
    "train_hyperparameters": ['tukey', 1, 2, 3], #if not false will hillclimb for each hyperparameter instead of doing comparative stuff, using which hyperparmeter, min, max, steps. Just have this as a mixed list [hyperparameter, min, max, steps]
    "k": 1, # hyperparameter, number of nearest base classes to statistics transfer, in paper 1 or 5
    "tukey_parameter": 0.5, # hyperparameter
    "alpha": 0.2, # hyperparameter that determines the degree of dispersion of features sampled from the calibrated distribution
    "generated_features": 200, # hyperparameter, number of generated features in the novel classes
    # the size of the images that we'll learn on - we'll shrink them from the original size for speed
    "image_size": (84, 84),
    "epochs": 10,
    "just_for_timing": False, #runs it once, it's up to you to time it
    "tsne": False, # makes t-SNE stuff
    "results": f"result_{name}.json",
    "k_shots": 1
}

create_file(f"parameter_configs/train_{name}.json", paras)

print(sys.argv)