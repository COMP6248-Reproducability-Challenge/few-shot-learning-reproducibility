# Plot ad hoc data instances
import json
from collections import defaultdict

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
import numpy as np
import torchbearer
from torchbearer import Trial
from torch import optim
import torch
import torch.nn.functional as F
from torch import nn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

paras = {
    "allTrain": "data/train/all",  # all train data location
    "allValid": "data/valid/all",  # all valid data location
    "allTest": "data/test/all",  # all test data location
    "all_novel_indices": [3, 6, 7, 12],  # novel indices
    "do_tukey": True,
    "train_hyperparameters": ['tukey', 1, 10, 10],
    # if not false will hillclimb for each hyperparameter instead of doing comparative stuff, using which hyperparmeter, min, max, steps. Just have this as a mixed list [hyperparameter, min, max, steps]
    "k": 1,  # hyperparameter, number of nearest base classes to statistics transfer, in paper 1 or 5
    "k_shots": 5,
    "tukey_parameter": 0.5,  # hyperparameter
    "alpha": 0.2,
    # hyperparameter that determines the degree of dispersion of features sampled from the calibrated distribution
    "generated_features": 200,  # hyperparameter, number of generated features in the novel classes
    # the size of the images that we'll learn on - we'll shrink them from the original size for speed
    "image_size": (84, 84),
    "epochs": 10,
    "just_for_timing": True,  # runs it once, it's up to you to time it
    "tsne": True,  # makes t-SNE stuff,
    "results": "results.json",
    "split": False # if not false, splits train into test and train in the given ratio
}

# train_hyperparameter names: k, tukey, alpha,genFeatures

try:
    loc = sys.argv[1]
    with open(loc, 'r') as f:
        paras = json.load(f)
except IndexError:
    pass

# the number of images that will be processed in a single step
batch_size = 128

transform = transforms.Compose([
    transforms.Resize(paras['image_size']),
    transforms.ToTensor(),  # convert to tensor
    transforms.Lambda(lambda x: tukey(x, paras['tukey_parameter'], paras['do_tukey'])),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Transform (xi) N×Ki=1 with Tukey’s Ladder of Powers
# for each class in the training set
#   Calibrate the mean µ 0 and the covariance Σ0 for class yi using xi with Equation 6
#   Sample features for class yi from the calibrated distribution as Equation 7
def tukey(features, hyperparam, do_tukey):
    if not do_tukey:
        return features
    if hyperparam != 0:
        return torch.pow(features, hyperparam)
    return torch.log(features)


def cov_inner(mean, x):
    return (x - mean) @ (x - mean).T


def cov(features, mean):
    value = 0
    for f in features:
        covf = []
        for i in range(len(f)):
            covf.append((f[i] - mean[i]) @ (f[i] - mean[i]).T)
        value = torch.stack(covf) + value
    return value / (len(features) - 1)


def to_dictionary(loader, dataset, novel_indices):
    base_classes = defaultdict(list)
    novel_classes = defaultdict(list)
    iterator = iter(loader)
    for i in range(int(len(dataset) / batch_size) + 1):
        images, labels = next(iterator)
        for j in range(len(images)):
            if labels[j].item() in novel_indices:
                novel_classes[labels[j].item()].append(images[j])
            else:
                base_classes[labels[j].item()].append(images[j])
    return base_classes, novel_classes


def from_dictionary(dictionary, feature_list, label_list):
    for label, features in dictionary.items():
        for feature in features:
            feature_list.append(feature)
            label_list.append(label)
    return feature_list, label_list


def training_procedure(all_dataset, novel_indices, batch_size):
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)
    base_classes, novel_classes = to_dictionary(all_loader, all_dataset, novel_indices)
    base_means = {}
    base_covs = {}
    k = paras["k"]
    alpha = paras["alpha"]
    for label, features in base_classes.items():  # get base classes' statistics
        base_means[label] = torch.mean(torch.stack(features), dim=0)
        base_covs[label] = cov(features, base_means[label])

    sampled_classes = {}
    for label, features in novel_classes.items():
        calibrated_means = []
        calibrated_covs = []
        for sample_element in features:
            euclidean_distances = {}
            for base_label, base_mean in base_means.items():
                euclidean_distances[-(torch.linalg.norm(base_mean - sample_element)) ** 2] = base_label

            # select the top k elements
            top_k = list(dict(sorted(euclidean_distances.items(), reverse=True)).items())[:k]

            # calibrate the mean and covariance for each novel class
            sum_mean = 0
            sum_cov = 0
            for dist, label_inner in top_k:
                sum_mean += base_means[label_inner]
                sum_cov += base_covs[label_inner]
            calibrated_means.append((sum_mean + sample_element) / (k + 1))
            calibrated_covs.append((sum_cov / k) + alpha)

        # sample features
        sample_features = []
        for i in range(paras["generated_features"]):
            calib_to_use_index = i % len(calibrated_means)
            sample_features.append(
                torch.normal(calibrated_means[calib_to_use_index], calibrated_covs[calib_to_use_index]))
        sampled_classes[label] = sample_features

    all_features = []
    all_labels = []
    from_dictionary(sampled_classes, all_features, all_labels)
    from_dictionary(novel_classes, all_features, all_labels)
    from_dictionary(base_classes, all_features, all_labels)

    all_together = list(zip(all_features, all_labels))

    total_classes = len(novel_classes) + len(base_classes)
    return all_together, total_classes


def rebase_to_novel(dataset, novel_indices):
    loader = DataLoader(dataset, batch_size=batch_size)
    all_features = []
    all_labels = []
    iterator = iter(loader)
    for i in range(int(len(dataset) / batch_size) + 1):
        images, labels = next(iterator)
        for j in range(len(images)):
            if labels[j].item() in novel_indices:
                all_features.append(images[j])
                all_labels.append(labels[j].item())
    return list(zip(all_features, all_labels))


def reduce_to_k(dataset, novel_indices): #TODO work and implement
    loader = DataLoader(dataset, batch_size=batch_size)
    all_features = []
    all_labels = []
    iterator = iter(loader)
    k_counter = [0] * len(novel_indices)
    for i in range(int(len(dataset) / batch_size) + 1):
        images, labels = next(iterator)
        for j in range(len(images)):
            if labels[j].item() in novel_indices:
                if k_counter[novel_indices.index(labels[j].item())] < paras['k_shots']:
                    all_features.append(images[j])
                    all_labels.append(labels[j].item())
                    k_counter[novel_indices.index(labels[j].item())] += 1
            else:
                all_features.append(images[j])
                all_labels.append(labels[j].item())
    return list(zip(all_features, all_labels))


def plot_tsne(support_set, generated_set, targetLoc):
    embeddings = TSNE(n_components=2)
    s_x = [d[0].flatten().detach().cpu().numpy() for d in support_set]
    s_y = [d[1] for d in support_set]
    g_x = [d[0].flatten().detach().cpu().numpy() for d in generated_set]
    g_y = [d[1] for d in generated_set if np.isin(d[0].flatten().detach().cpu().numpy(), s_x, invert=True).any()]
    s_x = embeddings.fit_transform(s_x)
    g_x = embeddings.fit_transform(g_x)
    fig, ax = plt.subplots()
    ax.scatter(g_x[:, 0], g_x[:, 1], c=g_y, alpha=0.2, cmap=plt.cm.Spectral)
    ax.scatter(s_x[:, 0], s_x[:, 1], c=s_y, cmap=plt.cm.Spectral, edgecolor='black')
    plt.title("t-SNE Plot of Samples and generated features")
    plt.savefig(targetLoc+".png")


# Model Definition
class CNN(nn.Module):
    def __init__(self, n_channels_in, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels_in, 30, (5, 5), padding=0)
        self.conv2 = nn.Conv2d(30, 15, (3, 3), padding=0)
        self.fc1 = nn.Linear(5415, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, n_classes)

    def forward(self, x):
        # YOUR CODE HERE
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))
        out = F.dropout(out, 0.2)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


def train_model(train, valid, test, classes, batch, num_epochs):
    # model = CNN(3, classes)
    model = models.resnet18()
    train_loader = DataLoader(train, batch_size=batch)
    val_loader = DataLoader(valid, batch_size=batch)
    test_loader = DataLoader(test, batch_size=batch)

    # define the loss function and the optimiser
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters())

    trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy'], verbose=1).to(device)
    trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
    trial.run(epochs=num_epochs)
    results_test = trial.evaluate(data_key=torchbearer.TEST_DATA)
    results_val = trial.evaluate(data_key=torchbearer.VALIDATION_DATA)
    print(results_test)

    return results_test, results_val


def mean_and_confidence_interval_inner(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    #confidence for 95%
    plus_minus = 1.960 * std / np.sqrt(len(data))

    return mean, plus_minus


def mean_and_confidence_interval(data): # should be input as arrays of tuples, where the tuples are (test value, validation value)
    # data_t = np.transpose(data)
    # print(data)
    # print(data_t)

    test_acc = []
    val_acc = []

    for d in data:
        test_acc.append(d[0]['test_acc'])
        val_acc.append(d[1]['val_acc'])

    return mean_and_confidence_interval_inner(test_acc), mean_and_confidence_interval_inner(val_acc)


results = []
if paras['just_for_timing']: # run once
    print('just for timing')
    all_train_dataset = ImageFolder(paras['allTrain'], transform)
    if paras['split']:
        train_ids, test_ids = train_test_split(list(range(len(all_train_dataset))), test_size=paras['split'])
        all_train_dataset = Subset(all_train_dataset, train_ids)
        all_test_dataset = Subset(all_train_dataset, train_ids)
    else:
        all_test_dataset = ImageFolder(paras['allTest'], transform)
    all_val_dataset = ImageFolder(paras['allValid'], transform)
    all_train_dataset = reduce_to_k(all_train_dataset, paras['all_novel_indices'])

    allTrain_allVal_results = []
    newTrain_allVal_results = []

    run_count = 1 #probably good to have at least 30
    for i in range(run_count):
        train_dataset, total_classes = training_procedure(all_train_dataset, paras['all_novel_indices'], batch_size)
        train_dataset = reduce_to_k(train_dataset, paras['all_novel_indices'])
        allTrain_allVal_results.append(train_model(all_train_dataset, all_val_dataset, all_test_dataset, total_classes, batch_size, paras['epochs']))
        newTrain_allVal_results.append(train_model(train_dataset, all_val_dataset, all_test_dataset, total_classes, batch_size, paras['epochs']))
    results.append(mean_and_confidence_interval(allTrain_allVal_results))
    results.append(mean_and_confidence_interval(newTrain_allVal_results))

    with open(paras['results'], "w") as f:
        json.dump(results, f, indent=4)
elif type(paras['train_hyperparameters']) is list: # apply that hyperparameter a load of times, running each time to get the accuracies between min and max for the given number of steps, and output the mean and range of those.
    print("train hypers")
    step_size = (paras['train_hyperparameters'][2] - paras['train_hyperparameters'][1]) / float(
        paras['train_hyperparameters'][3]-1)
    all_train_dataset = ImageFolder(paras['allTrain'], transform)
    if paras['split']:
        train_ids, test_ids = train_test_split(list(range(len(all_train_dataset))), test_size=paras['split'])
        all_train_dataset = Subset(all_train_dataset, train_ids)
        all_test_dataset = Subset(all_train_dataset, train_ids)
    else:
        all_test_dataset = ImageFolder(paras['allTest'], transform)
    all_val_dataset = ImageFolder(paras['allValid'], transform)

    all_train_dataset = reduce_to_k(all_train_dataset, paras['all_novel_indices'])

    results = {}
    allTrain_allVal_scores = []
    newTrain_allVal_scores = []
    stepValues = []

    for i in range(paras['train_hyperparameters'][3]):
        new_hyper_value = i * step_size + paras['train_hyperparameters'][1]
        if paras['train_hyperparameters'][0] == 'k': # 1 through 10, 10 steps
            k = new_hyper_value
        elif paras['train_hyperparameters'][0] == 'tukey': #
            tukey_parameter = new_hyper_value
        elif paras['train_hyperparameters'][0] == 'alpha':
            alpha = new_hyper_value
        elif paras['train_hyperparameters'][0] == 'genFeatures':
            generated_features = new_hyper_value

        print(f'{paras["train_hyperparameters"][0]} = {new_hyper_value}')

        allTrain_allVal_results = []
        newTrain_allVal_results = []
        
        train_hyperparameter_rounds = 30 #probably good to have at least 30
        for i in range(train_hyperparameter_rounds):
            print(f'round {i}')
            train_dataset, total_classes = training_procedure(all_train_dataset, paras['all_novel_indices'], batch_size)
            train_dataset = reduce_to_k(train_dataset, paras['all_novel_indices'])
            allTrain_allVal_results.append(train_model(all_train_dataset, all_val_dataset, all_test_dataset, total_classes, batch_size, paras['epochs']))
            newTrain_allVal_results.append(train_model(train_dataset, all_val_dataset, all_test_dataset, total_classes, batch_size, paras['epochs']))

        allTrain_allVal_results.append(mean_and_confidence_interval(allTrain_allVal_results))
        newTrain_allVal_results.append(mean_and_confidence_interval(newTrain_allVal_results))

        results[new_hyper_value] = {
            'allTrain_all': allTrain_allVal_results,
            'newTrain_all': newTrain_allVal_results
        }

        with open(paras['results'], "w") as f:
            json.dump(results, f, indent=4)

elif paras['tsne']: # do tsne visualisation of novel classes showing support set (outlined) and generated set
    print("tsne")
    all_train_dataset = ImageFolder(paras['allTrain'], transform)
    all_val_dataset = ImageFolder(paras['allValid'], transform)
    novel_train_dataset = rebase_to_novel(all_train_dataset, paras['all_novel_indices'])
    generated_dataset, _ = training_procedure(all_train_dataset, paras['all_novel_indices'], batch_size)
    novel_generated_dataset = rebase_to_novel(generated_dataset, paras['all_novel_indices'])
    novel_generated_dataset = reduce_to_k(generated_dataset, paras['all_novel_indices'])
    novel_val_dataset = ImageFolder(paras['allValid'], transform)
    novel_val_dataset = rebase_to_novel(novel_val_dataset, paras['all_novel_indices'])
    plot_tsne(novel_train_dataset, novel_generated_dataset, paras['results']+"1")
    print(6)
    plot_tsne(novel_train_dataset, novel_val_dataset, paras['results']+"2")
else: # give the mean and confidence interval
    print("mean and confidence")
    all_train_dataset = ImageFolder(paras['allTrain'], transform)
    if paras['split']:
        train_ids, test_ids = train_test_split(list(range(len(all_train_dataset))), test_size=paras['split'])
        all_train_dataset = Subset(all_train_dataset, train_ids)
        all_test_dataset = Subset(all_train_dataset, train_ids)
    else:
        all_test_dataset = ImageFolder(paras['allTest'], transform)
    all_val_dataset = ImageFolder(paras['allValid'], transform)
    all_train_dataset = reduce_to_k(all_train_dataset, paras['all_novel_indices'])

    allTrain_allVal_results = []
    newTrain_allVal_results = []
    
    run_count = 30 #probably good to have at least 30
    for i in range(run_count):
        print(f"round {i}")
        train_dataset, total_classes = training_procedure(all_train_dataset, paras['all_novel_indices'], batch_size)
        train_dataset = reduce_to_k(train_dataset, paras['all_novel_indices'])
        allTrain_allVal_results.append(train_model(all_train_dataset, all_val_dataset, all_test_dataset, total_classes, batch_size, paras['epochs']))
        newTrain_allVal_results.append(train_model(train_dataset, all_val_dataset, all_test_dataset, total_classes, batch_size, paras['epochs']))
    results.append(mean_and_confidence_interval(allTrain_allVal_results))
    results.append(mean_and_confidence_interval(newTrain_allVal_results))

    with open(paras['results'], "w") as f:
        json.dump(results, f, indent=4)


# titles = ["Unchanged, novel validation", "Changed, novel validation", "Unchanged, all validation", "Changed, all validation"]
#
#
# data = {}
#
# for title, result in zip(titles, results):
#     data[title] = result

# TODO comment everything - reference functions from the paper etc.
