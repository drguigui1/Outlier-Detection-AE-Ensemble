# -*- coding: utf-8 -*-

import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class AdDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]


def gen_latent_size(input_size):
    '''
    Generate the size of the latent space according to the input size
    '''
    return random.randint(3, input_size // 2)

def gen_alpha_coef():
    '''
    Generate alpha coefficient to determine the difference between layers in term of size
    '''
    return round(random.uniform(0.5, 0.8), 2)

# # get the number of node for each layer (randomly)
def gen_layers_size(input_size, latent_size, alpha):
    '''
    Generate a list with the layer size
    '''
    layers_node_nb = []
    curr_layer_size = input_size
    while curr_layer_size > latent_size:
        layers_node_nb.append(curr_layer_size)
        curr_layer_size = math.floor(curr_layer_size * alpha)
    layers_node_nb.append(latent_size)
    return layers_node_nb

def create_random_mask(shape1, shape2, perc=0.2):
    '''
    Create random matrix mask (with 1 and 0)
    1: keep the weight activated
    0: remove the weight connexion

    perc: proba of 0 values
    '''
    t = torch.zeros(shape1, shape2)

    for i in range(shape1):
        for j in range(shape2):
            val = random.randint(0, 10)
            val = 1 if val >= perc * 10 else 0
            t[i][j] = val
    return t

def build_data_loaders(nb_loaders, datas, batchs_size):
    '''
    ex:
        batchs_size = [32, 64, 128, 256]
    '''
    nb_samples, nb_features = datas.shape
    inf, sup = int(nb_samples * 0.7), int(nb_samples * 0.99)
    data_loaders = []
    for _ in range(nb_loaders):
        # generate number of sample to user from the dataset in the current data loader
        nb_samples_subset = random.randint(inf, sup)
        datas_pos = np.random.choice(nb_samples, nb_samples_subset, replace=False)

        # get chosen data
        chosen_datas = datas[datas_pos]

        # generate randomly the number of batch
        batch_nb = random.randint(0, len(batchs_size) - 1)
        batch_size = batchs_size[batch_nb]

        # append the data loader
        training_dataset = AdDataset(torch.from_numpy(chosen_datas))
        data_loaders.append(DataLoader(training_dataset, shuffle=True, batch_size=batch_size))

    return data_loaders