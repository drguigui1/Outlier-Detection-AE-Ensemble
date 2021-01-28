# -*- coding: utf-8 -*-

import random
from torchsummary import summary
import torch
from torch import nn
from torch import optim

from utils import gen_latent_size, gen_alpha_coef, gen_layers_size, create_random_mask, build_data_loaders

class Autoencoder(nn.Module):
    def __init__(self, layers_size):
        '''
        Initialize the autoencoders parameters
        'layers_size': ['22', '11', '5'] (all the size of the layers for the encoding)

        'layers': [<Object layer>, <Object layer> ...] (list with all the layers objects)
        '''
        super().__init__()

        lays = []

        # Encoder Model
        # set the first encoding layer
        lays.append(nn.Linear(in_features=layers_size[0], out_features=layers_size[1]))

        # not need to consider first and second layer
        # first layer is the input dim
        # second layer is already append
        for idx in range(2, len(layers_size)):
            lays.append(nn.Linear(in_features=layers_size[idx - 1], out_features=layers_size[idx]))

        # Decoder Model
        # no need to consider first and last layer
        for idx in range(1, len(layers_size)):
            lays.append(nn.Linear(in_features=layers_size[::-1][idx - 1], out_features=layers_size[::-1][idx]))

        self.layers = nn.ModuleList(lays)

        # set the layers mask (to remove specific weights connexions)
        self.masks = []
        for idx in range(len(self.layers)):
            sh = self.layers[idx].weight.shape
            self.masks.append(create_random_mask(sh[0], sh[1], round(random.uniform(0.2, 0.8), 1)))

    def forward(self, input_features):
        self.layers[0].weight = nn.Parameter(self.layers[0].weight * self.masks[0])
        model = self.layers[0](input_features)
        model = torch.sigmoid(model)
        bn = nn.BatchNorm1d(num_features=self.layers[0].out_features)
        model = bn(model)
        # everything accept the last layer
        for idx in range(1, len(self.layers) - 1):
            self.layers[idx].weight = nn.Parameter(self.layers[idx].weight * self.masks[idx])
            model = self.layers[idx](model)
            model = torch.relu(model)
            dropout_lay = nn.Dropout(0.3)
            bn = nn.BatchNorm1d(num_features=self.layers[idx].out_features)
            model = dropout_lay(model)
            model = bn(model)

        # process the last layer
        self.layers[-1].weight = nn.Parameter(self.layers[-1].weight * self.masks[-1])
        model = self.layers[-1](model)
        model = torch.sigmoid(model)
        bn = nn.BatchNorm1d(num_features=self.layers[-1].out_features)
        model = bn(model)
        return model

class RandomAutoencoder:
    def __init__(self, input_size):
        self.input_size = input_size
        self.latent_size = gen_latent_size(input_size)
        self.alpha = gen_alpha_coef()
        self.layers_size = gen_layers_size(input_size, self.latent_size, self.alpha)

        self.model = Autoencoder(self.layers_size).to('cpu')
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    def print_model(self):
        print("Input size: {}".format(self.input_size))
        print("Latent size: {}".format(self.latent_size))
        print("Alpha coef: {}".format(self.alpha))
        print("Layers_size: {}".format(self.layers_size))

    def model_summary(self):
        if self.model:
            print(self.model)

    def train_model(self, n_epochs, train_loader):
        for epoch in range(n_epochs):
            loss = 0
            print("Epoch nb {}".format(epoch + 1))
            print("[=",end='')
            for idx, batch_features in enumerate(train_loader):
                batch = batch_features.to('cpu').float()

                # zero the gradient
                self.optimizer.zero_grad()

                # feedforward
                outputs = self.model.forward(batch)

                # compute the loss
                train_loss = self.criterion(outputs, batch)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                self.optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

                if idx % (len(train_loader) // 20) == 0:
                    print("=",end='')

            print("]")
            # compute the epoch training loss
            loss = loss / len(train_loader)
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, n_epochs, loss))

    def outlier_score_vector(self, datas):
        '''
        Get the outlier score vector of the autoencoder
        '''
        outputs = self.model(datas.float())
        return torch.sum((datas - outputs)**2, dim=1)


class EnsembleRandomAutoencoder:
    def __init__(self, batchs_size, nb_autoencoder, datas, input_size, nb_epochs=10):
        self.nb_autoencoder = nb_autoencoder
        self.nb_epochs = nb_epochs
        self.autoencoders = []

        # build all the autoencoders
        for _ in range(nb_autoencoder):
            self.autoencoders.append(RandomAutoencoder(input_size))

        # build the data loaders
        self.data_loaders = build_data_loaders(nb_autoencoder, datas, batchs_size)

    def display_loaders_info(self):
        '''
        Display information about the data loaders
        '''
        for idx, data_loader in enumerate(self.data_loaders):
            print("Data Loader nb: {}".format(idx + 1))
            print("Nb batch:       {}".format(len(data_loader)))
            for elm in data_loader:
                batch_size = elm.shape[0]
                break
            print("Batch size:     {}".format(batch_size))
            print("Nb samples:     {}".format(batch_size * len(data_loader)))
            print("-------------")

    def models_summary(self):
        '''
        Summary of all the models
        '''
        for idx, autoencoder in enumerate(self.autoencoders):
            print("Model nb {}".format(idx + 1))
            autoencoder.model_summary()
            print("-------------")

    def print_models(self):
        '''
        Print all the autoencoders models
        '''
        for idx, autoencoder in enumerate(self.autoencoders):
            print("Model nb {}".format(idx + 1))
            autoencoder.print_model()
            print("-------------")

    def fit(self):
        '''
        Fit all the autoencoders
        '''
        for idx, autoencoder in enumerate(self.autoencoders):
            print("TRAINING ae nb. {}".format(idx + 1))

            autoencoder.train_model(self.nb_epochs, self.data_loaders[idx])
            print()

    def outliers_scoring(self, datas):
        '''
        Outliers scoring for all the autoencoders and all the datas
        '''
        scores = torch.empty(self.nb_autoencoder, datas.shape[0])
        datas_tensor = torch.tensor(datas)

        for idx, ae_model in enumerate(self.autoencoders):
            scores[idx, :] = ae_model.outlier_score_vector(datas_tensor)
        return torch.median(scores, dim=0)[0]