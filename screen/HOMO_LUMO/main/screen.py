# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:51:28 2023

@author: BM109X32G-10GPU-02
"""

import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score

import preprocess as pp


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        if task == 'classification':
            self.W_property = nn.Linear(dim, 2)
        if task == 'regression':
            self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

    def mlp(self, vectors):
        """Classifier or regressor based on multilayer perceptron."""
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs

    def forward_classifier(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_labels = torch.cat(data_batch[-1])

        if train:
            molecular_vectors = self.gnn(inputs)
            predicted_scores = self.mlp(molecular_vectors)
            loss = F.cross_entropy(predicted_scores, correct_labels)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_scores = self.mlp(molecular_vectors)
            predicted_scores = predicted_scores.to('cpu').data.numpy()
            predicted_scores = [s[1] for s in predicted_scores]
            correct_labels = correct_labels.to('cpu').data.numpy()
            return predicted_scores, correct_labels

    def forward_regressor(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return predicted_values, correct_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            if task == 'classification':
                loss = self.model.forward_classifier(data_batch, train=True)
            if task == 'regression':
                loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_classifier(self, dataset):
        N = len(dataset)
        P, C = [], []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_scores, correct_labels = self.model.forward_classifier(
                                               data_batch, train=False)
            P.append(predicted_scores)
            C.append(correct_labels)
        AUC = roc_auc_score(np.concatenate(C), np.concatenate(P))
        return AUC

    def test_regressor(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        Ts, Ys =   [], []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            ( predicted_values,correct_values) = self.model.forward_regressor(
                                               data_batch, train=False)
          
            Ts.append(correct_values)
            Ys.append(predicted_values)
            
            SAE += sum(np.abs(predicted_values-correct_values))
       
        T, Y = map(str, np.concatenate(Ts)), map(str, np.concatenate(Ys))
        #MSE = SE_sum / N
        predictions = '\n'.join(['\t'.join(x) for x in zip( T, Y)])
        MAEs = SAE / N  # mean absolute error.
        return MAEs,predictions
    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Correct\tPredict\n')
            f.write(predictions + '\n')
if __name__ == "__main__":

    task='regression'  # target is a real value (e.g., energy eV).
    dataset='pre'
    model_name='lumo'
    pretrain = True
    radius=1
    dim=56
    layer_hidden=6
    layer_output=6
    
    batch_train=32
    batch_test=32
    lr=3e-4
    lr_decay=0.99
    decay_interval=10
    iteration=600
    
    (radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, decay_interval,
     iteration) = map(int, [radius, dim, layer_hidden, layer_output,
                            batch_train, batch_test,
                            decay_interval, iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    (dataset, N_fingerprints) = pp.create_predict(task, dataset, radius, device)
    print('-'*100)

    print('The preprocess has finished!')

    print('-'*100)

    print('Creating a model.')
    torch.manual_seed(1234)
    model = MolecularGraphNeuralNetwork(
            N_fingerprints, dim, layer_hidden, layer_output).to(device)
    if pretrain == True:
        model.load_state_dict(torch.load(model_name + ".h5"))
    
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_result = '../output/result-prelu-' + '.txt'
    if task == 'classification':
        result = 'Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test'
    if task == 'regression':
        result = 'Epoch\tTime(sec)\tLoss_train\tMAE_dev\tMAE_test'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    np.random.seed(1234)

    prediction_test,predictions_test = tester.test_regressor(dataset)
    
    file_test_result  ='../output/'+ 'lumo_pre_prediction'+ '.txt'
    
    tester.save_predictions(predictions_test , file_test_result )
    from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
    import pandas as pd 
    res  = pd.read_table(file_test_result)
    # smi = pd.read_csv()
    # smiles = smi[:,0]
    # a = []
    # for ind, i in enumurate res['Predict']:
    #     if i < -5.5:
            
            
    #         a.append(i)
    # print(len(a))
    # file = pd.DataFrame(a)
    r2 = r2_score(res ['Correct'], res ['Predict'])
    mae = mean_absolute_error(res ['Correct'], res ['Predict'])
