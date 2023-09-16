# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:32:59 2023

@author: BM109X32G-10GPU-02
"""


import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.constraints import max_norm
import pandas as pd
import numpy as np
import sys
from dataset import predict_smiles
from sklearn.metrics import r2_score,roc_auc_score

import os
from model import  PredictModel,BertModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



def main(smiles):
    # tasks = ['caco2', 'logD', 'logS', 'PPB', 'tox']
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    keras.backend.clear_session()
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights','addH':True}
    medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights2','addH':True}
    medium3 = {'name': 'Medium', 'num_layers': 8, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights2',
               'addH': True}
    large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights','addH':True}
    medium_without_H = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_without_H','addH':False}
    medium_without_pretrain = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256,'path': 'medium_without_pretraining_weights','addH':True}

    arch = medium3## small 3 4 128   medium: 6 6  256     large:  12 8 516

    pretraining = False
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 80
    task = 'data'
    seed = 14
    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']
    dff = d_model * 2
    vocab_size =60
    dropout_rate = 0.1

    tf.random.set_seed(seed=seed)
    graph_dataset = predict_smiles(smiles, addH=addH)
    # graph_dataset = Graph_Regression_Dataset('data/reg/{}.csv', smiles_field='SMILES',
    #                                                         label_field='PCE',addH=addH)        
    test_dataset = graph_dataset.get_data()
    
    #value_range = graph_dataset.value_range()

    x, adjoin_matrix, y = next(iter(test_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]

    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.2)
    preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
    model.load_weights('regression_weights/{}.h5'.format('15data'))

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)

        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')

   
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, total_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.total_step = total_steps
            self.warmup_steps = total_steps*0.10

        def __call__(self, step):
            arg1 = step/self.warmup_steps
            arg2 = 1-(step-self.warmup_steps)/(self.total_step-self.warmup_steps)

            return 10e-5* tf.math.minimum(arg1, arg2)

    steps_per_epoch = len(test_dataset)
    value_range = 1



    y_true = []
    y_preds = []

    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
   
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
 
    return y_preds

if __name__ == "__main__":
    y_preds = main('CCCCCCCCC1=CC=C(C2(C3=CC=C(CCCCCCCC)C=C3)C3=CC4=C(C=C3C3=C2C2=C(C=C(C5=CC=C(/C=C6/C(=O)C7=C(C=CC=C7)C6=C(C#N)C#N)C6=NSN=C56)S2)S3)C(C2=CC=C(CCCCCCCC)C=C2)(C2=CC=C(CCCCCCCC)C=C2)C2=C4SC3=C2SC(C2=CC=C(/C=C4\C(=O)C5=C(C=CC=C5)C4=C(C#N)C#N)C4=NSN=C24)=C3)C=C1')
 