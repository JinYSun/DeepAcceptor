# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:05:03 2023

@author: BM109X32G-10GPU-02
"""

import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.constraints import max_norm
import pandas as pd
import numpy as np

from dataset import Graph_Regression_test #Graph_Regression_Dataset,
from sklearn.metrics import r2_score,roc_auc_score

import os
from model import  PredictModel,BertModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



def main(seed):
    # tasks = ['caco2', 'logD', 'logS', 'PPB', 'tox']
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    keras.backend.clear_session()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
    print(task)
    seed = seed

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size =60
    dropout_rate = 0.1

    tf.random.set_seed(seed=seed)
    graph_dataset = Graph_Regression_test('data/reg/{}.csv', addH=addH)
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
    model.load_weights('regression_weights/{}.h5'.format('data'))

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

    test_r2 = r2_score(y_true, y_preds)
    test_mse = keras.metrics.mse(y_true.reshape(-1), y_preds.reshape(-1)).numpy() * (value_range**2)
    print('test r2:{:.4f}'.format(test_r2), 'test mse:{:.4f}'.format(test_mse))
    prediction_test=np.vstack((y_true,y_preds))

    return test_r2, prediction_test

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = [np.arange(3)]
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(3))
    plt.xticks(xlocations, rotation=90)
    plt.yticks(xlocations)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(3)) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
if __name__ == "__main__":
    result =[]
    r2_list = []
    for seed in [24]:
        print(seed)
        r2 ,prediction_val= main(seed)
        result.append(prediction_val)
        r2_list.append(r2)
    print(r2_list)
    from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
    from scipy.stats import pearsonr
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    re = []
 