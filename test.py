# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:40:48 2022

@author: BM109X32G-10GPU-02
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.constraints import max_norm
import pandas as pd
import numpy as np

from dataset import Graph_Regression_test 
from sklearn.metrics import r2_score,roc_auc_score
from tqdm import tqdm
import os
from model import  PredictModel,BertModel

def main(seed):
    # tasks = ['caco2', 'logD', 'logS', 'PPB', 'tox']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    keras.backend.clear_session()
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights','addH':True}
    medium3 = {'name': 'Medium', 'num_layers': 8, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights3','addH':True}
    medium2 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights2',
               'addH': True}
    large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights','addH':True}
    medium_without_H = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_without_H','addH':False}
    medium_without_pretrain = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256,'path': 'medium_without_pretraining_weights','addH':True}

    arch = medium3 ## small 3 4 128   medium: 6 6  256     large:  12 8 516

    pretraining = True
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

    #tf.random.set_seed(seed=seed)
    graph_dataset = Graph_Regression_test('data/reg/{}.csv', addH=addH)
        
    predct_dataset= graph_dataset.get_data()
    
    #value_range = graph_dataset.value_range()

    x, adjoin_matrix, y = next(iter(predct_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]

    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.2)
    preds = model(x, mask=None, adjoin_matrix=adjoin_matrix, training=False)
    model.load_weights('regression_weights/{}.h5'.format('data'))

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

    steps_per_epoch = len(predct_dataset)
    value_range = 1



    y_true = []
    y_preds = []

    for x, adjoin_matrix, y in tqdm(predct_dataset):
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=None, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
   
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)

    test_r2 = r2_score(y_true, y_preds)
    test_mse = keras.metrics.mse(y_true.reshape(-1), y_preds.reshape(-1)).numpy() * (value_range**2)
    print('test r2:{:.4f}'.format(test_r2), 'test mse:{:.4f}'.format(test_mse))
    prediction_test=np.vstack((y_true,y_preds))

    return test_r2, prediction_test

if __name__ == "__main__":
    result =[]
    r2_list = []
    for seed in [15]:
        print(seed)
        r2,prediction_test= main(seed)
        result.append(prediction_test)
        r2_list.append(r2)
    print(r2_list)
    from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
    from scipy.stats import pearsonr
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    prediction_t=(prediction_test[1,:]).T
    plt.plot(prediction_test[0,:], prediction_test [1,:], '.', color = 'blue')
    plt.plot([0,20], [0,20], color ='red')
    plt.ylabel('Predicted PCE')
    plt.xlabel('Computational PCE') 
    re = []
    for i in range (len(result)):
        data = result[i]
        mae = mean_absolute_error(data[0],data[1])
        r2=r2_score(data[0],data[1])
        mse = mean_squared_error(data[0],data[1])
        
        r=pearsonr(data[0],data[1])[0]
        res=np.vstack((mae,r2,mse,r))
        re.append(res)
 