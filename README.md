# <u>DeepAcceptor</u>

### **Computational design and screening of acceptor materials for organic solar cells**

![workflow](workflow.jpg)

## <u>Motivation</u>

It is a time-consuming and costly process to develop affordable and high-performance organic photovoltaic materials. Developing reliable computational methods to predict the power conversion efficiency (PCE) is crucial to triage unpromising molecules in large-scale databases and accelerate the material discovery process. In this study, a deep learning-based framework (DeepAcceptor) has been built to design and discover high-efficient small molecule acceptor materials. Specifically, an experimental dataset was constructed by collecting data from publications. Then, a BERT-based model was customized to predict PCEs by taking fully advantages of the atom, bond, connection information in molecular structures of acceptors, and this customized architecture is termed as abcBERT. The computation molecules and experimental molecules were used to pre-train and fine-tune the model, respectively. The molecular graph was used as the input and the computation molecules and experimental molecules were used to pretrain and finetune the model, respectively. In sum, DeepAcceptor is a promising method to predict the PCE and speed up the discovery of high-performance acceptor materials.

------



### <u>Depends</u>

We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).

[python3.8](https://www.python.org/download) 		

[Anaconda](https://www.anaconda.com/)

**By using the *environment.yml* file, it will install all the required packages.**

```
git clone https://github.com/jinysun/deepacceptor.git
cd DeepAcceptor
conda env create -f environment.yml
conda activate deepacceptor
```

------

## <u>Usage</u>

| The code of abcBERT is as follows.                           |
| ------------------------------------------------------------ |
| -- [pretrain](https://github.com/JinYSun/DeepAcceptor/blob/master/abcBERT/pretrain.py): contains the codes for masked atom prediction pre-training task. |
| -- [regression](https://github.com/JinYSun/DeepAcceptor/blob/master/abcBERT/regression.py): contain the code for fune-tuning on specified tasks |
| -- [dataset](https://github.com/JinYSun/DeepAcceptor/blob/master/abcBERT/dataset.py): contain the code to building dataset for pre-traing and fine-tuning |
| -- [utils](https://github.com/JinYSun/DeepAcceptor/blob/master/abcBERT/utils.py): contain the code to convert molecules to graphs |
| --[predict](https://github.com/JinYSun/DeepAcceptor/blob/master/abcBERT/predict.py): contain the code for predict the properties |

------



## <u>Data pre-processing</u>

abcBERT is a model for predicting PCE based on molecular graph,  so we need to convert SMILES strings to Graph. The related method is  shown in [`deepacceptor/utils.py`](https://github.com/JinYSun/DeepAcceptor/blob/master/abcBERT/utils.py)

First, put the test file in the file data/reg/.

Then, run the utils.py as follows.

    import pandas as pd 
    f = pd.read_csv (r"data/reg/test.csv")
    re = []
    pce = f['PCE']
    for ind,smile in enumerate ( f.iloc[:,0]):
        
        atom,adj = mol_to_geognn_graph_data_MMFF3d(smile)
        np.save('data/reg/test/adj'+str(ind)+'.npy',np.array(adj))
        re.append([atom,'data/reg/test/adj'+str(ind)+'.npy',pce[ind] ])
    r = pd.DataFrame(re)
    r.to_csv('data/reg/test/test.csv')
    print('Done!')

------



## <u>Model training</u>

1. #### Pretrain the model

   ```
   import pretrain
   pretrain.main()
   ```

2. #### Finetune the model

       import regression
       from regression import *
       result =[]
       r2_list = []
       seed = 12
       r2,prediction_val,prediction_test= main(seed)

------



## <u>Predicting PCE</u>

The PCE prediction is obtained by feeding the the processed molecules into the already trained abcBERT model with [predict.py](https://github.com/JinYSun/DeepAcceptor/blob/master/abcBERT/predict.py) 

    import predict
    from predict import *
    result =[]
    r2_list = []
    seed = 12
    r2,prediction_val= main(seed)

**The example codes for usage is included in the [test.ipynb](https://github.com/JinYSun/DeepAcceptor/blob/master/abcBERT/test.ipynb)**

------



## <u>Designing and Screening</u>

![screen](screen/screen.jpg)

### Molecular generation

[BRICS](https://github.com/JinYSun/DeepAcceptor/blob/master/screen/general.py)+[VAE](https://github.com/JinYSun/DeepAcceptor/tree/master/screen/vae): A fragments-based molecule design framework was built by using the  breaking of retrosynthetically interesting chemical substructures  (BRICS) algorithm and variational autoencoder (VAE) to obtain a database with specific potential molecular properties. 

### Basic properties

[Basic properties](https://github.com/JinYSun/DeepAcceptor/blob/master/screen/properties.py): The Gen database was screened with some basic properties such as molecular size, log*P*, the number of H-bond acceptors and donors, number of rotatable bonds. These properties were calculated by using RDKit. 

### HOMO & LUMO matching

[GNN](https://github.com/JinYSun/DeepAcceptor/tree/master/screen/HOMO_LUMO) was trained on a NFA dataset including HOMO and LUMO  computing by DFT. The dataset including 51000 NFAs was splited randomly  with a ratio of 8:1:1. The MAE and R2 of the predicted HOMO are 0.052  and 0.972. 

## SAscore

[SAscore](https://github.com/JinYSun/DeepAcceptor/tree/master/screen/SAscore)  was used to synthetic accessibility and complexity. 

### **Molecular polarities and charge distribution**

[Properties](https://github.com/JinYSun/DeepAcceptor/blob/master/screen/properties.py) related to molecular polarity and charge distribution were calculated by RDKit. 

------



## <u>Discussion</u> 

The [Discussion](https://github.com/JinYSun/Deepacceptor/tree/main/discussion) folder contains the scripts for evaluating the PCE prediction performance.  We compared sevaral common methods widely used in molecular property prediction, such as [MolCLR](https://github.com/JinYSun/DeepAcceptor/blob/main/discussion/MolCLR.py) [GNN](https://github.com/JinYSun/DeepAcceptor/blob/main/discussion/GNN.py),[RF](https://github.com/JinYSun/DeepAcceptor/blob/main/discussion/RF.py), [ANN](https://github.com/JinYSun/Deepacceptor/blob/main/discussion/ANN.py),[QDF](https://github.com/JinYSun/DeepAcceptor/blob/main/discussion/QDF.py).

## <u>Contact</u>

Jinyu Sun. E-mail: [jinyusun@csu.edu.cn](mailto:jinyusun@csu.edu.cn)
