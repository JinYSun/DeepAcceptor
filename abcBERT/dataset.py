import pandas as pd
import numpy as np
from utils import mol_to_geognn_graph_data_MMFF3d as smiles2adjoin
import tensorflow as tf

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br':  9,
         'B': 10,'I': 11,'Si':12,'Se':13,'<unk>':14,'<mask>':15,'<global>':16}

num2str =  {i:j for j,i in str2num.items()}

    
class Graph_Bert_Dataset(object):
    def __init__(self,path,smiles_field=['0'], adj=['1'],addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\n\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.adj = adj
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH

    def get_data(self):

        data = self.df
        
        train_idx = []
        idx = data.sample(frac=0.9).index
      
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]
       
        self.dataset1 = tf.data.Dataset.from_tensor_slices((data1[self.smiles_field],data1[self.adj]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]))).prefetch(50)
         
        self.dataset2 = tf.data.Dataset.from_tensor_slices((data2[self.smiles_field],data2[self.adj]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)
        return self.dataset1, self.dataset2

    def numerical_smiles(self, atom, adj):
        #smiles = smiles.numpy().decode()
        atom = np.array(atom)
        atom = atom[0].decode()
        
        atom = atom.replace('\n','')
        
        atom = atom.replace('[',' ')
        atom = atom.replace(']',' ')
        atom = atom.split("'")
        
        
        atoms_list = []
        for i in atom:
            if i not in [' ']:
                atoms_list.append(i)
       
        adj = np.array(adj)[0].decode()

        adjoin_matrix =np.load( adj )
        
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        temp[np.where(temp  == 0)]=-1e9
        
        
        adjoin_matrix = temp
        #adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list).astype('int64')
        weight = weight.astype('float32')
        return x, adjoin_matrix, y, weight

    def tf_numerical_smiles(self, atom,adj):
        #print(data)
        # x,adjoin_matrix,y,weight = tf.py_function(self.balanced_numerical_smiles,
        #                                           [data], [tf.int64, tf.float32 ,tf.int64,tf.float32])
        x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, (atom, adj),
                                                     [tf.int64, tf.float32, tf.int64, tf.float32])

        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        weight.set_shape([None])
        return x, adjoin_matrix, y, weight

class Graph_Regression_Dataset_test(object):
    def __init__(self,path,smiles_field='SMILES',label_field='PCE',normalize=False,max_len=1000,addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path.format('test'),sep='\t')
        else:
            self.df = pd.read_csv(path.format('test'))

        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len()<=max_len]
        self.addH =  addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field]-self.min)/(self.max-self.min)-0.5
            self.value_range = self.max-self.min


    def get_data(self):
        train_data = self.df
        self.dataset1 = tf.data.Dataset.from_tensor_slices((train_data[self.smiles_field], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1])))
        return self.dataset1

    def numerical_smiles(self, smiles,label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoins(smiles)
        atoms_list = list(atoms_list)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix      
        temp[np.where(temp  == 0)]=-1e9
        adjoin_matrix = temp 
        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('float32')
        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix , y
    
class predict_smiles(object):
    def __init__(self,smiles ,normalize=False,max_len=1000,addH=True):
     
        self.smiles_field = smiles
        
        self.label_field = float(0)
        self.vocab = str2num
        self.devocab = num2str
        #self.df = self.df[self.df[smiles_field].str.len()<=max_len]
        self.addH =  addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field]-self.min)/(self.max-self.min)-0.5
            self.value_range = self.max-self.min
    def numerical_smiles(self, atoms_list,adj,label):
                
        atom = np.array(atoms_list)            
        atoms_list = []
        for i in atom:
            if i not in [' ']:
                atoms_list.append(str(i,encoding='utf-8'))
        label = np.array(label)
       
        adj = np.array(adj)

        adjoin_matrix =adj  
       
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        #temp = np.ones((len(nums_list),len(nums_list)))
        #temp[1:, 1:] = adjoin_matrix
        #adjoin_matrix = (1-temp)*(-1e9)

        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix      
        temp[np.where(temp  == 0)]=-1e9
        
  
        adjoin_matrix = temp 
        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('float32')
        return x, adjoin_matrix,y

    def get_data(self):
        atom, adj = smiles2adjoin( self.smiles_field)        
        atom = np.array(atom)
        atoms_list = []
        for i in atom:
            if i not in [' ']:
                atoms_list.append(i)       
        adj = np.array(adj)
        adjoin_matrix = adj   
        self.dataset1 = tf.data.Dataset.from_tensors((atoms_list, adjoin_matrix,  self.label_field))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(1, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1])))   

        return self.dataset1

    def tf_numerical_smiles(self, atoms_list,adj,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, (atoms_list,adj,label), [tf.int64, tf.float32 ,tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix , y 
    
class Graph_Regression_test(object):
    def __init__(self,path, filename, smiles_field=['0'],adj = ['1'], label_field=['2'],normalize=False,max_len=1000,addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
           # self.df = pd.read_csv(path.format('train3'),sep='\t')
            #self.dt = pd.read_csv(path.format('test3'),sep='\t')
            self.dv = pd.read_csv(path.format('val3'),sep='\t')
        else:
            #self.df = pd.read_csv(path.format('train/train'))
           #self.dt = pd.read_csv(path.format('test/test'))
            self.dv = pd.read_csv(path.format(filename))
        self.smiles_field = smiles_field
        self.adj = adj
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        #self.df = self.df[self.df[smiles_field].str.len()<=max_len]
        self.addH =  addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field]-self.min)/(self.max-self.min)-0.5
            self.value_range = self.max-self.min


    def get_data(self):
        train_data = self.dv

       
        #idx = train_data.sample(frac=0.9).index
        # train_idx = []
        # #idx = train_data.sample(frac=0.9).index
      
        # train_idx.extend(idx)
        # data1 = train_data[train_data.index.isin(train_idx)]
        # data2 = train_data[~train_data.index.isin(train_idx)]
        self.dataset1 = tf.data.Dataset.from_tensor_slices((train_data[self.smiles_field],train_data[self.adj], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1]))).prefetch(100)
        return self.dataset1

    def numerical_smiles(self, atom,adj,label):
        atom = np.array(atom)
        atom = atom[0].decode()
        
        atom = atom.replace('\n','')
        
        atom = atom.replace('[',' ')
        atom = atom.replace(']',' ')
        atom = atom.split("'")
        
        
        atoms_list = []
        for i in atom:
            if i not in [' ']:
                try:
                    atoms_list.append(str(i,encoding='utf-8'))
                except:
                    atoms_list.append(str(i))
        label = np.array(label)[0]
       
        adj = np.array(adj)[0].decode()

        adjoin_matrix =np.load( adj )
        
        
       
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        #temp = np.ones((len(nums_list),len(nums_list)))
        #temp[1:, 1:] = adjoin_matrix
        #adjoin_matrix = (1-temp)*(-1e9)

        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix      
        temp[np.where(temp  == 0)]=-1e9
        
  
        adjoin_matrix = temp 
        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('float32')
        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,adj,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, (smiles,adj,label), [tf.int64, tf.float32 ,tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix , y    
    
class Graph_Regression(object):
    def __init__(self,path,smiles_field=['0'],adj = ['1'], label_field=['2'],normalize=False,max_len=1000,addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path.format('train3'),sep='\t')
            self.dt = pd.read_csv(path.format('test3'),sep='\t')
            #self.dv = pd.read_csv(path.format('val3'),sep='\t')
        else:
            self.df = pd.read_csv(path.format('train/train'))
            self.dt = pd.read_csv(path.format('test/test'))
            #self.dv = pd.read_csv(path.format('val/val'))
        self.smiles_field = smiles_field
        self.adj = adj
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        #self.df = self.df[self.df[smiles_field].str.len()<=max_len]
        self.addH =  addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field]-self.min)/(self.max-self.min)-0.5
            self.value_range = self.max-self.min


    def get_data(self):
        train_data = self.df

        test_data = self.dt
        data2=test_data
        #idx = train_data.sample(frac=0.9).index
        # train_idx = []
        # #idx = train_data.sample(frac=0.9).index
      
        # train_idx.extend(idx)
        # data1 = train_data[train_data.index.isin(train_idx)]
        # data2 = train_data[~train_data.index.isin(train_idx)]
        self.dataset1 = tf.data.Dataset.from_tensor_slices((train_data[self.smiles_field],train_data[self.adj], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1]))).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.adj],test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([1]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((data2[self.smiles_field],test_data[self.adj], data2[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        return self.dataset1,self.dataset2,self.dataset3

    def numerical_smiles(self, atom,adj,label):
        atom = np.array(atom)
        atom = atom[0].decode()
        
        atom = atom.replace('\n','')
        
        atom = atom.replace('[',' ')
        atom = atom.replace(']',' ')
        atom = atom.split("'")
        
        
        atoms_list = []
        for i in atom:
            if i not in [' ']:
                atoms_list.append(i)
        label = np.array(label)[0]
       
        adj = np.array(adj)[0].decode()

        adjoin_matrix =np.load( adj )
        
        
       
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        #temp = np.ones((len(nums_list),len(nums_list)))
        #temp[1:, 1:] = adjoin_matrix
        #adjoin_matrix = (1-temp)*(-1e9)

        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix      
        temp[np.where(temp  == 0)]=-1e9
        
  
        adjoin_matrix = temp 
        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('float32')
        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,adj,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, (smiles,adj,label), [tf.int64, tf.float32 ,tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix , y
    
class Inference_Dataset(object):
    def __init__(self,path,smiles_field='Smiles',addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH

    def get_data(self):

        data = self.df
        
        train_idx = []
        idx = data.sample(frac=0.9).index
      
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]
        print(len(data1))
        self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(1, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]))).prefetch(50)
        print(self.dataset1)
        self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(1, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)
        return self.dataset1, self.dataset2

    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoins(smiles,explicit_hydrogens=self.addH)
        print(atoms_list)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        temp[np.where(temp  == 0)]=-1e9
        adjoin_matrix = temp
        choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1
        y = np.array(nums_list).astype('int64')
       
        x = np.array(nums_list).astype('int64')
        
        return x, adjoin_matrix,  [smiles],atoms_list

    def tf_numerical_smiles(self, data):
        # x,adjoin_matrix,y,weight = tf.py_function(self.balanced_numerical_smiles,
        #                                           [data], [tf.int64, tf.float32 ,tf.int64,tf.float32])
        x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data],
                                                     [tf.int64, tf.float32, tf.int64, tf.float32])
        smiles.set_shape([1])
        atom_list.set_shape([None])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        weight.set_shape([None])
        return x, adjoin_matrix,smiles,atom_list

class Inference_Dataset(object):
    def __init__(self,sml_list,max_len=1000,addH=True):
        self.vocab = str2num
        self.devocab = num2str
        self.sml_list = [i for i in sml_list if len(i)<max_len]
        self.addH =  addH

    def get_data(self):

        self.dataset = tf.data.Dataset.from_tensor_slices((self.sml_list,))
        self.dataset = self.dataset.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1]),tf.TensorShape([None]))).cache().prefetch(20)

        return self.dataset

    def numerical_smiles(self, smiles):
        smiles_origin = smiles
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoins(smiles)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)
        x = np.array(nums_list).astype('int64')
        return x, adjoin_matrix,[smiles], atoms_list

    def tf_numerical_smiles(self, smiles):
        x,adjoin_matrix,smiles,atom_list = tf.py_function(self.numerical_smiles, [smiles], [tf.int64, tf.float32,tf.string, tf.string])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        smiles.set_shape([1])
        atom_list.set_shape([None])
        return x, adjoin_matrix,smiles,atom_list
