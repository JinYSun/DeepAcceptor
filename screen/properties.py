"""
This code is used to extract molecular descriptors using rdkit or CDKDesc from molecule SMILES.

Library versions:

Python 2.7.13
rdkit.__version__ = '2017.09.1'
"""

__author__ = "Robbin Bouwmeester"
__copyright__ = "Copyright 2017"
__credits__ = ["Robbin Bouwmeester","Prof. Lennart Martens","Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__version__ = "1.0"
__maintainer__ = "Robbin Bouwmeester"
__email__ = "Robbin.bouwmeester@ugent.be"

# Native imports
from subprocess import Popen
from subprocess import PIPE
from os import remove
import pandas as pd 
# rdkit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

def rdkit_descriptors(mol):
	"""
	Get the molecular descriptors by iterating over all possible descriptors.

    Parameters
    ----------
    mol : object
        An rdkit molecules object.
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	ret_dict = {}
	
	# Iterate over all descriptors, get their functions (func) and apply to the molecule object
	for name,func in Descriptors.descList:
		ret_dict[name] = func(mol)
	return(ret_dict)

def cdk_descriptors(mol,temp_f_smiles_name="tempsmiles.smi",temp_f_cdk_name="tempcdk.txt"):
	"""
	Get the molecular descriptors in cdk

    Parameters
    ----------
    mol : object
        An rdkit molecules object.
	temp_f_smiles_name : str
		Where to temporarily store the smiles
	temp_f_cdk_name : str
		Where to temporarily store the cdk file
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	ret_dict = {}

	# Get the SMILES from the rdkit object
	smiles = Chem.MolToSmiles(mol,1)
		
	# Write SMILES to a temp file
	temp_f_smiles = open(temp_f_smiles_name,"w")
	temp_f_smiles.write("%s temp" % smiles)
	temp_f_smiles.close()
	
	# Get the features, descriptors indicate the main class of descriptors
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="topological"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="geometric"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="constitutional"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="electronic"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="hybrid"))
	
	# Remove the temporary files
	remove(temp_f_smiles_name)
	remove(temp_f_cdk_name)

	return(ret_dict)


def call_cdk(infile="",outfile="",descriptors=""):
	"""
	Make a system call to cdk to get descriptors

    Parameters
    ----------
    infile : str
        String indicating the filename for the input SMILES
	outfile : str
		String indicating the filename for the output values
	descriptors : str
        Main class of molecular descriptors to retrieve
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	cmd = "java -jar CDKDescUI-1.4.6.jar -b %s -a -t %s -o %s" % (infile,descriptors,outfile)
	p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
	out = p.communicate()
	return(parse_cdk_file(outfile))

def parse_cdk_file(file):
	"""
	Parse the cdk output files

    Parameters
    ----------
    file : str
        String indicating the filename for the output of cdk
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	cdk_file = open(file).readlines()
	cols = cdk_file[0].strip().split()[1:]
	feats = cdk_file[1].strip().split()[1:]
	return(dict(zip(cols, feats)))

def getf(mol,progs=["rdkit"]):
	"""
	Main function that makes the call to rdkit or cdk

    Parameters
    ----------
    mol : object
        An rdkit molecules object.0
	progs : list
		List containing the tools to use for calculating molecular descriptors. Use "rdkit" and or "cdk"
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	ret_dict = {}
	if "rdkit" in progs: ret_dict["rdkit"] = rdkit_descriptors(mol)
	if "cdk" in progs: ret_dict["cdk"] = cdk_descriptors(mol)
	return(ret_dict)


if __name__ == "__main__":
    from tqdm import tqdm
    file = pd.read_csv(r"J:\screenacc\high.csv",header=None)
    fi = file.iloc[1:,1]
    f = []
    res = pd.DataFrame()
    for ind,smi in enumerate( tqdm(fi)):
         
          mol = Chem.MolFromSmiles(smi)
          try:
              a = getf(mol)
              b=a['rdkit']
              f.append(b)
          except:
              b=[]
          # a=pd.DataFrame(a)
          # res = pd.concat([a,res],axis=1)
           
          f.append(b)
          if b['MolLogP']<80 and b['MolLogP']>12.72: #and b['MolLogP']>8.76 
              if b['MolWt']<4000 and b['MolWt']>800: #and b['MolWt']>485
                  if b['NOCount']<25 :
                      if    b['NumHAcceptors']<32  and b['NumHAcceptors']>6:
                        if  b['NumHDonors']<2:
                         # if    b['NumHeteroatoms']<29.4 and b['NumHeteroatoms']>7.6: #and b['NumHeteroatoms']>5.7
                            #if      b['NumRadicalElectrons']<7:
                              if         b['NumRotatableBonds']>15 : #b['NumRotatableBonds']<96.6 and
                    
                                  if            b['RingCount']<40 and b['RingCount']>6:
                              #      if              b['NumAromaticHeterocycles']<10.5  :
                                      if                b['NumAromaticRings']<22 :
                                                      
                                                  if 'P' not in smi:
                                                              print(ind)
                                                              # file.iloc[ind,:]
                                                              f.append(b)
                                                         
    res= pd.DataFrame(f)
    #res = pd.concat([file,f1],axis=1)
   # res.to_csv(r"J:\screenacc\4.csv")
    # corrdf = res.corr()
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # corrdf
    # plt.subplots(figsize=(9, 9)) # 设置画面大小
    # sns.heatmap(corrdf,  cmap="Blues")
   
    # plt.show()

