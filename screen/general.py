# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:15:26 2021

@author: BM109X32G-10GPU-02
"""
from pandas import DataFrame
import numpy as np
import itertools
import pandas as pd
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, BRICS, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
data=pd.read_csv(r"high.csv",encoding='ISO-8859-1')
data = (data.iloc[:,0])

mols_list=np.array([Chem.MolFromSmiles(mol) for mol in data if mol is not None])

fragment_set = set()
for mol in mols_list:
    try:
        fragment = BRICS.BRICSDecompose(mol, allNodes=None, minFragmentSize=1, onlyUseReactions=None, silent=True, keepNonLeafNodes=True, singlePass=False, returnMols=False)#chai
        fragment_set = fragment_set | fragment
    except:
        continue
fragment_smiles = list(fragment_set)
 
frag_1dummy_s = [smiles for smiles in fragment_smiles  if smiles.count('*')==1 ]
frag_2dummy_s = [smiles for smiles in fragment_smiles if smiles.count('*')==2]
frag_1dummy1 = np.array([Chem.MolFromSmiles(smiles) for smiles in frag_1dummy_s])
frag_2dummy1 = np.array([Chem.MolFromSmiles(smiles) for smiles in frag_2dummy_s])
 
descriptor_calc = MoleculeDescriptors.MolecularDescriptorCalculator(['MolWt'])
MolWt_list1 = np.array([descriptor_calc.CalcDescriptors(mol)[0] for mol in frag_1dummy1])
frag_1dummy = frag_1dummy1[np.where((MolWt_list1<=200 ) & ( MolWt_list1>=50) )]
frag_2dummy = frag_1dummy1[np.where((MolWt_list1>=300) & (MolWt_list1<=800 ))]

descriptor_calc = MoleculeDescriptors.MolecularDescriptorCalculator(['MolWt'])
MolWt_list2 = np.array([descriptor_calc.CalcDescriptors(mol)[0] for mol in frag_2dummy1])
frag_3dummy = frag_2dummy1[np.where((MolWt_list2<=200 ) & ( MolWt_list2>=50) )]
frag_4dummy = frag_2dummy1[np.where((MolWt_list2>=500) & (MolWt_list2<=800 ))]

print('number of fragment:',len(frag_1dummy))
#>>> number of fragment: 333
print('number of fragment:',len(frag_2dummy))
img1 = Draw.MolsToGridImage(frag_2dummy[:10], molsPerRow=3,legends=frag_2dummy_s[:10])
img1
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Recap

sm =data[1]
#sm = celecoxib.canonical_smiles
celecoxib = Chem.MolFromSmiles(sm)
decomp = Chem.Recap.RecapDecompose(celecoxib)
print(type(decomp))
first_gen = [node.mol for node in decomp.children.values()]
print(first_gen)
node_type = [node for node in decomp.children.values()]
print(node_type)
Draw.MolsToGridImage(first_gen, molsPerRow=4, legends=[Chem.MolToSmiles(m) for m in first_gen])

def structure_generator(main_mol, fragment1,  r_position=0):
 
    """
    parameters
    ----------
    main_mol : RDkit Mol object
    fragment1 : RDkit Mol object
    fragment2 : RDkit Mol object
    r_position : int (1 or 2) 
        fragment1
    Returns
    ----------
    generated_molecule : RDkit Mol object
    """
    bond_list = [Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.QUADRUPLE, Chem.rdchem.BondType.QUINTUPLE,
             Chem.rdchem.BondType.HEXTUPLE, Chem.rdchem.BondType.ONEANDAHALF, Chem.rdchem.BondType.TWOANDAHALF,
             Chem.rdchem.BondType.THREEANDAHALF, Chem.rdchem.BondType.FOURANDAHALF, Chem.rdchem.BondType.FIVEANDAHALF,
             Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.IONIC, Chem.rdchem.BondType.HYDROGEN,
             Chem.rdchem.BondType.THREECENTER, Chem.rdchem.BondType.DATIVEONE, Chem.rdchem.BondType.DATIVE,
             Chem.rdchem.BondType.DATIVEL, Chem.rdchem.BondType.DATIVER, Chem.rdchem.BondType.OTHER,
             Chem.rdchem.BondType.ZERO]
 
    # make adjacency matrix and get atoms for main molecule
    level1_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(fragment1)
 
    for bond in fragment1.GetBonds():
        level1_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(bond.GetBondType())
        level1_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(bond.GetBondType())
    level1_atoms = []
    for atom in fragment1.GetAtoms():
        level1_atoms.append(atom.GetSymbol())
 
    r_index_in_level1_molecule_old = [index for index, atom in enumerate(level1_atoms) if atom == '*']
 
    for index, r_index in enumerate(r_index_in_level1_molecule_old):
        modified_index = r_index - index
        atom = level1_atoms.pop(modified_index)
        level1_atoms.append(atom)
        tmp = level1_adjacency_matrix[:, modified_index:modified_index + 1].copy()
        level1_adjacency_matrix = np.delete(level1_adjacency_matrix, modified_index, 1)
        level1_adjacency_matrix = np.c_[level1_adjacency_matrix, tmp]
        tmp = level1_adjacency_matrix[modified_index:modified_index + 1, :].copy()
        level1_adjacency_matrix = np.delete(level1_adjacency_matrix, modified_index, 0)
        level1_adjacency_matrix = np.r_[level1_adjacency_matrix, tmp]
 
    r_index_in_level1_molecule_new = [index for index, atom in enumerate(level1_atoms) if atom == '*']
 
    r_bonded_atom_index_in_level1_molecule = []
    for number in r_index_in_level1_molecule_new:
        r_bonded_atom_index_in_level1_molecule.append(np.where(level1_adjacency_matrix[number, :] != 0)[0][0])
 
    r_bond_number_in_level1_molecule = level1_adjacency_matrix[
        r_index_in_level1_molecule_new, r_bonded_atom_index_in_level1_molecule]
 
    level1_adjacency_matrix = np.delete(level1_adjacency_matrix, r_index_in_level1_molecule_new, 0)
    level1_adjacency_matrix = np.delete(level1_adjacency_matrix, r_index_in_level1_molecule_new, 1)
 
    for i in range(len(r_index_in_level1_molecule_new)):
        level1_atoms.remove('*')
    level1_size = level1_adjacency_matrix.shape[0]
 
    generated_molecule_atoms = level1_atoms[:]
    generated_adjacency_matrix = level1_adjacency_matrix.copy()
 
    frag_permutations = list(itertools.permutations([main_mol]))
    fragment_list = frag_permutations[r_position]
 
    for r_number_in_molecule, fragment_molecule in enumerate(fragment_list):  
 
        fragment_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(fragment_molecule)
 
        for bond in fragment_molecule.GetBonds():
            fragment_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(
                bond.GetBondType())
            fragment_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(
                bond.GetBondType())
        fragment_atoms = []
        for atom in fragment_molecule.GetAtoms():
            fragment_atoms.append(atom.GetSymbol())
 
        r_index_in_fragment_molecule = fragment_atoms.index('*')
 
        r_bonded_atom_index_in_fragment_molecule = \
            np.where(fragment_adjacency_matrix[r_index_in_fragment_molecule, :] != 0)[0][0]
 
        if r_bonded_atom_index_in_fragment_molecule > r_index_in_fragment_molecule:
            r_bonded_atom_index_in_fragment_molecule -= 1
 
        fragment_atoms.remove('*')
        fragment_adjacency_matrix = np.delete(fragment_adjacency_matrix, r_index_in_fragment_molecule, 0)
        fragment_adjacency_matrix = np.delete(fragment_adjacency_matrix, r_index_in_fragment_molecule, 1)
 
        main_size = generated_adjacency_matrix.shape[0]
        generated_adjacency_matrix = np.c_[generated_adjacency_matrix, np.zeros(
            [generated_adjacency_matrix.shape[0], fragment_adjacency_matrix.shape[0]], dtype='int32')]
        generated_adjacency_matrix = np.r_[generated_adjacency_matrix, np.zeros(
            [fragment_adjacency_matrix.shape[0], generated_adjacency_matrix.shape[1]], dtype='int32')]
 
        generated_adjacency_matrix[r_bonded_atom_index_in_level1_molecule[r_number_in_molecule], 
                                   r_bonded_atom_index_in_fragment_molecule + main_size] = \
            r_bond_number_in_level1_molecule[r_number_in_molecule]
 
        generated_adjacency_matrix[r_bonded_atom_index_in_fragment_molecule + main_size, 
                                   r_bonded_atom_index_in_level1_molecule[r_number_in_molecule]] = \
            r_bond_number_in_level1_molecule[r_number_in_molecule]
 
        generated_adjacency_matrix[main_size:, main_size:] = fragment_adjacency_matrix
 
        # integrate atoms
        generated_molecule_atoms += fragment_atoms
 
    # generate structures 
    generated_molecule = Chem.RWMol()
    atom_index = []
 
    for atom_number in range(len(generated_molecule_atoms)):
        atom = Chem.Atom(generated_molecule_atoms[atom_number])
        molecular_index = generated_molecule.AddAtom(atom)
        atom_index.append(molecular_index)
 
    for index_x, row_vector in enumerate(generated_adjacency_matrix):    
        for index_y, bond in enumerate(row_vector):      
            if index_y <= index_x:
                continue
            if bond == 0:
                continue
            else:
                generated_molecule.AddBond(atom_index[index_x], atom_index[index_y], bond_list[bond])
 
    generated_molecule = generated_molecule.GetMol()
 
    return generated_molecule

smiles= []

          
for i in range (len(frag_4dummy)) :
    fragment4 = frag_4dummy[i]
    for j in range(len (frag_3dummy)):
        fragment3 = frag_3dummy[j]
        for k in range (len(frag_1dummy)):
            fragment1 = frag_1dummy[k]
            for l in range(len(frag_2dummy)):
                    fragment2 = frag_2dummy[l]
                    mol = structure_generator(fragment4,fragment1 )
                    
                    mol1 = structure_generator(fragment3, mol)
                    mol = structure_generator(fragment2,mol1)
                    smile = Chem.MolToSmiles(mol)
                    smiles.append(smile)
dic={"SMILES":smiles}
DataFrame(dic).to_csv(r"brics.csv")

'''
main_mol = frag_1dummy[168]
fragment1 = frag_2dummy[4]
fragment2 = frag_1dummy[5]
fragment3 = frag_2dummy[50]
mol = structure_generator(fragment1, fragment3,main_mol)
mol = structure_generator(main_mol, fragment3, mol)
img3 = Draw.MolsToGridImage([fragment4, fragment1, fragment2, mol],
                           molsPerRow=4,
                           legends=['fragment4','fragment1','fragment2','generated_molecule'])
img3

'''