from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
m = Chem.MolFromSmiles('O=C(NCc1cc(OC)c(O)cc1)Cc1cocc1CC')
core = MurckoScaffold.GetScaffoldForMol(m)
m_core = [m, core]
Draw.MolsToGridImage(m_core, subImgSize=(250, 250))
MurckoScaffold.MurckoScaffoldSmilesFromSmiles('O=C(NCc1cc(OC)c(O)cc1)Cc1cocc1CC')
import os
from rdkit.Chem import rdRGroupDecomposition as rdRGD
from rdkit.Chem import RDConfig
fName = os.path.join(RDConfig.RDDocsDir, 'Book\\data\\s1p_chembldoc89753.txt')
suppl = Chem.SmilesMolSupplier(fName, delimiter=",", smilesColumn=9, nameColumn=10)
ms = [x for x in suppl if x]
core = Chem.MolFromSmarts('[*:1]c1nc([*:2])on1')
core

res, unmatched = rdRGD.RGroupDecompose([core], ms, asSmiles=True)
print(len(res), len(unmatched))
res[1]


Chem.Draw.MolsToGridImage([ms[22], core, Chem.MolFromSmiles(res[0]['R2'])], molsPerRow=3, subImgSize=(300, 300))
