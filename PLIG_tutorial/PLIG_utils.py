# Script to create Proximity Interaction Graphs. 
# Author: Marc Moesser
# This script uses adapted functions from the ECIF script supplied in https://github.com/DIFACQUIM/ECIF

import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from itertools import product


def GetAtomType(atom):
# Atom types are defined as follows:
# 1) Atom symbol
# 2) Explicit Valence
# 3) # Heavy Atom Neighbors
# 4) # Hydrogen Neighbors
# 5) Boolean: Is atom aromatic?
# 6) Boolean: is atom in a ring?

# This function can be used to identify all unique protein atom types in the dataset.
    
    AtomType = [atom.GetSymbol(),
                str(atom.GetExplicitValence()),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),
                str(int(atom.GetIsAromatic())),
                str(int(atom.IsInRing())), 
               ]

    return(";".join(AtomType))


def LoadSDFasDF(mol):
# This function converts the input ligand (.MOL file) into a pandas DataFrame with the ligand atom position in 3D (X,Y,Z)
    
    m = mol
    
    atoms = []

    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            entry = [int(atom.GetIdx())]
            entry.append(str(atom.GetSymbol()))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            atoms.append(entry)

    df = pd.DataFrame(atoms)
    df.columns = ["ATOM_INDEX","ATOM_TYPE","X","Y","Z"]
    
    return(df)


def LoadPDBasDF(PDB, Atom_Keys):
# This function converts a protein PDB file into a pandas DataFrame with the protein atom position in 3D (X,Y,Z)

    prot_atoms = []
    
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (len(i[12:16].replace(" ","")) < 4 and i[12:16].replace(" ","")[0] != "H") or (len(i[12:16].replace(" ","")) == 4 and i[12:16].replace(" ","")[1] != "H" and i[12:16].replace(" ","")[0] != "H"):
                prot_atoms.append([int(i[6:11]),
                         i[17:20]+"-"+i[12:16].replace(" ",""),
                         float(i[30:38]),
                         float(i[38:46]),
                         float(i[46:54])
                        ])
                
    f.close()
    
    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX","PDB_ATOM","X","Y","Z"])
    df = df.merge(Atom_Keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    if list(df["ATOM_TYPE"].isna()).count(True) > 0:
        print("WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")
    return(df)

def GetAtomContacts(PDB_protein, mol, Atom_Keys, distance_cutoff=6.0):
# This function returns the list of protein atom types the ligand interacts with for a given distance cutoff
# cutoff = 6 Angstrom is standard
    
    # Protein and ligand structure are loaded as pandas DataFrame
    Target = LoadPDBasDF(PDB_protein, Atom_Keys)
    Ligand = LoadSDFasDF(mol)
    
    # A cubic box around the ligand is created using the proximity threshold specified (here distance_cutoff = 6 Angstrom by default).
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]

    # Calculate the possible pairs
    Pairs = list(product(Target["ATOM_TYPE"], Ligand["ATOM_INDEX"]))
    Pairs = [str(x[0])+"-"+str(x[1]) for x in Pairs]
    Pairs = pd.DataFrame(Pairs, columns=["ATOM_PAIR"])
    
    Distances = cdist(Target[["X","Y","Z"]], Ligand[["X","Y","Z"]], metric="euclidean")
    Distances = Distances.reshape(Distances.shape[0]*Distances.shape[1],1)
    Distances = pd.DataFrame(Distances, columns=["DISTANCE"])

    #Select pairs with distance lower than the cutoff
    Pairs = pd.concat([Pairs,Distances], axis=1)
    Pairs = Pairs[Pairs["DISTANCE"] <= distance_cutoff].reset_index(drop=True)
    
    contact_pair_list = [i.split("-")[0] for i in Pairs["ATOM_PAIR"]]
    Pairs["PROT_ATOM"] = contact_pair_list
    Pairs["LIG_ATOM"] = [int(i.split("-")[1]) for i in Pairs["ATOM_PAIR"]]
    
    return Pairs


def atom_features(atom, features=["num_heavy_atoms", "total_num_Hs", "explicit_valence", "is_aromatic", "is_in_ring"]):
    # Computes the ligand atom features for graph node construction
    # The standard features are the following:
    # num_heavy_atoms = # of heavy atom neighbors
    # total_num_Hs = # number of hydrogen atom neighbors
    # explicit_valence = explicit valence of the atom
    # is_aromatic = boolean 1 - aromatic, 0 - not aromatic
    # is_in_ring = boolean 1 - is in ring, 0 - is not in ring

    feature_list = []
    if "num_heavy_atoms" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"]))
    if "total_num_Hs" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"]))
    if "explicit_valence" in features: #-NEW ADDITION FOR PLIG
        feature_list.append(atom.GetExplicitValence())
    if "is_aromatic" in features:
        
        if atom.GetIsAromatic():
            feature_list.append(1)
        else:
            feature_list.append(0)
    if "is_in_ring" in features:
        if atom.IsInRing():
            feature_list.append(1)
        else:
            feature_list.append(0)
    return np.array(feature_list)


def atom_features_PLIG(atom_idx, atom, contact_df, extra_features, Atom_Keys):
    # Generates the protein-ligand interaction features for the PLIG creation

    possible_contacts = list(dict.fromkeys(Atom_Keys["ATOM_TYPE"]))
    feature_list = np.zeros(len(possible_contacts), dtype=int)
    contact_df_slice = contact_df[contact_df["LIG_ATOM"] == atom_idx]

    #count the number of contacts between ligand and protein atoms
    for i,contact in enumerate(possible_contacts):
        for k in contact_df_slice["PROT_ATOM"]:
            if k == contact:
                feature_list[i] +=1
                
    extra_feature_array = atom_features(atom, extra_features)
    output = np.append(extra_feature_array, feature_list)

    return output

def mol_to_graph(mol, contact_df, Atom_Keys, extra_features=["num_heavy_atoms", "total_num_Hs", "explicit_valence","is_aromatic", "is_in_ring"]):
    #Final function to summarize the generation of PLIGS

    #Extra features are any extra features to be added to the protein-ligand interaction features.
    #In this work, we use the following ligand-based features as the ligand atom node features that are added to the interaction features:

    #num_heavy_atoms
    #total_num_hs
    #explicit_valence
    #is_aromatic
    #is_in_ring

    #However, this is freely customizable! Any additional features can be added here.



    c_size = len([x.GetSymbol() for x in mol.GetAtoms() if x.GetSymbol() != "H"])
    features = []
    heavy_atom_index = []
    idx_to_idx = {}
    counter = 0

    # Generate nodes
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            idx_to_idx[atom.GetIdx()] = counter
            counter +=1
            heavy_atom_index.append(atom.GetIdx())
            feature = atom_features_PLIG(atom.GetIdx(), atom, contact_df, extra_features, Atom_Keys)
            features.append(feature)

    #Generate edges
    edges = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
            edges.append([idx_to_idx[bond.GetBeginAtomIdx()], idx_to_idx[bond.GetEndAtomIdx()]])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    #return molecular graph with its node features and edge indices
    return c_size, features, edge_index
