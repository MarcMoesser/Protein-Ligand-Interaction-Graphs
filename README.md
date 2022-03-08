# Protein-Ligand-Interaction-Graphs

This is the repository for https://www.biorxiv.org/content/10.1101/2022.03.04.483012v1?rss=1

![PLIG](https://user-images.githubusercontent.com/57324636/156644947-34fffcd7-3f52-4922-a70c-ba3c97fda76d.png)

### Python Packages

Install the needed python packages using conda with the following command:

```
conda env create -f torch_geo.yml
```

Alternatively, the following packages can be install manually (not preferred). Please make sure the correct version of PyTorch is installed!!! Other versions of PyTorch (eg. 1.10) will crash the models.

+ conda create --name torch_geo
+ conda install -c conda-forge rdkit=2021
+ conda install pyyaml
+ conda install pytorch=1.9.0 cpuonly -c pytorch 
+ conda install pytorch-geometric -c rusty1s -c conda-forge
+ conda install -c conda-forge optuna
+ conda install -c conda-forge gpytorch
+ conda install -c conda-forge tqdm


## 1) Create Protein-Ligand Interaction Graphs

The code needed to generate PLIGs from a protein-ligand complex can be found in the "PLIG_tutorial/" folder.
## 2) Run a PLIG, ligand-based GNN or MLPNet model

All GNN implementations of PLIGs, ligand-based GNNs, as well as the MLPNet implementation of ECFP/FCFP and ECIF fingperints can be found in the "models_main/" folder.

The following data is supplied:
##### i) All hyperparameter tuned GNN PLIG, GNN ligand-based, MLPNet ECIF and MLPNet ECFP/FCFP models 
##### ii) Pre-prepared features (PLIGs, ECIF, ECFP/FCFP) for the PDBbind 2020 general + PDBbind 2016 refined set
##### iii) All config files needed to train all model+feature combinations on crystal and docked poses.

## 3) Figures and data presented in the publication

All raw-data to generate the figure published in https://www.biorxiv.org/content/10.1101/2022.03.04.483012v1?rss=1 can be found in the "publication" folder
