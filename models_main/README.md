# Protein-Ligand Interaction Graphs codebase

The "data" folder contains all necessary data to train and test all model+feature combinations on crystal and docked poses for the PDBbind general 2020 + refined 2016 combined set called "PDBBind_combined"

The "models" folder contains all models with already optimized hyperparameters (against PDBBind refined 2016) to be used for protein-ligand affinity prediction.

The "example_configs" folder contains the necessary config files to run the create_general_data.py and training.py scripts for all models as well as an example config file that uses a pre-trained model.

The "PDBbind_combined_pdbcodes.txt" is a list of all 14981 PDB codes used
## How to use the code

#### Installing packages
To install the necessary python packages go to the base folder in this repo and run:
```
conda env create -f torch_geo.yml
```

#### Feature generation
This repo comes with the PLIG, ECIF, FCFP and ECFP features pre-computed in the "data folder".

#### Important files for training
+ config.yml: A config file is needed for every run to set the necessary paremeters such as: model architecture, features, crystal vs docked etc.
+ create_general_data.py: This PyTorch data loader script that either generates fresh ligand-based graphs from the supplied SMILES, or loads in the pre-computed features above. This script will deposit the processed dataset split (train, valid, test) as .pt files in the "data/processed/" folder.
+ training.py: The main training script.

#### Running the model
+ First, run the create_general_data.py script with the desired config file
```
python create_general_data.py <configuration file>
```

+ Second, run the training script with the config file and a specified seed (integer) and an optional filename extension (str).
```
python training.py <configuration file> <seed> <file_name_extension> 
```

The training.py script will output 3 files total. A .model file will be deposited in the output/trained_model folder after successful training that saves the pre-trained weights. A "training_history" .csv file and a "test_predictions" .csv file will be added to the "output/training_logs" folder that specifies the performance at each epoch and saves the final prediction vs ground truth for every protein-ligand complex.

#### Running the pure_prediction.py script

If instead of training a new model, you want to test the performance of a pre-trained model on a new dataset, use the pure_prediction.py script. 
To run it:
```
python pure_prediction <pure_prediction configuration file>
```
