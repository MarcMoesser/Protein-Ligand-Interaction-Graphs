# 3D-Visualization of explanations using PyMol

The explanations can also be automatically mapped onto the 3D-structure of the molecule. However, interacting waters and sidechains have to be manually added to the images. To recreate the images of the publication, the `pymol_visualization.py` script has to be executed within PyMol, creating the visualization for the protein-ligand complex 5DWR. 

If you want to visualize different explanations, please replace the `### HARDCODED ###` block in the script with the appropriate sidechains and waters.

Please also note that only one exemplary PDB-file is supplied. To visualize other files, please add the corresponding PDB file without the ligand to the folder.