# Explaining PLIG predictions

Predictions on PLIGs can be explained using [GNNExplainer](https://arxiv.org/abs/1903.03894). 

1. Train a model. This can be done following the instructions in the [models_main](../models_main) folder. For convenience and reproducability, a trained model with the best-performing architecture is available in [example_model](example_model), which is used in the following steps to generate and visualize explanations. 

2. Generate the explanations for a chosen instance the `generate_explanations.py` program. It uses the PyTorch Geometric implementation of GNNExplainer.
    > For the example in the paper, use `python generate_explanations.py CONFIG PDB_ID (optional)NUM_EXPLANATIONS`

    * The explanations will be saved in the [explanation_outputs/](explanation_outputs/) directory.
    * While the model can be trained using the program in [models_main](../models_main), the explainer requires the data as decomposed matrices instead of a PyTorch Geometric Data objects. Thus the model needs to be slighty altered, which for this example can be found in [example_model/x_gat_PLIG_no_p.py](explainable_models/x_gat_PLIG_no_p.py). 
    * Note that GNNExplainer has since been migrated to a dedicated submodule within PyTorch Geometric, the implementation we use can be found in [explanation_utils/gnn_explainer.py](explanation_utils/gnn_explainer.py).

3. Visualize the explanations. For this, the [visualize_explanations.ipynb](visualize_explanations.ipynb) notebook can be used. With the pretrained example, this works out of the box. 

4. 3D-visualization with PyMol. The importance weights can also be mapped to the 3D-structure, for which a custom script was created. For clarity, important water molecules and interactions have to be implemented manually. The script for the example in the paper can be found in [pymol_visualization/pymol_visualization.py](pymol_visualization/pymol_visualization.py)