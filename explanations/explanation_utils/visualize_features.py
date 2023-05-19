
# %% imports
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import os


# %% 
def get_figsize(matrix, xticklabels=False, yticklabels=False, add_node_feat=False):
    fig_x = (matrix.shape[1]+add_node_feat)/2
    fig_y = (matrix.shape[0]+add_node_feat)/2
    return (fig_x, fig_y)

def plot_features(explanation, labels, cmap='Blues', dpi=600):
    fig, ax = plt.subplots(dpi=dpi)
    sns.heatmap(explanation, annot=True, fmt='.2f', annot_kws={"fontsize":3.1}, 
                cbar=False, cmap=cmap, ax=ax, yticklabels=labels,
                xticklabels=False)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6, length=2, width=.3)
    sns.despine(top=False, right=False)
    plt.tight_layout()
    return fig

def plot_indiv(explanation, node_labels, feat_labels, cmap='Blues', dpi=600):
    fig, ax = plt.subplots(dpi=dpi)
    sns.heatmap(explanation, annot=True, fmt='.1f', annot_kws={"fontsize":2.9}, 
                cbar=False, cmap=cmap, ax=ax, yticklabels=node_labels,
                xticklabels=feat_labels)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=4.5, length=2, width=.3, bottom=False, labelbottom=False,
                   top=True, labeltop=True, pad=0.5)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="left",
             rotation_mode="anchor")
    sns.despine(top=False, right=False)
    plt.tight_layout()
    return fig

def plot_molecule(smiles, pattern=None, legend=''):
    if os.path.isfile(smiles):
        mol = Chem.MolFromMol2File(smiles)
        Chem.Compute2DCoords(mol)
    else:
        mol = Chem.MolFromSmiles(smiles)
    # other labeling possibilities: SetProp('molAtomMapNumber', ...) or 
    #                               SetProp('atomLabel', ...)
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx()))
    if pattern:
        pattern = Chem.MolFromSmarts(pattern)
        
        alist = []
        blist = []
        for match in mol.GetSubstructMatches(pattern):
            alist.extend(match)
        
        for ha1 in alist:
            for ha2 in alist:
                if ha1 > ha2:
                    b = mol.GetBondBetweenAtoms(ha1, ha2)
                    if b:
                        blist.append(b.GetIdx())
        
        d = Draw.rdMolDraw2D.MolDraw2DCairo(600, 600)
        d.drawOptions().useBWAtomPalette()
        d.DrawMolecule(mol, legend=legend, highlightAtoms=alist, highlightBonds=blist)
        d.FinishDrawing()
        im = Image.open(io.BytesIO(d.GetDrawingText()))
        return im
    else:
        return Draw.MolToImage(mol, (600,600))
    
def plot_highlight_mol(smiles, a_highlights={}, b_highlights={}, legend='', number=True):
    # https://www.rapidtables.com/web/color/RGB_Color.html
    # highlight dict(atomIdx: list((r, g, b)))
    # e.g. h = {0: [(1.,0.,0.)], 3: [(0.,.55, 0.), (.33, .33, .33)]}
    Chem.rdDepictor.SetPreferCoordGen(True)
    if os.path.isfile(smiles):
        mol = Chem.MolFromMol2File(smiles)
        Chem.Compute2DCoords(mol)
    else:
        mol = Chem.MolFromSmiles(smiles)
    mol = Draw.PrepareMolForDrawing(mol)
    
    if number:
        for atom in mol.GetAtoms():
            atom.SetProp('atomNote', str(atom.GetIdx()))
    
    h_rads = {}
    h_lw_mult = {}
    
    d = Draw.rdMolDraw2D.MolDraw2DCairo(600, 600)
    d.drawOptions().useBWAtomPalette()
    if type(a_highlights) == dict:
        d.DrawMoleculeWithHighlights(mol,legend, a_highlights, b_highlights, h_rads,
                                 h_lw_mult, -1)
    elif type(a_highlights) == set:
        d.DrawMolecule(mol, a_highlights, b_highlights)
    d.FinishDrawing()
    im = Image.open(io.BytesIO(d.GetDrawingText()))
    return im

def calc_color(mask_value, max_mask, min_mask, max_color, min_color):
    m = (min_color - max_color)/(max_mask - min_mask)
    b = min_color - m * max_mask
    color = mask_value * m  + b
    return color
        

def calc_bond_highlight_color(smiles, aggr_mask, color):
    max_r, max_g, max_b = 1., 1., 1.
    if color:
        min_r, min_g, min_b = color[0]/255, color[1]/255, color[2]/255
    else:
        min_r, min_g, min_b = 0., 86/255, 138/255 # Blues
        # min_r, min_g, min_b = 175/255, 20/255, 122/255 # Pink
    
    max_mask = max(aggr_mask.values()).item()
    min_mask = min(aggr_mask.values()).item()
    
    b_highlights = {}
    
    if os.path.isfile(smiles):
        mol = Chem.MolFromMol2File(smiles)
        Chem.Compute2DCoords(mol)
    else:
        mol = Chem.MolFromSmiles(smiles)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        for k in aggr_mask.keys():
            if set(k) == set((a1, a2)):
                r = calc_color(aggr_mask[k].item(), max_mask, min_mask, max_r, min_r)
                g = calc_color(aggr_mask[k].item(), max_mask, min_mask, max_g, min_g)
                b = calc_color(aggr_mask[k].item(), max_mask, min_mask, max_b, min_b)
                
                b_highlights.update({bond.GetIdx(): [(r, g, b)]})
                    
    return b_highlights
    
def calc_atom_highligh_color(smiles, feat_mask, color):
    max_r, max_g, max_b = 1., 1., 1.
    if color:
        min_r, min_g, min_b = color[0]/255, color[1]/255, color[2]/255
    else:
        min_r, min_g, min_b = 0., 86/255, 138/255 # Blues
        # min_r, min_g, min_b = 175/255, 20/255, 122/255 # Pink
    
    
    max_mask = max(feat_mask).item()
    min_mask = min(feat_mask).item()
    
    a_highlights = {}
    
    if os.path.isfile(smiles):
        mol = Chem.MolFromMol2File(smiles)
        Chem.Compute2DCoords(mol)
    else:
        mol = Chem.MolFromSmiles(smiles)
    assert len(feat_mask) == mol.GetNumHeavyAtoms()
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        r = calc_color(feat_mask[i].item(), max_mask, min_mask, max_r, min_r)
        g = calc_color(feat_mask[i].item(), max_mask, min_mask, max_g, min_g)
        b = calc_color(feat_mask[i].item(), max_mask, min_mask, max_b, min_b)
        
        a_highlights.update({i: [(r, g, b)]})
    
    return a_highlights

def plot_highlight_mol_with_calc_colors(smiles, feat_mask, aggr_mask, color=None):
    return plot_highlight_mol(smiles, 
                              calc_atom_highligh_color(smiles, feat_mask, color=color),
                              calc_bond_highlight_color(smiles, aggr_mask, color=color))