from pymol_utils import rgb_to_hex, calc_color, generate_label
from pymol import cmd
import pymol
import pandas as pd

# set data file
pdb_id = "5dwr"

# some hints
print(f"\nShowing '{pdb_id}'")

# locate data / set variables
ligand_file = "../explainable_ligands/5dwr_ligand.mol2"
protein_file = "5dwr_protein.pdb"
node_mask_file = "../explanation_outputs/5dwr_100_explanations/node_masks.csv"
edge_mask_file = "../explanation_outputs/5dwr_100_explanations/node_edge_masks_aggr.csv"
ligand_name = pdb_id+'_ligand'
protein_name = pdb_id+'_protein'

# delete previos visualization
cmd.reinitialize()

# load data
cmd.load(ligand_file)
cmd.load(protein_file)
df = pd.read_csv(node_mask_file, index_col=0)
edge_df = pd.read_csv(edge_mask_file)

# show sidechains around ligand
cmd.set("cartoon_side_chain_helper", 1)

# define binding pocket
cmd.select("binding_site", f"byres {ligand_name} expand 6", enable=0)
cmd.show("stick", "binding_site")
# binding pocket without ligand
# cmd.select("binding_site", f"byres {protein_name} within 6 of {ligand_name}", enable=1) 

# zoom
# cmd.zoom(f"{ligand_name}")
cmd.zoom("binding_site")

# make ball and stick model
cmd.select("_", f"{ligand_name}")
cmd.hide("sticks", "_")
cmd.show("spheres", "_")
cmd.set("sphere_scale", 0.20)
cmd.set("stick_radius", 0.20)
cmd.hide("spheres", f"{ligand_name} and ele h")

# find polar contacts
cmd.dist("h_bonds1", f"{ligand_name}", f"{protein_name} and not solvent", mode=2) # INTERACTING WATER: use "all" instead of "{protein_name} and not solvent"
cmd.dist("h_bonds1", f"{ligand_name}", f"all", mode=2) 
cmd.hide("labels", "h_bonds1")

# hide waters
cmd.hide("nonbonded", f"{protein_name}")

# hide non-polar hydrogens in protein
cmd.hide("sticks", f"{protein_name} and ele h and not (neighbor (ele n+o))") 

### HARDCODED ### show interacting waters and backbone
interacting_waters = ["/5dwr_protein///HOH`97/O", "/5dwr_protein///HOH`112/O", "/5dwr_protein///HOH`1/O"] # INTERACTING WATERS
cmd.select("interacting_waters", " or ".join(interacting_waters)) # INTERACTING WATERS
cmd.show("nb_spheres", "interacting_waters") # INTERACTING WATERS
interacting_backbone = ["A/GLU`171/O", "A/GLU`171/C", "A/GLU`171/CA", "A/ASP`186/N", "A/ASP`186/CA", 
                        "A/ASP`186/H", "A/PHE`187/H", "A/PHE`187/N", "A/PHE`187/CA"]
interacting_sidechain = ["/5dwr_protein//A/GLU`89/OE2"]
cmd.select("interacting_backbones", " or ".join(interacting_backbone), enable=1)
cmd.create("interacting_bb", "interacting_backbones")
cmd.util.cbab("interacting_bb")
cmd.set("cartoon_gap_cutoff", 0)
cmd.set("cartoon_side_chain_helper", 0, "interacting_bb")
cmd.delete("interacting_backbones")
cmd.dist("h_bonds2", "interacting_waters", " or ".join(interacting_waters+interacting_backbone+interacting_sidechain), mode=2) # INTERACTING WATERS
cmd.hide("labels", "h_bonds2") # INTERACTING WATERS

### END HARDCODED ###

# make list of atom indices
ranks = []
elements = []
cmd.iterate(ligand_name, "ranks.append(int(rank)); elements.append(elem)")

# labeling presets
cmd.set("label_font_id", 7)
cmd.set("label_size", 20)
cmd.set("label_outline_color", "black")
cmd.set("label_color", "black")

cmd.bg_color("white")

# color protein
cmd.util.cbab(f"{protein_name}")

# variables for mask coloring 
max_r, max_g, max_b = 1., 1., 1. # max rgb values
color = (0x89, 0xd5, 0x48) # 89d548
min_r, min_g, min_b = color[0]/255, color[1]/255, color[2]/255 # min rgb values (i.e. strongest color)
max_mask = max(df["node_masks"])
min_mask = min(df["node_masks"])

# color ligand atoms by importance, label with rank and element
for rnk in ranks:
    if rnk < len(df["node_masks"]):
        cmd.select("_", f"{ligand_name} and rank {rnk}")
        mask_value = df["node_masks"].iloc[rnk]
        r = calc_color(mask_value, max_mask, min_mask, max_r, min_r)
        g = calc_color(mask_value, max_mask, min_mask, max_g, min_g)
        b = calc_color(mask_value, max_mask, min_mask, max_b, min_b)
        cmd.color("0x"+rgb_to_hex((r, g, b)), "_")
        cmd.label("_", "generate_label(rnk, elem)")

# reset varaibles for mask coloring
max_mask = max(edge_df["node_edge_masks_aggr"])
min_mask = min(edge_df["node_edge_masks_aggr"])
# color ligand bonds by importance
    # create a list of selections a1-a2, a2-a3, a2-a4, ... => from edge_df col 0
    # make selections to objects
    # color each object by importance
for bond in edge_df["Unnamed: 0"]:
    a1, a2 = eval(bond)
    cmd.select("_", f"{ligand_name} and (rank {a1} or rank {a2})")
    selection_name = "bond_"+str(a1)+"_"+str(a2)
    cmd.create(f"{selection_name}", "_")
    cmd.hide("spheres", f"{selection_name}")
    cmd.valence("guess", f"{selection_name}")
    cmd.show("sticks", f"{selection_name}")
    mask_value = edge_df.loc[edge_df["Unnamed: 0"] == bond, "node_edge_masks_aggr"].iloc[0]
    r = calc_color(mask_value, max_mask, min_mask, max_r, min_r)
    g = calc_color(mask_value, max_mask, min_mask, max_g, min_g)
    b = calc_color(mask_value, max_mask, min_mask, max_b, min_b)
    cmd.color("0x"+rgb_to_hex((r, g, b)), f"{selection_name}")
    cmd.group("bond_group", f"{selection_name}")

# label sidechains
cmd.set("label_size", 15, f"{protein_name}")
cmd.set("label_color", "black", f"{protein_name}")
cmd.set("label_outline_color", "black", f"{protein_name}")
# cmd.label(f"{protein_name} and name CA", "resn+resi")

# make bonds without color for hydrogens and aromaticity
# otherwise the double bond hint is rotated randomly around the bond
cmd.create("ligand_bonds", f"{ligand_name}")
cmd.hide("spheres", "ligand_bonds")
cmd.show("sticks", "ligand_bonds")
cmd.color("white", "ligand_bonds")
# hide non-polar hydrogens
cmd.hide("sticks", "ligand_bonds and ele h and not (neighbor (ele n+o))") 
# make sticks slightly smaller, otherwise raytracing doesn't show bond importance
cmd.set("stick_radius", 0.1999, "ligand_bonds")

# shadows off
cmd.set("ray_shadows", 0)

print("Type 'establishing_shot()' to generate surface view.\nRerun script to view importances.")
def establishing_shot():
    # shadows on
    cmd.set("ray_shadows", 1)

    # hide all
    cmd.hide()

    # new protein object without waters (so nonstandard AAs can be included)
    cmd.create("protein_no_solvent", f"{protein_name} and not solvent")
    
    # include nonstandard AAs
    cmd.set("surface_mode", 1)

    # surface on and color
    cmd.show("surface", "protein_no_solvent")
    cmd.color("white", "protein_no_solvent")

    # surface quality
    cmd.set("surface_quality", 1)

    # show copy of ligand
    cmd.load(ligand_file, "simple_ligand")
    cmd.util.cba("forest", "simple_ligand")
    cmd.hide("sticks", "simple_ligand and ele h and not (neighbor (ele n+o))")

    # zoom on protein
    cmd.zoom("protein_no_solvent")

    # ray
    cmd.ray()

cmd.ray()

def pic(filename: str):
    cmd.png(filename, width="8cm", height="8cm", dpi=300, ray=1)
    