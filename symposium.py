import numpy as np
import scanpy as sc
import dandelion as ddl
import networkx as nx
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import MDS
from umap import UMAP

from medoid import *
from mse import *

warnings.filterwarnings("ignore")

adata = sc.read("covid/BCR_Severe.h5ad")
adata = adata[adata.obs["patient_id"] == "CV1078"] # Subset to a single patient
                                                   # There is actually a typo here (should be "CV0178" instead of "CV1078")
                                                   # But when I correct this typo, ddl.pl.clone_network() throws an error when it's called.
                                                   # Very weird.

vdj = ddl.read_10x_vdj("covid/scbcr_cellranger_filtered_contig_annotations.csv")
vdj = vdj[vdj.data.patient_id == "CV0178"]
vdj, adata = ddl.pp.check_contigs(vdj, adata)

ddl.tl.find_clones(vdj, key = "junction")
ddl.tl.clone_size(vdj)
ddl.tl.generate_network(vdj, key = "junction")
ddl.tl.transfer(adata, vdj)

filtered = vdj[vdj.metadata.junction_VJ != "None"]

# Build a list of tuples with relevant information for each medoid (counting singletons as their own medoids)
# To do: need to filter or deal with the weird combined sequences e.g. TGTAACTCCCGGGACAGCAGTGGTAACCATCGTGTCTTC|TGCCAGTCCTATGACAGCAGCCTGAGTGGTGTGGTATTC
# ^ there is a large proportion of these.
# To do: some cells belong to more than one clonotype e.g. B_VJ_106_1_3|B_VJ_148_1_5 which stuffs up this algorithm
medoids = [{                                   
              "sequence" : find_medoid(x[1].junction_VJ),    # Medoid sequence ATGTAGTACGCTAGCT
              "name" : x[0],                                 # Clonotype name  B_VDJ_92_12_1_VJ_67_1_1
              "size" : len(x[1]),                            # Clonotype size  35.0
           }                                    
           
           for x in filtered.metadata.groupby("clone_id")]


# Build distance matrix from medoid sequences
dmat_medoids = get_sequence_distance_matrix([M["sequence"] for M in medoids])



## LAYOUT ALGORITHMS

# Dandelion default

# Note we can't directly compare these to the others since Dandelion computes the positions before we do our filtering and medoid steps.
pos_ddl_ = vdj.layout[0]
pos_ddl = {i : pos_ddl_[key] for i, key in enumerate(pos_ddl_)}


# UMAP
umap = UMAP(metric = "precomputed")
transformed = umap.fit_transform(dmat_medoids)
pos_umap = {i : np.array(x) for i, x in enumerate(transformed)}

# MDS
mds = MDS(metric = True, dissimilarity = 'precomputed')
transformed = mds.fit_transform(dmat_medoids)
pos_mds = {i : np.array(x) for i, x in enumerate(transformed)}


# Force-directed
medoids_dmat2 = dmat_medoids + 0.01 # Add a small quantity elementwise so that NetworkX doesn't treat 0-distances as absence of an edge.
                                    # This is a hack: better way would be to construct the nx.Graph manually

np.fill_diagonal(medoids_dmat2, 0)

G2 = nx.from_numpy_array(medoids_dmat2)

medoids_dmat2_trimmed = medoids_dmat2
medoids_dmat2_trimmed[medoids_dmat2_trimmed > 30] = 0 # Trim high-distance edges to see if it improves the force-directed layout
                                                      # If there's a way to ensure the graph stays connected we should do that

G3 = nx.from_numpy_array(medoids_dmat2_trimmed)

# Kamada-Kawai has two advantages: don't need to convert distances to proximity and it copes better with graphs with lots of edges (cf. Fruchterman-Reingold)
# The running time of either Kamada-Kawai or Fruchterman-Reingold is inconveniently long however.
pos_force = nx.kamada_kawai_layout(G2)
pos_force_trimmed = nx.kamada_kawai_layout(G3)



## PLOTTING

# We want the node numbers in the NetworkX graph that correspond to big and small clonotypes, so that we can plot them separately.
# We will treat clonotype sizes > 2 as big clonotypes.
big_clonotypes = [i for i, M in enumerate(medoids) if M["size"] > 2]
small_clonotypes = [i for i, M in enumerate(medoids) if M["size"] <= 2]

# Plot the big clonotypes with different sizes and colours so they stand out
big_node_sizes = [20 + 5 * M["size"] for M in medoids if M["size"] > 2]
big_node_colours = [plt.cm.tab20(i) for i, _ in enumerate(big_node_sizes)]


G = nx.from_numpy_array(dmat_medoids)

# We need to plot small (boring) clonotypes first so they don't cover the big (interesting) clonotypes
G_big_clonotypes = G.subgraph(big_clonotypes)
G_small_clonotypes = G.subgraph(small_clonotypes)


# Dandelion default
ddl.pl.clone_network(adata)

# UMAP
nx.draw_networkx(G_small_clonotypes, pos = pos_umap, node_size = 20, node_color = "lightgray", edgelist = [], with_labels = False)
nx.draw_networkx(G_big_clonotypes, pos = pos_umap, node_size = big_node_sizes, node_color = big_node_colours, edgelist = [], with_labels = False) 

plt.show()


# MDS
nx.draw_networkx(G_small_clonotypes, pos = pos_mds, node_size = 20, node_color = "lightgray", edgelist = [], with_labels = False)
nx.draw_networkx(G_big_clonotypes, pos = pos_mds, node_size = big_node_sizes, node_color = big_node_colours, edgelist = [], with_labels = False) 

plt.show()


# Force-directed
nx.draw_networkx(G_small_clonotypes, pos = pos_force, node_size = 20, node_color = "lightgray", edgelist = [], with_labels = False)
nx.draw_networkx(G_big_clonotypes, pos = pos_force, node_size = big_node_sizes, node_color = big_node_colours, edgelist = [], with_labels = False) 

plt.show()

nx.draw_networkx(G_small_clonotypes, pos = pos_force_trimmed, node_size = 20, node_color = "lightgray", edgelist = [], with_labels = False)
nx.draw_networkx(G_big_clonotypes, pos = pos_force_trimmed, node_size = big_node_sizes, node_color = big_node_colours, edgelist = [], with_labels = False) 

plt.show()


## NORMALISED MEAN SQUARED ERROR

dmat_ddl = pos_to_dmat(pos_ddl)
# We have to compare ddl_dmat to dmat_full because the other ones were calculated after filtering steps
dmat_full = get_sequence_distance_matrix(list(vdj.metadata.junction_VJ)) 

dmat_umap = pos_to_dmat(pos_umap)
dmat_mds = pos_to_dmat(pos_mds)
dmat_force = pos_to_dmat(pos_force)
dmat_force_trimmed = pos_to_dmat(pos_force_trimmed)

mse_ddl = normalised_mse(dmat_ddl, dmat_full)
mse_umap = normalised_mse(dmat_umap, dmat_medoids)
mse_mds = normalised_mse(dmat_mds, dmat_medoids)
mse_force = normalised_mse(dmat_force, dmat_medoids)
mse_force_trimmed = normalised_mse(dmat_force_trimmed, dmat_medoids)