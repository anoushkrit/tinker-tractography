# %%
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_fnames
from dipy.viz import window, actor, colormap

# %%
fname = get_fnames('fornix')

# %%
print(fname)

# %%
fornix = load_tractogram(fname, 'same', bbox_valid_check=False)
streamlines = fornix.streamlines

# %%
len(streamlines) # 300 streamlines in fornix
# stream_numpy = np.ndarray(streamlines)
len(streamlines[0]) # No of points per streamlines (points per streamline vary a lot)

# %%
qb = QuickBundles(threshold=10.)
clusters = qb.cluster(streamlines)

# %%
print("Nb. clusters:", len(clusters))
print("Cluster sizes:", map(len, clusters))
print("Small clusters:", clusters < 10)
print("Streamlines indices of the first cluster:\n", clusters[0].indices)
print("Centroid of the last cluster:\n", clusters[-1].centroid)
# TODO: why are there 12 points to one centroid of a cluster

# this centroid, is 12 points downsampled from original streamline which depicts the 
# overall bundle representation.

#%% 
print(len(clusters[3]))
191 + 61 + 47 + 1
# %%
# Enables/disables interactive visualization
interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, window.colors.white))
window.record(scene, out_path='fornix_initial.png', size=(1000, 1000))
if interactive:
    window.show(scene)

# %%
colormap = colormap.create_colormap(np.arange(len(clusters)))

scene.clear()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, window.colors.white, opacity=0.05))
scene.add(actor.streamtube(clusters.centroids, colormap, linewidth=0.4))
window.record(scene, out_path='fornix_centroids.png', size=(600, 600))
if interactive:
    window.show(scene)

scene.clear()

# %%
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap):
    colormap_full[cluster.indices] = color

scene.clear()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, colormap_full))
window.record(scene, out_path='fornix_clusters.png', size=(600, 600))
if interactive:
    window.show(scene)

# %%



