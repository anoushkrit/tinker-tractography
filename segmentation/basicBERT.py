#%% 
# Import and proxy
import nibabel as nib
import os
import numpy as np 
from dipy.io.streamline import load_tractogram, save_tractogram, load_trk
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.utils import (create_nifti_header, get_reference_info,
                           is_header_compatible)
from dipy.tracking.streamline import select_random_set_of_streamlines
from dipy.tracking.utils import density_map
from dipy.data.fetcher import (fetch_file_formats,
                               get_file_formats)
from dipy.segment.clustering import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_fnames
from dipy.viz import window, actor, colormap
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def resolve_proxy():
    import os
    os.environ['HTTP_PROXY']="http://10.8.0.1:8080"
    os.environ['HTTPS_PROXY']="http://10.8.0.1:8080"
    os.environ['http_proxy'] = "http://10.8.0.1:8080" 
    os.environ['https_proxy'] = "http://10.8.0.1:8080" 
resolve_proxy()
#%% 
# Read Fornix data
fname = get_fnames('fornix')
fornix = load_tractogram(fname, 'same', bbox_valid_check=False)
streamlines = fornix.streamlines

qb = QuickBundles(threshold=10.)
clusters = qb.cluster(streamlines) # creating clusters inside fornix using QuickBundles

#%% 
# Cluster Labels to the sub-streamlines
# 4 labels have been attached to 300 streamlines and there exists class imbalance
# here 0: cluster 0, 1: cluster 1, and so on 
streamline_label = []
for i in range(len(fornix.streamlines)):
    for c in np.arange(0,3,1):
        if i in clusters[c].indices:
            streamline_label.append(c)
        else:
            continue

# %% [markdown]
## BERT encoding
#%%
# TODO: #4 load completely merged .trk and then read the headers
# TODO: Pass these headers for each streamline as label for the streamline
# TODO: #6 implement BERT and tokenize the 3 D point data as it is to train BERT from scratch

# BERT encoding takes a fixed length of 512 tokens as input, so either [PAD] or truncate the input accordingly
# embedding vector size of 768 dims
#%%
# finding max length of streamline
max_len_streamlines = 0
for i in streamlines:
    if len(i) > max_len_streamlines: 
        max_len_streamlines = len(i)

np.array(streamlines)

#%%
p = nn.ConstantPad1d((0,21), 0)
p(streamlines[0]) 

# TODO: #7 resolve this padding issue
# pad_sequence(streamlines, padding_value = 0)
#%%
# PointEmbedding module
class PointEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PointEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # takes in a padded streamline of white matter
        # with a fixed length into the 
        self.relu = nn.ReLU() # encoded output 
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

#%%
# Attention module
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__() # self initialise the object of the class
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_weights = self.fc1(x) # attention is calculated by passing x to a hidden dim linear neural network
        attn_weights = torch.softmax(attn_weights, dim=1) # attention weights are normalised to probabilities
        x = torch.sum(attn_weights * x, dim=1) # now the new x is calculated by computing attention over x
        return x

#%%
# Define the PointWMA module
class PointWMA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(PointWMA, self).__init__()
        self.num_layers = num_layers
        self.embedding = PointEmbedding(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8) # attention heads = 8
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers) # how many encoder layers 
        # to be added in the encoder to make it a complete encoder
        self.attention = Attention(hidden_dim)
        # TODO: find the dimension of the output computed here
        self.fc1 = nn.Linear(hidden_dim, num_classes)
        # (hidden_dim: input_dim, num_classes: output_dim)

    def forward(self, x):
        x = pad_sequence(x, batch_first=True, padding_value=0)  # pad the input sequences
        x = self.embedding(x) # extract embedding which is nothing but the pointEmbedding module
        x = x.permute(1, 0, 2)  # transpose for transformer input
        x = self.transformer_encoder(x) #Transformer Encoder as defined in the encoder
        x = x.permute(1, 0, 2)  # transpose back
        x = self.attention(x) 
        x = self.fc1(x)
        return x
    
#%%
# Padding sequence and downsampling
#%%
# Define the point cloud dataset
class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, point_clouds, labels):
        # self.point_clouds = pad_stream(point_clouds) # padding streamlines to a fixed length
        self.point_clouds = point_clouds
        self.labels = labels

    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        return self.point_clouds[idx], self.labels[idx]
#%%
# Define the point clouds and labels
num_classes = 4
input_dim = 3 # adding more dimensions later which will give extra feature per point
hidden_dim = 8
num_layers = 2
#%%
# Create the point cloud dataset and dataloader
dataset = PointCloudDataset(fornix.streamlines, streamline_label)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

model = PointWMA(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model to classify the point clouds into different classes
for epoch in range(10):
    running_loss = 0.0
    for i, (pc_batch, label_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(pc_batch)
        loss = criterion(outputs, label_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss/10:.4f}")
            running_loss = 0.0

print("Training finished.")

#%%
hcp_105 = "/media/ang/Data/dMRI_data/105HCP/Fiber_Tracts/599469/tracts/AF_left.trk"
# trk_599469 = nib.streamlines.load(hcp_105).streamlines
trk_599469 = load_tractogram(hcp_105, 'same', bbox_valid_check=False)
print(trk_599469._dimensions)
print(np.array(trk_599469.streamlines).shape)
