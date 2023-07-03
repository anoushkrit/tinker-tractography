#%%
# Importing all the libraries
import nibabel as nib
import os
import numpy as np 
from torch.nn import functional as F
from dipy.io.streamline import load_tractogram, save_tractogram, load_trk

from dipy.segment.clustering import QuickBundles

from dipy.data import get_fnames
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch.utils.data import Dataset, DataLoader, random_split
#%%
def resolve_proxy():
    import os
    os.environ['HTTP_PROXY']="http://10.8.0.1:8080"
    os.environ['HTTPS_PROXY']="http://10.8.0.1:8080"
    os.environ['http_proxy'] = "http://10.8.0.1:8080" 
    os.environ['https_proxy'] = "http://10.8.0.1:8080" 
resolve_proxy()

def positional_encoding(seq_len, embed_dim):
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    encoding = torch.zeros((seq_len, embed_dim))
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding

class PointClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, num_heads):
        super(PointClassifier, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, pos_encoding):
        x = x.to(device) + pos_encoding.to(device)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Pooling over the sequence dimension
        x = self.fc(x)
        x = F.relu(x)
        x = self.output(x)
        return x
    
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            # pos_encoding = pos_encoding.to(device)
            output = model(data, pos_encoding)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data, pos_encoding)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

#%%
#TODO: Add all the fiber streamlines from the HCP with 72 classes

hcp842_fname = get_fnames('bundle_atlas_hcp842')
wbt_fname = get_fnames('target_tractogram_hcp')
fornix_fname = get_fnames('fornix')
fornix = load_tractogram(fornix_fname, 'same', bbox_valid_check=False)
streamlines = fornix.streamlines

# Attach labels to the fornix streamlines
qb = QuickBundles(threshold=10.)
clusters = qb.cluster(streamlines)
#%% 
# Cluster Labels to the sub-streamlines
# 4 labels have been attached to 300 streamlines and there exists class imbalance
# here 0: cluster 0, 1: cluster 1, and so on 

streamline_label = []
for i in range(len(streamlines) + 1):
    for c in np.arange(0,len(clusters),1): 
        if i in clusters[c].indices:
            streamline_label.append(c)
        else:
            continue
#%%
# list of all the fibers in different clusters
c0 = [0, 7, 8, 10, 11, 12, 13, 14, 15, 18, 26, 30, 33, 35, 41, 65, 66, 85, 100, 101, 105, 115, 116, 119, 122, 123, 124, 125, 126, 128, 129, 135, 139, 142, 143, 144, 148, 151, 159, 167, 175, 180, 181, 185, 200, 208, 210, 224, 237, 246, 249, 251, 256, 267, 270, 280, 284, 293, 296, 297, 299]
c1 = [1, 2, 3, 4, 5, 6, 9, 16, 17, 19, 20, 21, 22, 23, 24, 27, 28, 31, 32, 36, 37, 38, 40, 43, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 86, 87, 89, 90, 91, 92, 94, 96, 97, 99, 103, 104, 106, 107, 109, 110, 111, 112, 117, 120, 121, 127, 130, 132, 134, 136, 138, 140, 145, 146, 147, 149, 150, 152, 153, 155, 156, 157, 158, 160, 161, 163, 164, 165, 166, 168, 169, 170, 171, 173, 177, 178, 179, 182, 184, 186, 187, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 201, 202, 203, 204, 207, 209, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 225, 228, 230, 231, 233, 234, 236, 238, 239, 240, 241, 242, 244, 247, 248, 250, 252, 253, 255, 257, 260, 261, 262, 263, 264, 265, 266, 268, 269, 271, 274, 275, 276, 277, 278, 279, 281, 282, 285, 286, 287, 288, 289, 291, 292, 294, 295, 298]
c2 = [25, 29, 34, 39, 42, 45, 46, 71, 77, 83, 88, 93, 95, 98, 102, 108, 113, 114, 118, 131, 133, 137, 141, 154, 162, 172, 174, 176, 183, 188, 197, 205, 206, 211, 226, 227, 229, 232, 235, 243, 245, 254, 258, 259, 272, 273, 283]
c3 = [290]
ax = plt.axes(projection='3d')
def pst(streamlines, num, color):
    for n in num:
        ax.plot3D(streamlines[n].T[0], streamlines[n].T[1], streamlines[n].T[2], color)
pst(streamlines, c0, 'red')
pst(streamlines, c1, 'blue')
pst(streamlines, c2, 'green')
pst(streamlines, c3, 'black')

#%%
get_tensor = lambda x: [torch.from_numpy(x[i]) for i in range(len(x))]
np_streamlines = get_tensor(streamlines)
# find max length of the padding 
all_lengths = [len(s) for s in streamlines]
max(all_lengths) # 91
len_stream = 100
np_streamlines[0] = F.pad(np_streamlines[0], (0,0,0, len_stream - np_streamlines[0].shape[0]), "constant", 0)
stream_tensor = pad_sequence(np_streamlines,batch_first=True, padding_value=0)

# padding the first streamline to the desired/maximum length of the streamline
# max = 91, we will keep it to be 100

pad_array = torch.zeros(100, 1)
# Concatenate the arrays along the last dimension
streamlines_tensr = torch.zeros(300,100,4)

for i in range(len(stream_tensor)):
    streamlines_tensr[i] = torch.cat((stream_tensor[i], pad_array), dim=1)
    # streamlines_tensr[i] = torch.cat((stream_tensor[i], pad_array), dim=1)


# TODO: for adding euclidean distance feature
# torch.norm(stream_tensor[0] - stream_tensor[1], dim=1, p =2).unsqueeze(1).shape
# torch.norm(stream_tensor[0][0] - stream_tensor[0][1], dim=1, p =2).unsqueeze(1)

# positional encoding is set to 4, I think the minimum sinusoidal positional encoding is 4

pos_encoding = positional_encoding(100, embed_dim = 4)
#%%

class TractogramsDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
 
    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target
    
dataset = TractogramsDataset(streamlines_tensr, streamline_label)
train_set , val_set, test_set = random_split(dataset, [0.7, 0.2, 0.1])
#%%
batch_size = 4
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=True)
#%%
input_dim = 4
hidden_dim = 64
num_classes = 4
num_layers = 8
num_heads = 4
num_epochs = 100
model = PointClassifier(input_dim, hidden_dim, num_classes, num_layers, num_heads)


optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999))
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to('cuda')


#%%
for epoch in range(num_epochs):
    train_loss = 0.0
    for data, labels in train_loader:
        # labels = labels.type(torch.LongTensor)
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data.to(device), pos_encoding.to(device))  # Include positional encoding
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Print the average loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    # Perform evaluation on validation dataset
    val_accuracy = evaluate(model.to(device), val_loader, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}")

# Step 9: Test the model
test_accuracy = evaluate(model.to(device), test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f}")

