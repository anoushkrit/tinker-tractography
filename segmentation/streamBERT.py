#%% 
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

def resolve_proxy():
    import os
    os.environ['HTTP_PROXY']="http://10.8.0.1:8080"
    os.environ['HTTPS_PROXY']="http://10.8.0.1:8080"
    os.environ['http_proxy'] = "http://10.8.0.1:8080" 
    os.environ['https_proxy'] = "http://10.8.0.1:8080" 
resolve_proxy()
#%%
hcp842_fname = get_fnames('bundle_atlas_hcp842')
#%%
print(hcp842_fname)
# hcp842 dataset with 80 bundles and their labels.
# '/home/ang/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk'
# '/home/ang/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/*.trk'

#%%

wbt_fname = get_fnames('target_tractogram_hcp')
#%%
# hcp842 = load_tractogram(hcp842_fname, 'same', bbox_valid_check=False) 
# reference nii needs to be put in this to load as a tractogram
#%%

fname = get_fnames('fornix')
fornix = load_tractogram(fname, 'same', bbox_valid_check=False)
streamlines = fornix.streamlines

#%%
# .trk reading into numpy array 

hcp_105 = "/media/ang/Data/dMRI_data/105HCP/Fiber_Tracts/599469/tracts/AF_left.trk"
# trk_599469 = nib.streamlines.load(hcp_105).streamlines
trk_599469 = load_tractogram(hcp_105, 'same', bbox_valid_check=False)
# %% [markdown]
## BERT encoding
# TODO: load completely merged .trk and then read the headers
# TODO: Pass these headers for each streamline as label for the streamline
# TODO: implement BERT and tokenize the 3 D point data as it is to train BERT from scratch
# BERT encoding takes a fixed length of 512 tokens as input, so either [PAD] or truncate the input accordingly
# embedding vector size of 768 dims

#%%
import torch
import numpy as np
from transformers import BertTokenizer
#%%
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

#%%

from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5) # 5 classes as output
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
#%%
np.asarray(trk_599469.streamlines)[0].shape
streamlines

# %%
