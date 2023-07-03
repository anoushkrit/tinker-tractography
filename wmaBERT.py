#%%
# conda activate pointWMA
from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
import numpy as np 
import nibabel as nib
from dipy.segment.clustering import QuickBundles


#%%
def resolve_proxy():
    import os
    os.environ['HTTP_PROXY']="http://10.8.0.1:8080"
    os.environ['HTTPS_PROXY']="http://10.8.0.1:8080"
    os.environ['http_proxy'] = "http://10.8.0.1:8080" 
    os.environ['https_proxy'] = "http://10.8.0.1:8080" 
resolve_proxy()
#%%

hcp842_fname = get_fnames('bundle_atlas_hcp842')
wbt_fname = get_fnames('target_tractogram_hcp')
fornix_fname = get_fnames('fornix')
fornix = load_tractogram(fornix_fname, 'same', bbox_valid_check=False)
fornix_streamlines = fornix.streamlines
#%% Attach labels to the fornix streamlines
qb = QuickBundles(threshold=10.)
clusters = qb.cluster(fornix_streamlines)



# %%
from datasets.dataset_dict import DatasetDict
from datasets import Dataset

d = {'train':Dataset.from_dict({'label':y_train,'text':x_train}),
     'val':Dataset.from_dict({'label':y_val,'text':x_val}),
     'test':Dataset.from_dict({'label':y_test,'text':x_test})
     }

DatasetDict(d)

# %%


# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# %%
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
# %%
tokenized_imdb = imdb.map(preprocess_function, batched=True)
# %% This is where padding and truncating of the sequences occur
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# %%
import evaluate

accuracy = evaluate.load("accuracy")
# %%
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
# %%
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# %%

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)
# %%
training_args = TrainingArguments(
    output_dir="text_classification_transformer",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False, # for pushing the model to Hugging Face Hub
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# %%

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
# %%
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier(text)
# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
inputs = tokenizer(text, return_tensors="pt")
# %%

from transformers import AutoModelForSequenceClassification
import torch
model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
with torch.no_grad():
    logits = model(**inputs).logits
# %%

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
# %%
