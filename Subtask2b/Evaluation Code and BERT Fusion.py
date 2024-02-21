import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import transformers
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel, RemBertModel, RemBertConfig, BertForSequenceClassification
import ast
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import pandas as pd
import os
from sklearn import metrics
from PIL import Image

import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, MulticlassF1Score, MultilabelConfusionMatrix
import torchvision.transforms as transforms
import json
from networkx import DiGraph, relabel_nodes, all_pairs_shortest_path_length
from sklearn_hierarchical_classification.constants import ROOT

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_data = pd.read_json(r'task_data', dtype=False)
path = "Ximage_path"

images = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
images_df = pd.DataFrame(images, columns=['filepath'])
images_df['image'] = images_df['filepath'].str.split('\\').str[-1]
task_data = pd.merge(images_df, task_data,  left_on='image', right_on='image', how='right')

images = [str(i) for i in task_data['filepath'].values]
texts = [str(i) for i in task_data['text'].astype(str).values.tolist()]
ids = [str(i) for i in task_data['id'].astype(str).values.tolist()]

#reuse models from training code - here, we use the improved, post-evaluation model

class XLM(torch.nn.Module):
    def __init__(self):
        super(XLM, self).__init__()
        config = AutoConfig.from_pretrained('xlm-roberta-large')
        self.config = config
        self.l1 = transformers.AutoModel.from_pretrained('xlm-roberta-large', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)

    def forward(self, ids, attention_mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        return output_2

class EnsembleModel(nn.Module):
    def __init__(self, image_model, xlm_model):
        super(EnsembleModel, self).__init__()
        self.image_model = image_model
        self.xlm_model = xlm_model
        self.cls = torch.nn.Linear(self.xlm_model.config.hidden_size + 1000, 1)

    def forward(self, image_features, text_features, attention_mask, token_type_ids):
        img_feat = self.image_model(image_features)
        txt_feat_xlm = self.xlm_model(text_features, attention_mask=attention_mask, token_type_ids=token_type_ids)

        combined_feat = torch.cat((txt_feat_xlm, img_feat), dim=1)
        return self.cls(combined_feat)

xlm_model = XLM()
vision_model = torchvision.models.vgg19(pretrained=True)
ensemble_model = EnsembleModel(vision_model, xlm_model)

for param in vision_model.parameters():
    param.requires_grad = False
    
model_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

class VisionTextDataset_Eval(torch.utils.data.Dataset):
    def __init__(self, img, txt, tokenizer_xlm, transform):
        self.image = img
        self.text = txt
        self.tokenizer_xlm = tokenizer_xlm
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        text = self.text[idx]
        image = self.image[idx]

        text_encoded = self.tokenizer_xlm.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=True,
            padding='max_length',
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        image = Image.open(image).convert('RGB')
        image = self.transform(image)

        sample = {'input_ids': text_encoded['input_ids'],
                  'attention_mask': text_encoded['attention_mask'],
                  'token_type_ids': text_encoded['token_type_ids'],
                  'image': image}
        sample = {k: v.squeeze() for k, v in sample.items()}

        return sample
        
]
tokenizer_xlm = AutoTokenizer.from_pretrained("xlm-roberta-large", do_lower_case=True, use_fast=False)
eval_dataset = VisionTextDataset_Eval(img=images, txt=texts,
                                  tokenizer_xlm=tokenizer_xlm, transform=model_transforms)

eval_dataloader = DataLoader(eval_dataset, shuffle=False)
batch = next(iter(eval_dataloader))
for k, v in batch.items():
    print(v)
    print(k, v.size(), v.dtype)
    
    
def eval_2b(ensemble_model, path, data, submit_type): #use ensemble submit type for late fusion
    actuals, predictions, lbl_id = [], [], []

    ensemble_model.load_state_dict(torch.load(path))
    ensemble_model.to(device)
    ensemble_model.eval()
    with torch.no_grad():
        for batch in tqdm(data):
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = ensemble_model(text_features=batch['input_ids'],
                                 attention_mask=batch['attention_mask'],
                                 token_type_ids=batch['token_type_ids'],
                                 image_features=batch['image'])
            task_id = 0
            logits = outputs
            logits = torch.tensor(logits.detach().cpu())

            predictions.extend(torch.sigmoid(logits).tolist())

            if float(torch.sigmoid(logits).cpu().numpy()[0]) >= 0.5:
                pred = 'propagandistic'
                probability = float(torch.sigmoid(logits).cpu().numpy()[0])
            else:
                pred = 'non_propagandistic'
                probability = float(torch.sigmoid(logits).cpu().numpy()[0])
            
            prediction_output.append({'id': str(int(task_id)),
                                           'predicted_label': pred,
                                     'predicted_label_probability': probability})
            submit_output.append({'id': str(int(task_id)),
                                           'label': pred})
    if submit_type == 'ensemble':
        return prediction_output
    else:
        return submit_output
        
        
prediction_output = []
submit_output = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = 'path to model'

pred = eval_2b(ensemble_model, path, eval_dataloader, 'ensemble')

output = pd.DataFrame(pred)
output['id'] = ids
output = output.astype(str)


### late fusion with BERT(ex) ####

test = pd.read_json('subtask2b_prediction_probabilities')
test_bert = pd.read_json('subtask2b_prediction_probabilities_from_bertex')

predictions = test.merge(test_bert, right_on='id', left_on='id', suffixes=['_bert', '_vgg16'])
predictions.drop_duplicates(inplace=True)
predictions['weighted_output'] = (predictions['predicted_label_probability_bert'].astype(float) + predictions['predicted_label_probability_vgg16'].astype(float))/2 # can be modified to weighted
predictions['label'] = predictions['weighted_output'].map(lambda x: 'propagandistic' if x >= 0.5 else 'non_propagandistic')
predictions = predictions[['id', 'label']].astype(str)

predictions.to_json('bert_ensemble_subtask2b.json', orient='records')