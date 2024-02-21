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

task_data = pd.read_json(r'text_path', dtype=False)
path = "image_path"

images = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
images_df = pd.DataFrame(images, columns=['filepath'])
images_df['image'] = images_df['filepath'].str.split('\\').str[-1]
task_data = pd.merge(images_df, task_data,  left_on='image', right_on='image', how='right')

images = [str(i) for i in task_data['filepath'].values]
texts = [str(i) for i in task_data['text'].astype(str).values.tolist()]
ids = [str(i) for i in task_data['id'].astype(str).values.tolist()]
num_classes = 22

G = DiGraph()
G.add_edge(ROOT, "Logos")
G.add_edge("Logos", "Repetition")
G.add_edge("Logos", "Obfuscation, Intentional vagueness, Confusion")
G.add_edge("Logos", "Reasoning")
G.add_edge("Logos", "Justification")
G.add_edge('Justification', "Slogans")
G.add_edge('Justification', "Bandwagon")
G.add_edge('Justification', "Appeal to authority")
G.add_edge('Justification', "Flag-waving")
G.add_edge('Justification', "Appeal to fear/prejudice")
G.add_edge('Reasoning', "Simplification")
G.add_edge('Simplification', "Causal Oversimplification")
G.add_edge('Simplification', "Black-and-white Fallacy/Dictatorship")
G.add_edge('Simplification', "Thought-terminating cliché")
G.add_edge('Reasoning', "Distraction")
G.add_edge('Distraction', "Misrepresentation of Someone's Position (Straw Man)")
G.add_edge('Distraction', "Presenting Irrelevant Data (Red Herring)")
G.add_edge('Distraction', "Whataboutism")
G.add_edge(ROOT, "Ethos")
G.add_edge('Ethos', "Appeal to authority")
G.add_edge('Ethos', "Glittering generalities (Virtue)")
G.add_edge('Ethos', "Bandwagon")
G.add_edge('Ethos', "Ad Hominem")
G.add_edge('Ethos', "Transfer")
G.add_edge('Ad Hominem', "Doubt")
G.add_edge('Ad Hominem', "Name calling/Labeling")
G.add_edge('Ad Hominem', "Smears")
G.add_edge('Ad Hominem', "Reductio ad hitlerum")
G.add_edge('Ad Hominem', "Whataboutism")
G.add_edge(ROOT, "Pathos")
G.add_edge('Pathos', "Exaggeration/Minimisation")
G.add_edge('Pathos', "Loaded Language")
G.add_edge('Pathos', "Appeal to (Strong) Emotions")
G.add_edge('Pathos', "Appeal to fear/prejudice")
G.add_edge('Pathos', "Flag-waving")
G.add_edge('Pathos', "Transfer")

model_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# you can also copy module files from the Ensemble training models, and reconstruct them here

class mBERT(torch.nn.Module):
    def __init__(self):
        super(mBERT, self).__init__()
        config = AutoConfig.from_pretrained('bert-base-multilingual-uncased')
        self.config = config
        self.l1 = transformers.AutoModel.from_pretrained('bert-base-multilingual-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)

    def forward(self, ids, attention_mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        return output_2

class XLM(torch.nn.Module):
    def __init__(self):
        super(XLM, self).__init__()
        config = AutoConfig.from_pretrained('xlm-roberta-base')
        self.config = config
        self.l1 = transformers.AutoModel.from_pretrained('xlm-roberta-base', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)

    def forward(self, ids_ex, attention_mask_ex, token_type_ids_ex):
        _, output_1 = self.l1(ids_ex, attention_mask=attention_mask_ex, token_type_ids=token_type_ids_ex)
        output_2 = self.l2(output_1)
        return output_2

class EnsembleModel(nn.Module):
    def __init__(self, image_model, bert_model, xlm_model, num_classes):
        super(EnsembleModel, self).__init__()
        self.image_model = image_model
        self.bert_model = bert_model
        self.xlm_model = xlm_model
        self.cls = torch.nn.Linear(self.bert_model.config.hidden_size + self.xlm_model.config.hidden_size + 1000, num_classes)

    def forward(self, image_features, text_features, attention_mask, token_type_ids, description, attention_mask_ex, token_type_ids_ex):
        img_feat = self.image_model(image_features)
        txt_feat_bert = self.bert_model(text_features, attention_mask=attention_mask, token_type_ids=token_type_ids)
        txt_feat_xlm = self.xlm_model(description, attention_mask_ex=attention_mask_ex, token_type_ids_ex=token_type_ids_ex)

        combined_feat = torch.cat((txt_feat_bert,txt_feat_xlm, img_feat), dim=1)
        return self.cls(combined_feat)

bert_model = mBERT()
xlm_model = XLM()
vision_model = torchvision.models.vgg19(pretrained=True)
ensemble_model = EnsembleModel(vision_model, bert_model, xlm_model, num_classes)

for param in vision_model.parameters():
    param.requires_grad = False

# similarily, the same dataset reader can used from the Ensemble code

 class VisionTextDataset(torch.utils.data.Dataset):
    def __init__(self, img, txt, tokenizer_ml, tokenizer_xlm, n_classes, transform):
        self.image = img
        self.text = txt
        self.tokenizer_ml = tokenizer_ml
        self.transform = transform
        self.tokenizer_xlm = tokenizer_xlm
        self.n_classes = n_classes

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        text = self.text[idx]
        image = self.image[idx]

        text_encoded = self.tokenizer_ml.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=True,
            padding='max_length',
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        ex_text_encoded = self.tokenizer_xlm.encode_plus(
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
                  'input_ids_xlm': ex_text_encoded['input_ids'],
                  'attention_mask_xlm': ex_text_encoded['attention_mask'],
                  'token_type_ids_xlm': ex_text_encoded['token_type_ids'],
                  'image': image}
        sample = {k: v.squeeze() for k, v in sample.items()}
        return sample

tokenizer_ml = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased", do_lower_case=True, use_fast=False)
tokenizer_xlm = AutoTokenizer.from_pretrained("xlm-roberta-base", do_lower_case=True, use_fast=False)
eval_dataset = VisionTextDataset(img=images, txt=texts,
                                  tokenizer_ml=tokenizer_ml,
                                  tokenizer_xlm=tokenizer_xlm, n_classes=num_classes, transform=model_transforms)

eval_dataloader = DataLoader(eval_dataset, shuffle=False)
batch = next(iter(eval_dataloader))
for k, v in batch.items():
    print(v)
    print(k, v.size(), v.dtype)

def eval_2a(ensemble_model, path, data, submit_type): # choose submission type: ensemble or submit, use 'ensemble' for late fusion
    actuals, predictions, lbl_id = [], [], []

    ### change this model to the base model above ###

    ### rest remains the same ###

    ensemble_model.load_state_dict(torch.load(path))
    ensemble_model.to(device)
    ensemble_model.eval()
    with torch.no_grad():
        for batch in tqdm(data):
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = ensemble_model(text_features=batch['input_ids'],
                                 attention_mask = batch['attention_mask'],
                                 token_type_ids = batch['token_type_ids'],
                                 description = batch['input_ids_xlm'],
                                 attention_mask_ex = batch['attention_mask_xlm'],
                                 token_type_ids_ex = batch['token_type_ids_xlm'],
                                image_features = batch['image'])

            predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())


        for i in range(len(data)):
            predicted_probabilities = predictions[i]
            predicted_labels = [one_hot.classes_[j] for j in range(len(one_hot.classes_))
                                if predicted_probabilities[j] > 0.5] # normal label probability
            # modification below: first threshold for label, second threshold to move up the hierarchy
            predicted_labels_f1h = [one_hot.classes_[j] if predicted_probabilities[j] > 0.40
                                    else [pred for pred in G.predecessors(one_hot.classes_[j])][-1] for j in range(len(one_hot.classes_)) if predicted_probabilities[j] > 0.35]


            prediction_output.append({'id': str(int(0)),
                                           'predicted_labels': predicted_labels,
                                     'predicted_probabilities': {label: float(prob) for label, prob in zip(one_hot.classes_, predicted_probabilities)}})
            submit_output.append({'id': str(int(0)),
                                           'labels': predicted_labels})

    if submit_type == 'ensemble':
        return prediction_output
    else:
        return submit_output

prediction_output = []
submit_output = []
path = 'weights.pth' # path to model
pred = eval_2a(ensemble_model, path, eval_dataloader, 'submit')

# this step is for when the ID are string or text
output = pd.DataFrame(pred)
output['id'] = ids

output.to_json('name_your_file.json', orient='records')