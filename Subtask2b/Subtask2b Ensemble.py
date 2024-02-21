import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm.notebook import tqdm
import numpy as np
import transformers
from argparse import Namespace

from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    BertModel)

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, MulticlassF1Score, MultilabelConfusionMatrix
from sklearn import metrics

# external if you want to include

path_w = r'path.json'
with open(path_w) as f:
    web_ents = json.load(f)

def explode_frame(json_file, col):
    df = pd.json_normalize(json_file)
    df.set_index('Image ID', inplace=True)
    return df['Response.' + col].explode().pipe(lambda x: pd.json_normalize(x).set_index(x.index))

web_ents = explode_frame(web_ents, 'webEntities')
web_ents.dropna(subset='description')

path = r'images'
images = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
images_df = pd.DataFrame(images, columns=['filepath'])
images_df['image'] = images_df['filepath'].str.split('\\').str[-1]

df = pd.read_json('subtask2b_data.json')
df = pd.merge(df, images_df, on='image')
df.fillna(' ', inplace=True)

le = LabelEncoder()
df['encoded_labels'] = le.fit_transform(df['label']).tolist()
df['temp_id'] = df['id'].astype(str) + df['language']

# merge external data

web_entst = web_ents.merge(df, left_on=web_ents.index, right_on='image')
df = web_entst.groupby('temp_id').agg(list).reset_index()
df['description'] = df['description'].apply(lambda x: list(set(x))).astype(str)
df['image'] = df['image'].str[0]
df['encoded_labels'] = df['encoded_labels'].str[0]
df['filepath'] = df['filepath'].str[0]
df['label'] = df['label'].str[0]
df['text'] = df['text'].str[0]
df['language'] = df['language'].str[0]
df['id'] = df['id'].str[0]

train, test = train_test_split(df, test_size=0.3, shuffle=True, stratify=df[['label', 'language']])

images = [str(i) for i in train['filepath'].values]
texts = [str(i) for i in train['text'].astype(str).values.tolist()]
description = [str(i) for i in train['description'].astype(str).values.tolist()]
labels = df['encoded_labels'].values

images_val = [str(i) for i in test['filepath'].values]
texts_val = [str(i) for i in test['description'].astype(str).values.tolist()]
description_val = [str(i) for i in test['description'].astype(str).values.tolist()]
labels_val = test['encoded_labels'].values

# transformations for the raw images
# change size here for different model

model_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])


class VisionTextDataset(torch.utils.data.Dataset):
    def __init__(self, img, txt, description, lbs, tokenizer_ml, tokenizer_bert, n_classes, transform, id):
        self.image = img
        self.text = txt
        self.labels = lbs
        self.tokenizer_ml = tokenizer_ml
        self.tokenizer_bert = tokenizer_bert
        self.n_classes = n_classes
        self.transforms = transform
        self.description = description
        self.id = id

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        text = self.text[idx]
        image = self.image[idx]
        description = self.description[idx]
        id = self.id[idx]

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

        ex_text_encoded = self.tokenizer_bert.encode_plus(
            description,
            add_special_tokens=True,
            return_token_type_ids=True,
            padding='max_length',
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        image = Image.open(image).convert('RGB')
        image = self.transforms(image)

        sample = {'input_ids': text_encoded['input_ids'],
                  'attention_mask': text_encoded['attention_mask'],
                  'token_type_ids': text_encoded['token_type_ids'],
                  'image': image,
                  "targets": torch.tensor(self.labels[idx], dtype=torch.float),
                  'ids': id,
                  # drop the ex_ids if not using BERT
                  'input_ids_ex': ex_text_encoded['input_ids'],
                  'attention_mask_ex': ex_text_encoded['attention_mask'],
                  'token_type_ids_ex': ex_text_encoded['token_type_ids']
                  }
        sample = {k: v.squeeze() for k, v in sample.items()}

        return sample

tokenizer_ml = AutoTokenizer.from_pretrained("xlm-roberta-base", do_lower_case=True, use_fast=False)
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, use_fast=False) #bert model can be removed
train_dataset = VisionTextDataset(img=images, txt=texts, description = description, lbs=labels,
                                  tokenizer_ml=tokenizer_ml,
                                  tokenizer_bert=tokenizer_bert, n_classes=num_classes, transform=model_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
batch = next(iter(train_dataloader))

val_dataset = VisionTextDataset(img=images_val, txt=texts_val, description=description_val,
                                lbs=labels_val,
                                tokenizer_ml=tokenizer_ml, tokenizer_bert=tokenizer_bert, n_classes=num_classes, transform=model_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)


class BERTBaseML(torch.nn.Module):
    def __init__(self):
        super(BERTBaseML, self).__init__()
        config = AutoConfig.from_pretrained('xlm-roberta-base')
        self.config = config
        self.l1 = transformers.AutoModel.from_pretrained('xlm-roberta-base', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.cls = nn.Linear(768, 1)

    def forward(self, ids, attention_mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.cls(output_2)
        return output_3

class BERTBase(torch.nn.Module):
    def __init__(self):
        super(BERTBase, self).__init__()
        config = AutoConfig.from_pretrained('bert-base-uncased')
        self.config = config
        self.l1 = transformers.AutoModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.cls = nn.Linear(768, 1)

    def forward(self, ids_ex, attention_mask_ex, token_type_ids_ex):
        _, output_1 = self.l1(ids_ex, attention_mask=attention_mask_ex, token_type_ids=token_type_ids_ex)
        output_2 = self.l2(output_1)
        output_3 = self.cls(output_2)
        return output_3

class EnsembleModelBERT(nn.Module):
    def __init__(self, image_model, text_model, bert_model, num_classes):
        super(EnsembleModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.bert_model = bert_model
        self.cnn_fc = nn.Linear(1000,1)
        self.fc = nn.Linear(3, 1)

    def forward(self, image_features, text_features, attention_mask, token_type_ids, description, attention_mask_ex, token_type_ids_ex):
        img_feat = self.image_model(image_features)
        txt_feat_bert = self.text_model(text_features, attention_mask=attention_mask, token_type_ids=token_type_ids)
        txt_feat_external = self.bert_model(description, attention_mask_ex=attention_mask_ex, token_type_ids_ex=token_type_ids_ex)
        img_class = self.cnn_fc(img_feat)

        combined_feat = torch.cat((img_class, txt_feat_bert,txt_feat_external), dim=1)
        return self.fc(combined_feat)

class EnsembleModelBERT(nn.Module):
    def __init__(self, image_model, text_model, num_classes):
        super(EnsembleModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.cnn_fc = nn.Linear(1000,1)
        self.fc = nn.Linear(3, 1)

    def forward(self, image_features, text_features, attention_mask, token_type_ids):
        img_feat = self.image_model(image_features)
        txt_feat_bert = self.text_model(text_features, attention_mask=attention_mask, token_type_ids=token_type_ids)
        img_class = self.cnn_fc(img_feat)

        combined_feat = torch.cat((img_class, txt_feat_bert), dim=1)
        return self.fc(combined_feat)

text_model = BERTBaseML()
bert_model = BERTBase()
vision_model = torchvision.models.vgg19(pretrained=True)

ensemble_model = EnsembleModelBERT(vision_model, text_model, bert_model, num_classes)
#ensemble_model = EnsembleModel(vision_model, text_model, num_classes) ## use if not including BERT(ex)

for param in vision_model.parameters():
    param.requires_grad = False


def train_epoch(ensemble_model, train_dataloader):
    ensemble_model.to(device)
    train_loss = 0

    train_outputs = []
    train_targets = []

    ensemble_model.train()

    for batch in tqdm(train_dataloader):
        # get the inputs;
        batch = {k: v.to(device) for k, v in batch.items()}

        # forward + backward + optimize
        outputs = ensemble_model(text_features=batch['input_ids'],
                                 attention_mask=batch['attention_mask'],
                                 token_type_ids=batch['token_type_ids'],
                                 image_features=batch['image'],
                                 # drop the below if not using BERT(ex)
                                 description=batch['input_ids_ex'],
                                 attention_mask_ex=batch['attention_mask_ex'],
                                 token_type_ids_ex=batch['token_type_ids_ex'])

        targets = batch['targets'].unsqueeze(1)
        train_targets.extend(targets.cpu().detach().numpy().tolist())
        train_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    pred = np.array(train_outputs) >= 0.5

    accuracy = metrics.accuracy_score(train_targets, pred)
    f1_score_micro = metrics.f1_score(train_targets, pred, average='micro')
    f1_score_macro = metrics.f1_score(train_targets, pred, average='macro')

    return accuracy, f1_score_micro, f1_score_macro, loss


def val_epoch(ensemble_model, val_dataloader):
    ensemble_model.to(device)
    fin_outputs = []
    fin_targets = []
    ensemble_model.eval()
    with torch.no_grad(
    for batch in tqdm(val_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
    outputs = ensemble_model(text_features=batch['input_ids'],
                             attention_mask=batch['attention_mask'],
                             token_type_ids=batch['token_type_ids'],
                             image_features=batch['image'],
                             #drop the below if not using BERT(ex)
                             description=batch['input_ids_ex'],
                             attention_mask_ex=batch['attention_mask_ex'],
                             token_type_ids_ex=batch['token_type_ids_ex'])

    targets = batch['targets'].unsqueeze(1)
    fin_targets.extend(targets.cpu().detach().numpy().tolist())
    fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    loss = criterion(outputs, targets)
    pred = np.array(fin_outputs) >= 0.5

    accuracy = metrics.accuracy_score(fin_targets, pred)
    f1_score_micro = metrics.f1_score(fin_targets, pred, average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, pred, average='macro')

    return accuracy, f1_score_micro, f1_score_macro, loss

num_epochs = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()
learning_rate = 1e-5
optimizer = optim.AdamW(ensemble_model.parameters(), lr=learning_rate, weight_decay=1e-5)

metric = BinaryAccuracy()
f1 = BinaryF1Score()

best_f1 = 0

for epochs in range(num_epochs):
    accuracy, f1_score_micro, f1_score_macro, loss = train_epoch(ensemble_model, train_dataloader)
    print(f"\n Epoch:{epochs + 1} / {num_epochs}, Training Accuracy:{accuracy:.5f}, Training F1 Micro: {f1_score_micro:.5f},Training F1 Macro:{f1_score_macro:.5f}, Training Loss: {loss:.5f}")

    val_accuracy, f1_micro_val, f1_macro_val, loss = val_epoch(ensemble_model, val_dataloader)
    print(f"\n Epoch:{epochs + 1} / {num_epochs}, Val accuracy:{val_accuracy:.5f}, Val F1 Micro: {f1_micro_val:.5f}, Val F1 Macro:{f1_macro_val:.5f}, Validation Loss: {loss:.5f}")

    if f1_macro_val > best_f1: #modify to loss or whichever, we typically used loss but also tested F1
        print('Saving Model :)')
        torch.save(ensemble_model, f'ensemble_model_{epochs}e_vgg16_bertex_stack.pth')
        torch.save(ensemble_model.state_dict(),f'ensemble_model_{epochs}e_vgg16_bertex_weights_stack.pth')
        best_f1 = f1_macro_val
    else:
        best_f1 = best_f1