

import warnings

import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm.notebook import tqdm
import numpy as np
import transformers

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
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, MulticlassF1Score, MultilabelConfusionMatrix
from sklearn import metrics

print(torch.cuda.current_device())
torch.cuda.empty_cache()

### read a csv instead ###

task_data = pd.read_json('aug_task_data2b.json')
one_hot = LabelBinarizer()
task_data['techniques_encoded'] = one_hot.fit_transform(task_data['label'])
num_classes = len(task_data['techniques_encoded'].unique())
print(num_classes, flush=True)
#### read a csv instead ####

path = "image_path"
images = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
images_df = pd.DataFrame(images, columns=['filepath'])
images_df['image'] = images_df['filepath'].str.split('/').str[-1]

task_data = pd.merge(images_df, task_data,  left_on='image', right_on='image', how='left')

task_data = task_data[['id', 'image', 'text', 'label', 'techniques_encoded', 'filepath', 'language']]
task_data.dropna(inplace=True)

train, test = train_test_split(task_data, test_size=0.30, shuffle=True, stratify=task_data['language'])

images = [str(i) for i in train['filepath'].values]
texts = [str(i) for i in train['text'].astype(str).values.tolist()]
labels = train['techniques_encoded'].values.tolist()

images_val = [str(i) for i in test['filepath'].values]
texts_val = [str(i) for i in test['text'].astype(str).values.tolist()]
labels_val = test['techniques_encoded'].values.tolist()

model_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

class VisionTextDataset(torch.utils.data.Dataset):
    def __init__(self, img, txt, lbs, tokenizer_xlm, n_classes, transform):
        self.image = img
        self.text = txt
        self.labels = lbs
        self.tokenizer_xlm = tokenizer_xlm
        self.n_classes = n_classes
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
                  'image': image,
                  "targets": torch.tensor(self.labels[idx], dtype=torch.float)}
        sample = {k: v.squeeze() for k, v in sample.items()}

        return sample

tokenizer_xlm = AutoTokenizer.from_pretrained("xlm-roberta-large", do_lower_case=True, use_fast=False)
train_dataset = VisionTextDataset(img=images, txt=texts, lbs=labels,
                                  tokenizer_xlm=tokenizer_xlm, n_classes=num_classes, transform=model_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(v)
    print(k, v.size(), v.dtype)

val_dataset = VisionTextDataset(img=images_val, txt=texts_val, lbs=labels_val,
                                  tokenizer_xlm=tokenizer_xlm, n_classes=num_classes, transform=model_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)


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


def train_epoch(ensemble_model, train_dataloader):
    ensemble_model.to(device)
    train_loss = 0

    train_outputs = []
    train_targets = []

    ensemble_model.train()

    for batch in train_dataloader:
        # get the inputs;
        batch = {k: v.to(device) for k, v in batch.items()}

        # forward + backward + optimize
        outputs = ensemble_model(text_features=batch['input_ids'],
                                 attention_mask=batch['attention_mask'],
                                 token_type_ids=batch['token_type_ids'],
                                 image_features=batch['image'])

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
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = ensemble_model(text_features=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     token_type_ids=batch['token_type_ids'],
                                     image_features=batch['image'])

            targets = batch['targets'].unsqueeze(1)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            loss = criterion(outputs, targets)
    pred = np.array(fin_outputs) >= 0.5

    accuracy = metrics.accuracy_score(fin_targets, pred)
    f1_score_micro = metrics.f1_score(fin_targets, pred, average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, pred, average='macro')

    return accuracy, f1_score_micro, f1_score_macro, loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()
learning_rate = 5e-6
optimizer = optim.AdamW(ensemble_model.parameters(), lr=learning_rate, weight_decay=1e-5)

metric = BinaryAccuracy()
f1 = BinaryF1Score()

num_epochs = 40

best_loss = 0

for epochs in range(num_epochs):
    accuracy, f1_score_micro, f1_score_macro, loss = train_epoch(ensemble_model, train_dataloader)
    print(f"\n Epoch:{epochs + 1} / {num_epochs}, Training Accuracy:{accuracy:.5f}, Training F1 Micro: {f1_score_micro:.5f},Training F1 Macro:{f1_score_macro:.5f}, Training Loss: {loss:.5f}", flush=True)

    val_accuracy, f1_micro_val, f1_macro_val, loss = val_epoch(ensemble_model, val_dataloader)
    print(f"\n Epoch:{epochs + 1} / {num_epochs}, Val accuracy:{val_accuracy:.5f}, Val F1 Micro: {f1_micro_val:.5f}, Val F1 Macro:{f1_macro_val:.5f}, Validation Loss: {loss:.5f}", flush=True)

    if f1_macro_val < best_loss:
        print('Saving Model :)')
        torch.save(ensemble_model, f'2bensemble_model_improved_{epochs}e.pth')
        torch.save(ensemble_model.state_dict(),f'2bensemble_model_improved_{epochs}e_weights.pth')
        best_loss = f1_macro_val
    else:
        best_loss = best_loss
