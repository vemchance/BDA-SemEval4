

import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import transformers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, MulticlassF1Score, MultilabelConfusionMatrix
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)

print(torch.cuda.current_device())

### read a csv instead ###

task_data = pd.read_json('task_data.json')
task_data['techniques_encoded'] = task_data['labels'].apply(lambda y: ['None'] if len(y)==0 else y)
one_hot = MultiLabelBinarizer()
task_data['techniques_encoded'] = one_hot.fit_transform(task_data['techniques_encoded'])[:, 1:].tolist()
num_classes = len(task_data['techniques_encoded'].iloc[0])
print(num_classes, flush=True)
#### read a csv instead ####

train, val = train_test_split(task_data, test_size=0.30, shuffle=True)

train_text = train['text'].values.tolist()
train_labels = train['techniques_encoded'].values.tolist()

val_text = val['text'].values.tolist()
val_labels = val['techniques_encoded'].values.tolist()

class CustomDataset(Dataset):

    def __init__(self, text, labels, tokenizer):
        self.tokenizer = tokenizer
        self.text = text
        self.targets = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        text = self.text[index]

        max_len = self.tokenizer.model_max_length
        if max_len > 1024:
            max_len = 512

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation = True
        )
        ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', return_dict=False)

train_set = CustomDataset(train_text, train_labels, tokenizer)
val_set = CustomDataset(val_text, val_labels, tokenizer)

train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=8, shuffle=False)


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
        outputs = ensemble_model(batch['ids'],
                                 attention_mask=batch['mask'],
                                 token_type_ids=batch['token_type_ids'])

        targets = batch['targets']
        train_targets.extend(targets.cpu().detach().numpy().tolist())
        train_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        loss = criterion(outputs, targets)

        loss.backward()
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
            outputs = ensemble_model(batch['ids'],
                                     attention_mask=batch['mask'],
                                     token_type_ids=batch['token_type_ids'])
            targets = batch['targets']
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            loss = criterion(outputs, targets)
    pred = np.array(fin_outputs) >= 0.5

    accuracy = metrics.accuracy_score(fin_targets, pred)
    f1_score_micro = metrics.f1_score(fin_targets, pred, average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, pred, average='macro')

    return accuracy, f1_score_micro, f1_score_macro, loss


class mBERTBase(torch.nn.Module):
    def __init__(self):
        super(mBERTBase, self).__init__()
        config = AutoConfig.from_pretrained('bert-base-multilingual-uncased')
        self.config = config
        self.l1 = transformers.AutoModel.from_pretrained('bert-base-multilingual-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.cls = torch.nn.Linear(768, num_classes)

    def forward(self, ids, attention_mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.cls(output_2)
        return output_3
        
class mBERTBaseD(torch.nn.Module):
    def __init__(self):
        super(mBERTBaseD, self).__init__()
        config = AutoConfig.from_pretrained('bert-base-multilingual-uncased')
        self.config = config
        self.l1 = transformers.AutoModel.from_pretrained('bert-base-multilingual-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.6)
        self.cls = torch.nn.Linear(768, num_classes)

    def forward(self, ids, attention_mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.cls(output_2)
        return output_3

class XLMRobertaBase(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaBase, self).__init__()
        config = AutoConfig.from_pretrained('xlm-roberta-base')
        self.config = config
        self.l1 = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-large', return_dict=False)
        self.l2 = torch.nn.Dropout(0.4)
        self.cls = torch.nn.Linear(768, num_classes)

    def forward(self, ids, attention_mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.cls(output_2)
        return output_3


class NLPEnsembleDouble(torch.nn.Module):
    def __init__(self, bert_model, xlm_model, num_classes):
        super(NLPEnsembleDouble, self).__init__()

        self.mbert = bert_model
        self.xlm = xlm_model

        self.cls = torch.nn.Linear(self.mbert.config.hidden_size + self.xlm.config.hidden_size, num_classes)

    def forward(self, text_features, attention_mask, token_type_ids):
        txt1 = self.mbert(text_features, attention_mask, token_type_ids)
        txt2 = self.xlm(text_features, attention_mask, token_type_ids)

        combined_feat = torch.cat((txt1, txt2), dim=1)
        return self.cls(combined_feat)


class NLPEnsembleQuad(torch.nn.Module):
    def __init__(self, bert_model1, bert_model2, xlm_model1, num_classes):
        super(NLPEnsembleQuad, self).__init__()

        self.mbert1 = bert_model1
        self.mbert2 = bert_model2
        self.xlm1 = xlm_model1

        self.cls = torch.nn.Linear((num_classes*3),num_classes)

    def forward(self, text_features, attention_mask, token_type_ids):
        txt1 = self.mbert1(text_features, attention_mask, token_type_ids)
        txt2 = self.xlm1(text_features, attention_mask, token_type_ids)
        txt3 = self.mbert2(text_features, attention_mask, token_type_ids)

        combined_feat = torch.cat((txt1, txt2, txt3), dim=1)
        return self.cls(combined_feat)

m1 = mBERTBase()
m2 = XLMRobertaBase()
m3 = mBERTBaseD()

# ensemble uses two models, quadsemble is our final architecture

ensemble_model = NLPEnsembleDouble(m1, m2, num_classes)
quadsemble_model = NLPEnsembleQuad(m1, m2, m3, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()
learning_rate = 1e-5
optimizer = optim.AdamW(quadsemble_model.parameters(), lr=learning_rate, weight_decay=1e-5)

num_epochs = 10


metric = BinaryAccuracy()
f1 = BinaryF1Score()

best_f1 = 0

for epochs in range(num_epochs):
    accuracy, f1_score_micro, f1_score_macro, loss = train_epoch(quadsemble_model, train_dataloader)
    print(f"\n Epoch:{epochs + 1} / {num_epochs}, Training Accuracy:{accuracy:.5f}, Training F1 Micro: {f1_score_micro:.5f},Training F1 Macro:{f1_score_macro:.5f}, Training Loss: {loss:.5f}", flush=True)

    val_accuracy, f1_micro_val, f1_macro_val, loss = val_epoch(quadsemble_model, val_dataloader)
    print(f"\n Epoch:{epochs + 1} / {num_epochs}, Val accuracy:{val_accuracy:.5f}, Val F1 Micro: {f1_micro_val:.5f}, Val F1 Macro:{f1_macro_val:.5f}, Validation Loss: {loss:.5f}", flush=True)

    if f1_macro_val > best_f1:
        print('Saving Model :)')
        torch.save(quadsemble_model, f'meme_triad_stack_best.pth')
        torch.save(quadsemble_model.state_dict(),f'meme_triad_stack_weights.pth')
        best_f1 = f1_macro_val
    else:
        best_f1 = best_f1

#torch.save(quadsemble_model, 'meme_triad_35.pth')
#torch.save(quadsemble_model.state_dict(),'meme_triad_35_weights.pth')