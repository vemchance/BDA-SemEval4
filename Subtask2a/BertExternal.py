

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
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertModel, BertTokenizer
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, MulticlassF1Score, MultilabelConfusionMatrix
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

print(torch.cuda.current_device())
torch.cuda.empty_cache()

### read a csv instead ###

task_data = pd.read_json('external_data.json')
task_data['techniques_encoded'] = task_data['labels'].apply(lambda y: ['None'] if len(y)==0 else y)
one_hot = LabelBinarizer()
task_data['techniques_encoded'] = one_hot.fit_transform(task_data['labels'])
task_data.dropna(inplace=True)
num_classes = len(task_data['techniques_encoded'].unique())
print(num_classes, flush=True)
#### read a csv instead ####

train, val = train_test_split(task_data, test_size=0.30, shuffle=True, stratify=task_data['language'])

train_text = train['description'].values.tolist()
train_labels = train['techniques_encoded'].values.tolist()

val_text = val['description'].values.tolist()
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
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', return_dict=False)

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
                                 attention_mask=batch['attention_mask'],
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
                                     attention_mask=batch['attention_mask'],
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


class BertBase(torch.nn.Module):
    def __init__(self):
        super(BertBase, self).__init__()
        self.l1 = transformers.AutoModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, num_classes)

    def forward(self, ids, attention_mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


ensemble_model = BertBase()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()
learning_rate = 1e-5
optimizer = optim.Adam(ensemble_model.parameters(), lr=learning_rate, weight_decay=1e-5)

num_epochs = 30


metric = BinaryAccuracy()
f1 = BinaryF1Score()

best_f1 = 0

for epochs in range(num_epochs):
    accuracy, f1_score_micro, f1_score_macro, loss = train_epoch(ensemble_model, train_dataloader)
    print(f"\n Epoch:{epochs + 1} / {num_epochs}, Training Accuracy:{accuracy:.5f}, Training F1 Micro: {f1_score_micro:.5f},Training F1 Macro:{f1_score_macro:.5f}, Training Loss: {loss:.5f}", flush=True)

    val_accuracy, f1_micro_val, f1_macro_val, loss = val_epoch(ensemble_model, val_dataloader)
    print(f"\n Epoch:{epochs + 1} / {num_epochs}, Val accuracy:{val_accuracy:.5f}, Val F1 Micro: {f1_micro_val:.5f}, Val F1 Macro:{f1_macro_val:.5f}, Validation Loss: {loss:.5f}", flush=True)

    if f1_macro_val > best_f1:
        print('Saving Model :)')
        torch.save(ensemble_model, f'bert_external_2a_best.pth')
        torch.save(ensemble_model.state_dict(),f'bert_external_2a_best_weights.pth')
        best_f1 = f1_macro_val
    else:
        best_f1 = best_f1

torch.save(ensemble_model, 'bert_external_2a.pth')
torch.save(ensemble_model.state_dict(),'bert_external_2a_weights.pth')