
from modeling.character_bert import CharacterBertModel

from transformers import BertForSequenceClassification

import time
import numpy as np
import matplotlib.pyplot as plt
import json

import torch
from collections import Counter

labels = []
sentences = []

testlabels = []
testsentences = []

labelsindecies=[]
testlabelsindecies=[]
def split(word):
    return [char for char in word]


def charactfy(str):
    l = split(str)
    concat = ""
    for word in l:
        concat += word + " "
 
    return concat


def addLabel(label):
    if label in labels:
        return labels.index(label)
    else:
        labels.append(label)
        return labels.index(label)

def OOVFilter(sentence, label):
    i = 0
    while i < len(sentence):
        if sentence[i].lower() != label[i]:
            sentences.append(charactfy(sentence[i]))
            labelsindecies.append(addLabel(label[i])) 
        i += 1

def OOVFilterExact(sentence, label):
    i = 0
    while i < len(sentence):
        if sentence[i].lower() != label[i]:
            sentences.append(charactfy(label[i]))
            labelsindecies.append(addLabel(label[i])) 
        i += 1


def TestOOVFilter(sentence, label):
    i = 0
    while i < len(sentence):
        if sentence[i].lower() != label[i]:
            testsentences.append(charactfy(sentence[i]))
            testlabelsindecies.append(addLabel(label[i]))
        i += 1


jsonFile = open("train_data.json", "r")
values = json.load(jsonFile)
jsonFile.close()
# Use TFDS to load the Portugese-English translation dataset from the TED Talks Open Translation Project.
# This dataset contains approximately 50000 training examples, 1100 validation examples, and 2000 test examples.


jsonFile2 = open("test_data.json", "r")
testinput = json.load(jsonFile2)
jsonFile2.close()

jsonFile3 = open("test_truth.json", "r")
testoutput = json.load(jsonFile3)
jsonFile3.close()


i2 = 0
while i2 < len(testinput):
    TestOOVFilter(testinput[i2]["input"], testoutput[i2]["output"])
    i2 += 1

i3=0
for item in values:
    OOVFilter(item["input"], item["output"])
    i3 +=1
    #if i3 > 1000:
        #break
    #OOVFilterExact(item["input"], item["output"])

#train_examples = tf.data.Dataset.from_tensor_slices((sentences, labels))
#val_examples = tf.data.Dataset.from_tensor_slices((testsentences, testlabels))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,num_labels = np.size(labels))
model = model.cuda()

#### REPLACING BERT WITH CHARACTER_BERT ####

#### character_bert_model = CharacterBertModel.from_pretrained(
 ####    './pretrained-models/bert-base-uncased/')
#### model.bert = character_bert_model


model.train()
 
from transformers import AdamW
#optimizer = AdamW(model.parameters(), lr=1e-5)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
# This variable contains all of the hyperparemeter information our training loop needs
optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)

from transformers import BertTokenizer

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
 
n_epochs = 80 # or whatever
batch_size = 32 # or whatever
train_loss_set = []

for epoch in range(n_epochs):
    nb_tr_steps = 0
    tr_loss = 0
    # X is a torch Variable
    permutation = torch.randperm(len(sentences))
    model.train()
    for i in range(0,len(sentences), batch_size):
        #optimizer.zero_grad()

        indices = permutation[i:i+batch_size]

        text_batch, batch_y =  np.array(sentences)[indices].tolist() , np.array(labelsindecies)[indices].tolist() 
        optimizer.zero_grad()
        
        encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']

         
        attention_mask = encoding['attention_mask']
          
        labels2 = torch.tensor(batch_y).unsqueeze(0) 
    
        outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels2.to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()  
        train_loss_set.append(loss.item())   
        #maz = torch.argmax(outputs.logits,1, keepdim=False)
        # Update tracking variables
        tr_loss += loss.item() 
        nb_tr_steps += 1 
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    permutation = torch.randperm(len(testsentences))

    for i in range(0,len(testsentences), batch_size):
        #optimizer.zero_grad()

        indices = permutation[i:i+batch_size]

        text_batch, batch_y =  np.array(testsentences)[indices].tolist() , np.array(testlabelsindecies)[indices].tolist() 
   
        encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']

         
        attention_mask = encoding['attention_mask']
          
        b_labels = torch.tensor(batch_y).unsqueeze(0) 
        with torch.no_grad():
            output = model(input_ids.to(device),token_type_ids=None,  attention_mask=attention_mask.to(device) )
 
        # Move logits and labels to CPU
        logits = output.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        


 