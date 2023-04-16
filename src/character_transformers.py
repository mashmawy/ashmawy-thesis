 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator 
from typing import Iterable, List
import time
import numpy as np
import matplotlib.pyplot as plt
import json

SRC_LANGUAGE = 'err'
TGT_LANGUAGE = 'normal'

# Place-holders
token_transform = {}
vocab_transform = {}
 

# Create source and target language tokenizer. Make sure to install the dependencies.
# pip install -U spacy
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
token_transform[SRC_LANGUAGE] = get_tokenizer(None, language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer(None, language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
   
    for data_sample in data_iter:
        yield token_transform[language](data_sample)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


 
 
testLabels =[]
testSentences =[]

trainLabels =[]
trainSentences =[]

valLabels =[]
valSentences =[]
def split(word): 
    return [char for char in word]  

def charactfy(str):
    l = split(str) 
    concat = ''
    for word in l:
        concat += word + ' '  
    return concat + " [org] " + str.lower() 

def TrainFilter(sentence, label): 
    if(sentence.lower() != label.lower()):
        trainSentences.append(charactfy(sentence))
        trainLabels.append(label + " oov") 
        trainSentences.append(charactfy(label))
        trainLabels.append(label + " iv") 
    elif( "#" not in label and "http" not in label and "@" not in label):
        trainSentences.append(charactfy(label))
        trainLabels.append(label + " iv") 
    
def ValidationFilter(sentence, label): 
    if(sentence.lower() != label.lower()):
        valSentences.append(charactfy(sentence))
        valLabels.append(label+ " oov")
    elif( "#" not in label and "http" not in label and "@" not in label):
        valSentences.append(charactfy(label))
        valLabels.append(label + " iv")  
         
def TestFilter(sentence, label): 
    if(sentence.lower() != label.lower()):
        testSentences.append(charactfy(sentence))
        testLabels.append(label+ " oov")
    elif( "#" not in label and "http" not in label and "@" not in label):
        testSentences.append(charactfy(sentence))
        testLabels.append(label + " iv")  
         
  
 #  'multilexnorm//data//en//train.norm'
def AddTrain(lang):
    datafile = open('multilexnorm//data//' + lang +'//train.norm', 'r', encoding="utf8")
    Lines = datafile.readlines()
    datafile.close()
    i2 = 0
    while i2 < len(Lines):
        words=  Lines[i2].strip().split('\t')
        if(len(words)> 1):
            TrainFilter(str(words[0]).strip().lower().replace('\n',''),str(words[1]).strip().lower().replace('\n',''))  
        i2 += 1   
# 'multilexnorm//data//en//dev.norm'
def AddValidation(lang):
    datafile2 = open('multilexnorm//data//' + lang +'//dev.norm', 'r', encoding="utf8")
    Lines2 = datafile2.readlines()
    datafile2.close()
    i23 = 0
    while i23 < len(Lines2):
        words=  Lines2[i23].strip().split('\t')
        if(len(words)> 1):
            ValidationFilter(str(words[0]).strip().lower().replace('\n',''),str(words[1]).strip().lower().replace('\n',''))  
        i23 += 1   

#'multilexnorm//data//en//test.norm'
def AddTest(lang):
    datafile2 = open('multilexnorm//data//' + lang +'//test.norm', 'r', encoding="utf8")
    Lines2 = datafile2.readlines()
    datafile2.close()
    i23 = 0
    while i23 < len(Lines2):
        words=  Lines2[i23].strip().split('\t')
        if(len(words)> 1):
            TestFilter(str(words[0]).strip().lower().replace('\n',''),str(words[1]).strip().lower().replace('\n',''))  
        i23 += 1   


#AddTrain('da')
#AddTest('da')

#AddTrain('de')
#AddValidation('de')
#AddTest('de')

#AddTrain('en')
#AddValidation('en')
#AddTest('en')

#AddTrain('es')
#AddTest('es')

#AddTrain('hr')
#AddValidation('hr')
#AddTest('hr')
 
#AddTrain('iden')
#AddValidation('iden')
#AddTest('iden')


#AddTrain('it')
#AddTest('it')


#AddTrain('nl')
#AddValidation('nl')
#AddTest('nl')



#AddTrain('sl')
#AddValidation('sl')
#AddTest('sl')



#AddTrain('sr')
#AddValidation('sr')
#AddTest('sr')


#AddTrain('tr') 
#AddTest('tr')

AddTrain('trde') 
AddTest('trde')


#for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    #train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
vocab_transform[SRC_LANGUAGE] = build_vocab_from_iterator(yield_tokens(trainSentences, SRC_LANGUAGE),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)
vocab_transform[TGT_LANGUAGE] = build_vocab_from_iterator(yield_tokens(trainLabels, TGT_LANGUAGE),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)
# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)
 




















from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)



#During training, we need a subsequent word mask that will prevent model to look into the future words when making predictions. 
#We will also need masks to hide source and target padding tokens. Below, let’s define a function that will take care of both.


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



#Let’s now define the parameters of our model and instantiate the same. Below, we also define our loss function which is the cross-entropy loss and the optmizer used for training.
torch.manual_seed(0) 
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 1024
BATCH_SIZE = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
NUM_EPOCHS = 5

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)






from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# function to collate data samples into batch tesors
def collate_fn2(batch,ln):
    src_batch= []
    for src_sample  in batch:
        src_batch.append(text_transform[ln](src_sample.rstrip("\n"))) 

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX) 
    return src_batch






from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    permutation = torch.randperm(len(trainSentences))
    optimizer.zero_grad()
    nb_tr_steps=0
    for i in range(0,len(trainSentences), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE] 
        if(indices.size(dim=0) < 8):
            break 
        src, tgt =collate_fn2(  np.array(trainSentences)[indices].tolist(),SRC_LANGUAGE) , collate_fn2(np.array(trainLabels)[indices].tolist(),TGT_LANGUAGE)

        src = src.type(torch.LongTensor).to(DEVICE)
        tgt = tgt.type(torch.LongTensor).to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward() 
        optimizer.step()
        losses += loss.item()
        nb_tr_steps += 1 

    return losses / nb_tr_steps

def flat_accuracy(preds, trainLabels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = trainLabels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def evaluate(model):
    model.eval()
    losses = 0
    acc =0
    nb_tr_steps=0
    permutation = torch.randperm(len(valSentences))
    for i in range(0,len(valSentences), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        src, tgt =collate_fn2(  np.array(valSentences)[indices].tolist(),SRC_LANGUAGE) , collate_fn2(np.array(valLabels)[indices].tolist(),TGT_LANGUAGE)

        src = src.type(torch.LongTensor).to(DEVICE)
        tgt = tgt.type(torch.LongTensor).to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        acc += flat_accuracy(logits.reshape(-1, logits.shape[-1]).detach().cpu().numpy(), tgt_out.reshape(-1).to('cpu').numpy())
        nb_tr_steps += 1 

    if nb_tr_steps == 0 :     
        return 0,0
    
    return losses / nb_tr_steps , acc / nb_tr_steps


def test(model):
    model.eval()
    losses = 0
    acc =0
    nb_tr_steps=0
    permutation = torch.randperm(len(testSentences))
    for i in range(0,len(testSentences), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        src, tgt =collate_fn2(  np.array(testSentences)[indices].tolist(),SRC_LANGUAGE) , collate_fn2(np.array(testLabels)[indices].tolist(),TGT_LANGUAGE)

        src = src.type(torch.LongTensor).to(DEVICE)
        tgt = tgt.type(torch.LongTensor).to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        acc += flat_accuracy(logits.reshape(-1, logits.shape[-1]).detach().cpu().numpy(), tgt_out.reshape(-1).to('cpu').numpy())
        nb_tr_steps += 1 

    return losses / nb_tr_steps , acc / nb_tr_steps


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
#Now we have all the ingredients to train our model. Let’s do it!

from timeit import default_timer as timer


for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss , acc = evaluate(transformer) 
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f},Val Acc: {acc:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss, 
            }, 'models//trde//trans512') 
test_loss , test_acc = test(transformer) 
print((f"Test loss: {test_loss:.3f},Test Acc: {test_acc:.3f}"))
