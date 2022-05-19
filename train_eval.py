import pickle
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from model import ArticleSummarizer
from model_utils import train, evaluate, get_data_chunk


data_base = 'data'
dataset_name = 'cnn_dailymail'

data_path = data_base+'/'+dataset_name

model_data_folder = data_path+'/top_sentence_embs'
model_save_path = 'model/summarizer_model_3.pt'

load_existing_model = False

batch_size = 128
num_epochs = 10

mydevice = 'cuda'
# mydevice = 'cpu'


if load_existing_model:
    summarizer1 = torch.load(model_save_path).to(mydevice)
    print('Loaded Saved Model')
else:
    summarizer1 = ArticleSummarizer(
        input_size=768, hidden_size=256, output_size=2, 
        enc_seq_len=10, summary_max_len=4, batch_size=batch_size, 
        enc_num_layers=1, enc_bidirectional=True, 
        dec_num_layers=1, dec_bidirectional=False, attention_method="key:encoder",
        use_teacher_forcing=True, device=mydevice
    ).to(mydevice)
    print('Created Model')

criterion = nn.NLLLoss()
model_optimizer = optim.Adam(summarizer1.parameters(), lr=0.001)


print('Loading Data...')
train_chunks = []
for chunk_num in range(1,6):
    train_chunk, num_batches = get_data_chunk(model_data_folder, 'train', chunk_num, batch_size)
    train_chunk = train_chunk[:-1] # ignore last batch as its not same size
    train_chunks.append(train_chunk)
eval_chunk, num_e_batches = get_data_chunk(model_data_folder, 'validation', 1, batch_size)
eval_chunk = eval_chunk[:-1] # ignore last batch as its not same size

losses = []
for epoch in range(num_epochs):
    c_losses = []
    print('Training...')
    for train_chunk in train_chunks:
        summarizer1, model_optimizer, loss, info = train(summarizer1, criterion, model_optimizer, train_chunk)
        c_losses.append(loss)
    loss = np.mean(c_losses)
    losses.append(loss)
    print('Evaluating...')
    e_loss, e_acc, e_info = evaluate(summarizer1, criterion, eval_chunk)
    print('Epoch:',epoch,'- Train Loss:',loss,'- Eval Loss:',e_loss,'- Eval Acc:',e_acc)
    torch.save(summarizer1, model_save_path)
    
print('All Epoch Train Loss:', losses)
print('Final Eval Accuracy:', e_acc)
torch.save(summarizer1, model_save_path)

