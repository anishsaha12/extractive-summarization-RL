import random
import time
import pickle
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

action_map = {
    'gold_keep': 1,
    'gold_skip': 2,
    'mdl_keep': 1,
    'mdl_skip': 0
}

def train(model, criterion, optimizer, train_data): # one epoch
    train_data = tqdm.tqdm(train_data)
    losses = []
    info = []
    model.train()
    for batch in train_data:
        optimizer.zero_grad()
        loss, logits, labels, future_rewards_batch, decoder_attention = model.loss(batch, criterion)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        info.append(
            (
                logits.detach().cpu().numpy(), labels.cpu().numpy(), 
                future_rewards_batch.cpu().numpy(), decoder_attention
            )
        )
    epoch_avg_loss = np.mean(losses)
    return model, optimizer, epoch_avg_loss, info

def evaluate(model, criterion, eval_data):
    eval_data = tqdm.tqdm(eval_data)
    losses = []
    info = []
    accs = []

    model.eval()
    model.use_teacher_forcing = False
    
    with torch.no_grad():
        for batch in eval_data:
            loss, logits, labels, future_rewards_batch, decoder_attention = model.loss(batch, criterion)
            losses.append(loss.item())
            info.append(
                (
                    logits.detach().cpu().numpy(), labels.cpu().numpy(), 
                    future_rewards_batch.cpu().numpy(), decoder_attention
                )
            )
        epoch_avg_loss = np.mean(losses)
        e_info = []
        accs = []
        for b_info in info:
            preds = b_info[0]
            preds = preds.reshape(-1,model.enc_seq_len, 2)
            labels = b_info[1]
            decoder_attention = b_info[3]
            actions = preds.argmax(axis=-1)
            actions[np.cumsum(actions, axis=-1)>model.summary_max_len] = action_map['mdl_skip']
            accuracy = sum(actions.flatten()==labels.flatten())/len(labels.flatten())
            actions = np.where(
                actions==action_map['mdl_keep'], 
                action_map['gold_keep'], 
                action_map['gold_skip']
            )
            e_info.append(
                (preds, labels, decoder_attention, accuracy, actions)
            )
            accs.append(accuracy)
        epoch_avg_acc = np.mean(accs)
    return epoch_avg_loss, epoch_avg_acc, e_info

def inference(model, inf_data):
    inf_data = tqdm.tqdm(inf_data)
    info = []
    
    model.eval()
    model.use_teacher_forcing = False
    
    with torch.no_grad():
        for batch in inf_data:
            article_ids, sentences_batch, _, _ = batch
            sentences_batch = torch.tensor(sentences_batch, dtype=torch.float, device=model.device)
            logits, decoder_attention = model.forward(sentences_batch)
            info.append(
                (
                    logits.detach().cpu().numpy(), decoder_attention
                )
            )
        e_info = []
        for b_info in info:
            preds = b_info[0]
            preds = preds.reshape(-1,model.enc_seq_len, 2)
            decoder_attention = b_info[1]
            actions = preds.argmax(axis=-1)
            actions[np.cumsum(actions, axis=-1)>model.summary_max_len] = action_map['mdl_skip']
            actions = np.where(
                actions==action_map['mdl_keep'], 
                action_map['gold_keep'], 
                action_map['gold_skip']
            )
            
            e_info.append(
                (preds, decoder_attention, actions)
            )
    return e_info

def get_data_chunk(model_data_folder, split, chunk_num, batch_size):
    with open(
          model_data_folder+'/'+split+'/'+split+'_data_article_ids_part_'+str(chunk_num)+".pkl", 
          "rb"
        ) as input_f:
            a_ids = pickle.load(input_f)
    with open(
          model_data_folder+'/'+split+'/'+split+'_data_part_'+str(chunk_num)+".pkl",
          "rb"
        ) as input_f:
            article_inputs = pickle.load(input_f)
    with open(
          model_data_folder+'/'+split+'/'+split+'_gold_actions_part_'+str(chunk_num)+".pkl",
          "rb"
        ) as input_f:
            gold_actions = pickle.load(input_f)
    with open(
          model_data_folder+'/'+split+'/'+split+'_gold_future_rewards_part_'+str(chunk_num)+".pkl",
          "rb"
        ) as input_f:
            gold_future_rewards = pickle.load(input_f)
    
    tot_size = len(gold_future_rewards)
    data = []
    for i in range(0, tot_size, batch_size):
        batch = (
            a_ids[i:i+batch_size], article_inputs[i:i+batch_size], 
            gold_actions[i:i+batch_size], gold_future_rewards[i:i+batch_size]
        )
        data.append(batch)

    return data, len(data)

