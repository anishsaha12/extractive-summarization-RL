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

class ArticleEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(ArticleEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size, self.hidden_size, 
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

    def forward(self, input_sents, hidden):
        output, hidden = self.rnn(
            input_sents, hidden
        )
        
        if self.bidirectional:
            output = (output[:, :, :self.hidden_size]+output[:, :, self.hidden_size:])/2
            hidden = (hidden[:self.num_layers, :, :]+hidden[self.num_layers:, :, :])/2
        
        return output, hidden

    def initHidden(self, mydevice, batch_size=1):
        # shape is (num_directions*num_layers, BATCH_SIZE, hidden_size)
        h0 = torch.zeros(
            (2 if self.bidirectional else 1)*self.num_layers, 
            batch_size, 
            self.hidden_size,
            dtype=torch.float,
            device=mydevice
        )
        return h0

class SummaryStateEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SummaryStateEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size, self.hidden_size, 
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False  # want to keep next sentence as last seen sentence
        )

    def forward(self, input_sents, hidden, input_lengths):
        x = rnn.pack_padded_sequence(input_sents, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(
            x, hidden
        )
        output, _ = rnn.pad_packed_sequence(output, batch_first=True, padding_value=0.)
        return output, hidden

    def initHidden(self, mydevice, batch_size=1):
        # shape is (num_directions*num_layers, BATCH_SIZE, hidden_size)
        h0 = torch.zeros(
            self.num_layers, 
            batch_size, 
            self.hidden_size,
            dtype=torch.float,
            device=mydevice
        )
        return h0

class Attention(nn.Module):
    def __init__(self, hidden_size, seq_len, batch_size, method="key:encoder"):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.method = method
        
        if self.method=="key:encoder":
            self.Wa = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.Ua = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, self.hidden_size))
        elif self.method=="key:hidden":
            self.Za = nn.Linear(self.hidden_size*2, self.seq_len)
        else:
            raise NotImplementedError
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        

    def forward(self, query, key, value):
        if self.method=="key:encoder":
            # query:next_sent, key:encoder_outputs, value:encoder_outputs
            out = torch.tanh(
                self.Wa(query.squeeze(0).unsqueeze(1)) + # BATCH_SIZE x 1 (seq_len) x hidden_size
                self.Ua(key) # BATCH_SIZE x enc_seq_len x hidden_size
            ) # BATCH_SIZE x encoder_seq_len x hidden_size
            attention_weights = out.bmm(self.va.unsqueeze(-1)).squeeze(-1) # BATCH_SIZE x enc_seq_len
        elif self.method=="key:hidden":
            # query:next_sent, key:hidden, value:encoder_outputs
            out = torch.cat((
                query.squeeze(0), # BATCH_SIZE x hidden_size
                key.squeeze(0) # BATCH_SIZE x hidden_size
            ), 1) # BATCH_SIZE x 2*hidden_size
            attention_weights = self.Za(out) # BATCH_SIZE x enc_seq_len
        else:
            raise NotImplementedError
        
        attention_weights = F.softmax(attention_weights, dim=-1) # # BATCH_SIZE x enc_seq_len
        context = torch.bmm(attention_weights.unsqueeze(1), value) # BATCH_SIZE x 1 (seq_len) x hidden_size
        
        output = torch.cat((query.squeeze(0), context.squeeze(1)), 1) # BATCH_SIZE x 2*hidden_size
        output = self.attn_combine(output) # BATCH_SIZE x hidden_size
        output = F.relu(output) # BATCH_SIZE x hidden_size
        return output, attention_weights

class BufferDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, enc_seq_len, batch_size, num_layers=1, bidirectional=False, method="key:encoder"):
        super(BufferDecoderRNN, self).__init__()
        self.enc_seq_len = enc_seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.method = method
        
        if self.method not in ['key:encoder','key:hidden']:
            raise NotImplementedError

        self.summaryState = SummaryStateEncoderRNN(
            input_size=input_size, 
            hidden_size=self.hidden_size, 
            num_layers=1
        )
        self.rnn = nn.GRU(
            self.hidden_size, self.hidden_size, 
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.attention = Attention(
            hidden_size=self.hidden_size, 
            seq_len=self.enc_seq_len, 
            batch_size=self.batch_size, 
            method=self.method
        )
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, summary_sents, summary_lengths, hidden, encoder_outputs):
        _, next_sent = self.summaryState(summary_sents, hidden, summary_lengths) # 1 x BATCH_SIZE x hidden_size
        
        if self.method == 'key:encoder':
            context, attention_weights = self.attention(next_sent, encoder_outputs, encoder_outputs) 
        elif self.method == 'key:hidden':
            context, attention_weights = self.attention(next_sent, hidden, encoder_outputs) 
        
        context = context.unsqueeze(1)
        output, hidden = self.rnn(context, hidden)
        if self.bidirectional:
            output = (output[:, :, :self.hidden_size]+output[:, :, self.hidden_size:])/2
            hidden = (hidden[:self.num_layers, :, :]+hidden[self.num_layers:, :, :])/2
        
        output = F.log_softmax(
            self.out(output.squeeze(1)),
            dim=-1
        )
        return output, hidden, attention_weights

    def initHidden(self, mydevice, batch_size=1):
        # shape is (num_directions*num_layers, BATCH_SIZE, hidden_size)
        h0 = torch.zeros(
            (2 if self.bidirectional else 1)*self.num_layers, 
            batch_size, 
            self.hidden_size,
            dtype=torch.float,
            device=mydevice
        )
        return h0
    
class ArticleSummarizer(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, 
        enc_seq_len, summary_max_len, batch_size, 
        enc_num_layers=1, enc_bidirectional=False, 
        dec_num_layers=1, dec_bidirectional=False, attention_method="key:encoder",
        use_teacher_forcing=True, device='cpu'
    ):
        super(ArticleSummarizer, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.enc_seq_len = enc_seq_len
        self.summary_max_len = summary_max_len
        self.batch_size = batch_size
        
        self.enc_num_layers = enc_num_layers
        self.enc_bidirectional = enc_bidirectional
        
        self.dec_num_layers = dec_num_layers
        self.dec_bidirectional = dec_bidirectional
        self.attention_method = attention_method
        self.use_teacher_forcing = use_teacher_forcing
        
        if self.attention_method not in ['key:encoder','key:hidden']:
            raise NotImplementedError

        self.encoder = ArticleEncoderRNN(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.enc_num_layers,
            bidirectional=self.enc_bidirectional
        )
        self.decoder = BufferDecoderRNN(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            output_size=self.output_size, 
            enc_seq_len=self.enc_seq_len , 
            batch_size=self.batch_size, 
            num_layers=self.dec_num_layers, 
            bidirectional=self.dec_bidirectional, 
            method=self.attention_method
        )
        
    def _extend_summary_action(self, summaries, summary_lens, next_sents, actions):
        summaries[actions,np.array(summary_lens)[actions],:] = next_sents[actions]
        summary_lens = [e+1 if a==True else e for (e,a) in zip(summary_lens,actions)]
        return summaries, summary_lens

    def _add_sent_to_summary(self, summaries, summary_lens, next_sents):
        batch_size = len(summary_lens)
        summaries[range(batch_size),summary_lens,:] = next_sents
        summary_lens = [e+1 for e in summary_lens]
        return summaries, summary_lens

    def _remove_sent_from_summary(self, summaries, summary_lens):
        batch_size = len(summary_lens)
        summary_lens = [e-1 for e in summary_lens]
        summaries[range(batch_size),summary_lens,:] = torch.zeros(  
            batch_size, 
            summaries.shape[-1],
            dtype=torch.float,
            device=summaries.device.type
        )
        return summaries, summary_lens

    def encode_decode(self, sentences_batch, target_actions):
        """
        Params:
            sentences_batch: (batch_size x enc_seq_len x input_size)
            targets: (batch_size x enc_seq_len)
        Outputs:
            decoder_attention: (batch_size x enc_seq_len (decoding timestep) x enc_seq_len (encoding timestep))
            logits: (batch_size*enc_seq_len, output_size)
            labels: (batch_size*enc_seq_len)
        """
        decoder_attention = []
        logits = torch.zeros(
            self.enc_seq_len,
            self.batch_size, 
            self.output_size,
            device=self.device
        )
        labels = torch.tensor(np.where(
            target_actions.reshape(-1)==action_map['gold_keep'], 
            action_map['mdl_keep'], 
            action_map['mdl_skip']
        ), device=self.device)

        hidden = self.encoder.initHidden(self.device, self.batch_size)
        encoder_outputs, encoder_hidden = self.encoder(sentences_batch, hidden)

        summary_state_next_sent_batch = torch.zeros(      # current summary sents + next sent (at end)
            self.batch_size, 
            self.summary_max_len+1,
            self.input_size,
            dtype=torch.float,
            device=self.device
        )
        summary_lengths = [0 for i in range(self.batch_size)]

        decoder_hidden = encoder_hidden

        for time_step in range(self.enc_seq_len):
            _, summary_lengths = self._add_sent_to_summary(
                summary_state_next_sent_batch, summary_lengths, 
                sentences_batch[:, time_step, :]
            )

            decoder_output, decoder_hidden, attention_weights = self.decoder(
                summary_state_next_sent_batch, summary_lengths, decoder_hidden, encoder_outputs
            )

            _, summary_lengths = self._remove_sent_from_summary(
                summary_state_next_sent_batch, summary_lengths
            )

            if self.use_teacher_forcing:
                extend_sumaries = target_actions[:, time_step]==action_map['gold_keep']
            else:
                topv, topi = decoder_output.topk(1)
                decoder_actions = topi.squeeze().detach().cpu().numpy()
                extend_sumaries = decoder_actions==action_map['mdl_keep']
            extend_sumaries = extend_sumaries&(np.array(summary_lengths)<self.summary_max_len)

            _, summary_lengths = self._extend_summary_action(
                summary_state_next_sent_batch, summary_lengths, 
                sentences_batch[:, time_step, :], 
                extend_sumaries
            )

            decoder_attention.append(attention_weights.detach().cpu().numpy())
            logits[time_step] = decoder_output

            if all(np.array(summary_lengths)==self.summary_max_len):
                break

        logits = logits.transpose(1, 0).contiguous().view(-1, self.output_size)
        return logits, labels, np.array(decoder_attention).transpose(1,0,2)
    
    def forward(self, sentences_batch):
        """
        Params:
            sentences_batch: (batch_size x enc_seq_len x input_size)
        Outputs:
            decoder_attention: (batch_size x enc_seq_len (decoding timestep) x enc_seq_len (encoding timestep))
            logits: (batch_size*enc_seq_len, output_size)
        """
        decoder_attention = []
        logits = torch.zeros(
            self.enc_seq_len,
            self.batch_size, 
            self.output_size,
            device=self.device
        )

        hidden = self.encoder.initHidden(self.device, self.batch_size)
        encoder_outputs, encoder_hidden = self.encoder(sentences_batch, hidden)

        summary_state_next_sent_batch = torch.zeros(      # current summary sents + next sent (at end)
            self.batch_size, 
            self.summary_max_len+1,
            self.input_size,
            dtype=torch.float,
            device=self.device
        )
        summary_lengths = [0 for i in range(self.batch_size)]

        decoder_hidden = encoder_hidden

        for time_step in range(self.enc_seq_len):
            _, summary_lengths = self._add_sent_to_summary(
                summary_state_next_sent_batch, summary_lengths, 
                sentences_batch[:, time_step, :]
            )

            decoder_output, decoder_hidden, attention_weights = self.decoder(
                summary_state_next_sent_batch, summary_lengths, decoder_hidden, encoder_outputs
            )

            _, summary_lengths = self._remove_sent_from_summary(
                summary_state_next_sent_batch, summary_lengths
            )

            topv, topi = decoder_output.topk(1)
            decoder_actions = topi.squeeze().detach().cpu().numpy()
            extend_sumaries = decoder_actions==action_map['mdl_keep']
            
            extend_sumaries = extend_sumaries&(np.array(summary_lengths)<self.summary_max_len)

            _, summary_lengths = self._extend_summary_action(
                summary_state_next_sent_batch, summary_lengths, 
                sentences_batch[:, time_step, :], 
                extend_sumaries
            )

            decoder_attention.append(attention_weights.detach().cpu().numpy())
            logits[time_step] = decoder_output

            if all(np.array(summary_lengths)==self.summary_max_len):
                break

        logits = logits.transpose(1, 0).contiguous().view(-1, self.output_size)
        return logits, np.array(decoder_attention).transpose(1,0,2)
    
    def loss(self, batch, criterion):
        article_ids, sentences_batch, target_actions, future_rewards_batch = batch
        sentences_batch = torch.tensor(sentences_batch, dtype=torch.float, device=self.device)
        target_actions = np.array(target_actions)
        
        logits, labels, decoder_attention = self.encode_decode(sentences_batch, target_actions)
        if future_rewards_batch==None:
            future_rewards_batch = torch.ones(
                self.batch_size*self.enc_seq_len , device=self.device
            )
        else:
            future_rewards_batch = np.array(future_rewards_batch)
            future_rewards_batch = torch.tensor(
                future_rewards_batch.reshape(-1), dtype=torch.float, device=self.device
            )
        # multiply with intermediate future-rewards (REINFORCE)
        logits[range(len(labels)),labels] = logits[range(len(labels)),labels]*future_rewards_batch
        loss = criterion(logits, labels)
        return loss, logits, labels, future_rewards_batch, decoder_attention
