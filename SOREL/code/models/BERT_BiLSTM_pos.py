from matplotlib.style import context
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import rnn
from transformers import AutoModel

class BERT_BiLSTM_pos(nn.Module):
    def __init__(self, config):
        super(BERT_BiLSTM_pos, self).__init__()
        self.config = config

        vocab_size = config['vocab_size']
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']

        self.word_emb = nn.Embedding(vocab_size, input_size)
        self.bert_model = AutoModel.from_pretrained(self.config['bert_model'])

        if self.config['freeze_bert']:
            for param in self.bert_model.parameters():
                param.requires_grad = False


        self.coref_embed = nn.Embedding(self.config['max_length'], self.config['coref_size'], padding_idx=0)

        if self.config['use_single_rnn'] or self.config['use_re_rnn']:
            self.rnn = EncoderLSTM(input_size+self.config['coref_size'], hidden_size, num_layers, True, True, config['dropout'], False)
            self.linear_rnn = nn.Linear(num_layers * hidden_size, hidden_size)
        else:
            self.linear_rnn = nn.Linear(input_size+self.config['coref_size'], hidden_size)
        
        if self.config['use_re_rnn']:
            self.linear_rnn_sent = nn.Linear(input_size+self.config['coref_size'], hidden_size)
        
        if self.config['use_bilinear']:
            self.so_bili = torch.nn.Bilinear(hidden_size, hidden_size, config['re_embedding'])
        else:
            self.bili_linear = nn.Linear(2*hidden_size, config['re_embedding'])

        self.linear_re = nn.Linear(config['re_embedding'], config['so_relation_num'])

        if self.config['use_sep_rnn']:
            self.sent_rnn = EncoderLSTM(input_size+self.config['coref_size'], hidden_size, num_layers, True, True, config['dropout'], False)
            self.sent_lienar_re = nn.Linear(num_layers * hidden_size, hidden_size)

        self.sent_emb_linear = nn.Linear(hidden_size + config['re_embedding'], config['re_embedding'])
        self.linear_se = nn.Linear(config['re_embedding'], 1)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, context_idxs, h_mapping, t_mapping, sent_h_mapping, sent_t_mapping, context_pos, input_lengths):
        if self.config['use_bert']:
            sent = self.bert_model(context_idxs)['last_hidden_state']
            bert_sent = sent
        else:
            sent = self.word_emb(context_idxs)
        
        sent = torch.cat([sent, self.coref_embed(context_pos)], dim=-1)

        if self.config['use_single_rnn']:
            context_output = self.rnn(sent, input_lengths)
            context_output = torch.relu(self.linear_rnn(context_output))
            sent_context_output = context_output
        elif self.config['use_re_rnn']:
            context_output = self.rnn(sent, input_lengths)
            context_output = torch.relu(self.linear_rnn(context_output))
            sent_context_output = self.linear_rnn_sent(sent)
        else:
            context_output = self.linear_rnn(sent)
            sent_context_output = context_output

        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)

        if self.config['use_bilinear']:
            re_embed = self.so_bili(start_re_output, end_re_output)
        else:
            re_embed = self.bili_linear(torch.cat([start_re_output, end_re_output], dim=-1))
        re_pred = self.linear_re(re_embed)
        re_pred = self.softmax(re_pred)
        
        if self.config['use_sep_rnn']:
            sent_context_output = self.sent_rnn(sent, input_lengths)
            sent_context_output = torch.relu(self.sent_lienar_re(sent_context_output))
            sent_start_output = torch.matmul(sent_h_mapping, sent_context_output)
        else:
            sent_start_output = torch.matmul(sent_h_mapping, sent_context_output)


        sent_emb = torch.cat([sent_start_output.unsqueeze(1).expand(-1, re_embed.size(1), -1, -1), re_embed.unsqueeze(2).expand(-1, -1, sent_h_mapping.size(1), -1)], dim=-1)
        sent_emb = self.sent_emb_linear(sent_emb)
        sent_pred = self.linear_se(sent_emb).squeeze(3)


        return re_pred, sent_pred


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units# if not bidir else num_units * 2
                output_size_ = num_units
            if bidir:
                output_size_ //= 2
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units // 2 if bidir else num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units // 2 if bidir else num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            
            output, hidden = self.rnns[i](output, (hidden, c))

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]
