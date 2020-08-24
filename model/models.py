import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel

# exact replication of the model from DSC
class DSC(BaseModel):

    def __init__(self, use_lens=True):
        self.use_lens = use_lens
        super().__init__()

        self.fc_in = 15*8*2 + 3
        self.conv1_start = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_start.weight)
        self.conv1_drop_start = nn.Dropout(0.2)

        self.conv2_start = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_start.weight)
        self.conv2_drop_start = nn.Dropout(0.2)

        self.conv3_start = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_start.weight)
        self.conv3_drop_start = nn.Dropout(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_end.weight)
        self.conv1_drop_end = nn.Dropout(0.2)

        self.conv2_end = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_end.weight)
        self.conv2_drop_end = nn.Dropout(0.2)

        self.conv3_end = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_end.weight)
        self.conv3_drop_end = nn.Dropout(0.2)

        self.fc1 = nn.Linear(self.fc_in, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        if not self.use_lens: lens = torch.zeros_like(lens)

        # [128, 2, 140, 4]
        start, end = seqs[:, 0], seqs[:, 1]
        # reshape because 1D convolutions expect [B, C, L] dimensions
        start, end = start.view(-1, 4, 140), end.view(-1, 4, 140)

        # # 4, 140
        # x = F.relu(self.conv1_drop_start(self.conv1_start(start)))
        # # [32, 134]
        # x = F.max_pool1d(x, 2, stride=2)
        # # 32, 67
        # x = F.relu(self.conv2_drop_start(self.conv2_start(x)))
        # # 8, 64
        # x = F.max_pool1d(x, 2, stride=2)
        # # 8, 32
        # x = F.relu(self.conv3_drop_start(self.conv3_start(x)))
        # # 8, 30
        # x = F.max_pool1d(x, 2, stride=2)
        # # 8, 15

        x = F.max_pool1d(F.relu(self.conv1_drop_start(self.conv1_start(start))), 2, stride=2)
        x = F.max_pool1d(F.relu(self.conv2_drop_start(self.conv2_start(x))), 2, stride=2)
        x = F.max_pool1d(F.relu(self.conv3_drop_start(self.conv3_start(x))), 2, stride=2)

        xx = F.max_pool1d(F.relu(self.conv1_drop_end(self.conv1_end(end))), 2, stride=2)
        xx = F.max_pool1d(F.relu(self.conv2_drop_end(self.conv2_end(xx))), 2, stride=2)
        xx = F.max_pool1d(F.relu(self.conv3_drop_end(self.conv3_end(xx))), 2, stride=2)

        feats = torch.cat((x, xx), dim=1)
        feats = feats.view(-1, self.fc_in-3)
        feats = torch.cat((feats, lens), dim=1)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))

        return y


class DSC_4_SEQ(BaseModel):

    def __init__(self):
        super().__init__()

        self.fc_in = 15*8*4 + 3
        # sequence before start
        self.conv1_bef_start = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_bef_start.weight)
        self.conv1_drop_bef_start = nn.Dropout(0.2)

        self.conv2_bef_start = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_bef_start.weight)
        self.conv2_drop_bef_start = nn.Dropout(0.2)

        self.conv3_bef_start = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_bef_start.weight)
        self.conv3_drop_bef_start = nn.Dropout(0.2)

        # start sequence
        self.conv1_start = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_start.weight)
        self.conv1_drop_start = nn.Dropout(0.2)

        self.conv2_start = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_start.weight)
        self.conv2_drop_start = nn.Dropout(0.2)

        self.conv3_start = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_start.weight)
        self.conv3_drop_start = nn.Dropout(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_end.weight)
        self.conv1_drop_end = nn.Dropout(0.2)

        self.conv2_end = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_end.weight)
        self.conv2_drop_end = nn.Dropout(0.2)

        self.conv3_end = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_end.weight)
        self.conv3_drop_end = nn.Dropout(0.2)

        # end conv blocks here...
        self.conv1_after_end = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_after_end.weight)
        self.conv1_drop_after_end = nn.Dropout(0.2)

        self.conv2_after_end = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_after_end.weight)
        self.conv2_drop_after_end = nn.Dropout(0.2)

        self.conv3_after_end = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_after_end.weight)
        self.conv3_drop_after_end = nn.Dropout(0.2)

        self.fc1 = nn.Linear(self.fc_in, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        # [128, 2, 142, 4] or [128, 2, 140, 4]
        # lens = torch.zeros_like(lens)
        bef_start, start, end, after_end = seqs[:, 0], seqs[:, 1], seqs[:, 2], seqs[:, 3]
        bef_start, start, end, after_end = bef_start.view(-1, 4, 140), start.view(-1, 4, 140), \
                                           end.view(-1, 4, 140), after_end.view(-1, 4, 140)

        x = F.max_pool1d(F.relu(self.conv1_drop_bef_start(self.conv1_bef_start(bef_start))), 2, stride=2)
        x = F.max_pool1d(F.relu(self.conv2_drop_bef_start(self.conv2_bef_start(x))), 2, stride=2)
        x = F.max_pool1d(F.relu(self.conv3_drop_bef_start(self.conv3_bef_start(x))), 2, stride=2)

        xx = F.max_pool1d(F.relu(self.conv1_drop_start(self.conv1_start(start))), 2, stride=2)
        xx = F.max_pool1d(F.relu(self.conv2_drop_start(self.conv2_start(xx))), 2, stride=2)
        xx = F.max_pool1d(F.relu(self.conv3_drop_start(self.conv3_start(xx))), 2, stride=2)

        xxx = F.max_pool1d(F.relu(self.conv1_drop_end(self.conv1_end(end))), 2, stride=2)
        xxx = F.max_pool1d(F.relu(self.conv2_drop_end(self.conv2_end(xxx))), 2, stride=2)
        xxx = F.max_pool1d(F.relu(self.conv3_drop_end(self.conv3_end(xxx))), 2, stride=2)

        xxxx = F.max_pool1d(F.relu(self.conv1_drop_after_end(self.conv1_after_end(after_end))), 2, stride=2)
        xxxx = F.max_pool1d(F.relu(self.conv2_drop_after_end(self.conv2_after_end(xxxx))), 2, stride=2)
        xxxx = F.max_pool1d(F.relu(self.conv3_drop_after_end(self.conv3_after_end(xxxx))), 2, stride=2)

        feats = torch.cat((x, xx, xxx, xxxx), dim=1)
        feats = feats.view(-1, self.fc_in-3)
        feats = torch.cat((feats, lens), dim=1)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))

        return y

# 1 layers = seems best, as high as 85.8, fastest to train
# 2 layers = slow to train, stop at 85 after 63 epochs
# 3 layers = stopped after 20 epochs took like 20 min and it was still only at 82
# 19141 parameters
class BiLSTM(BaseModel):
    def __init__(self, three_len_feats=True):
        super().__init__()
        self.three_feats = three_len_feats
        self.LSTM_dim = 50
        self.seq_length = 140
        self.dim_fc = 64
        self.dropout_prob = 0.5
        self.lstm_layer = 1
        self.lstm_dropout = 0.2
        self.in_fc = 2 * self.LSTM_dim * self.lstm_layer + (3 if self.three_feats else 1)
        # mapping from sparse 4-d to dense 4-d
        self.embedding = nn.Linear(4, 4, bias=True)

        self.lstm1 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.lstm2 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.fc1 = nn.Linear(self.in_fc, self.dim_fc)
        self.drop_fc = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.dim_fc, 1)

    def forward(self, seqs, lens):
        # [256, 140, 4] input
        start, end = seqs[:, 0], seqs[:, 1]
        embedding = F.relu(self.embedding(start))
        output, (h_n, c_n) = self.lstm1(embedding)
        # output: [256, 140, 2*50], c_n/h_n: [2, 256, 25]
        # take only the last output as input for classification
        x = h_n.view(-1, self.LSTM_dim*self.lstm_layer)

        embedding2 = F.relu(self.embedding(end))
        output, (h_n, c_n) = self.lstm2(embedding2)
        xx = h_n.view(-1, self.LSTM_dim*self.lstm_layer)

        feats = torch.cat((x, xx), dim=1)
        if self.three_feats:
            feats = torch.cat((feats, lens.view(-1, 3)), dim=1)
        else:
            feats = torch.cat((feats, lens.view(-1, 1)), dim=1)
        feats = feats.view(-1, self.in_fc)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))
        return y

class AttnBiLSTM(BaseModel):
    def __init__(self, LSTM_dim=50, fc_dim=128, attn_dim=100, conv_size=3, attn_dropout=0.2, n_heads=4, head_dim=50,
                 seq_length=140, fc_dropout=0.5, attn_mode='heads', use_lens=True):
        super().__init__()
        assert conv_size % 2 == 1, "Only uneven convolution sizes allowed because uneven conv same padding support implemented"
        assert attn_mode in ['heads', 'no_query', 'single_head', 'conv', 'heads_no_query']
        self.conv_size = conv_size
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.use_lens = use_lens

        self.LSTM_dim = LSTM_dim
        self.seq_length = seq_length
        self.dim_fc = fc_dim
        self.dropout_prob = fc_dropout
        self.attn_dim = attn_dim
        self.in_fc = attn_dim + 3
        self.attn_mode = attn_mode
        self.bn = torch.nn.BatchNorm1d(LSTM_dim)

        self.embedding = nn.Linear(4, 4)

        assert LSTM_dim % 2 == 0, "Only even LSTM dim lengths allowed to avoid rounding issues"
        # halving because BiLSTM
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2,
                             bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2,
                             bidirectional=True, batch_first=True)

        if attn_mode == 'single_head':
            self.attention = AttentionBlock(LSTM_dim, attn_dim, attn_dropout)
        elif attn_mode == 'no_query':
            self.attention = AttentionBlockWithoutQuery(LSTM_dim, attn_dim)
        elif attn_mode == 'heads':
            self.attention = AttentionBlockWithHeads(LSTM_dim, attn_dim, n_heads, head_dim, attn_dropout)
        elif attn_mode == 'heads_no_query':
            self.attention = AttentionBlockWithoutQueryWithHeads(LSTM_dim, attn_dim, n_heads, head_dim, attn_dropout)
        elif attn_mode == 'conv':
            self.attention = AttentionBlockWithConv(seq_length, LSTM_dim, attn_dim, conv_size, attn_dropout)

        self.fc1 = nn.Linear(self.in_fc, self.dim_fc)
        self.drop_fc = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.dim_fc, 1)


    def forward(self, seqs, lens):
        if not self.use_lens: lens = torch.zeros_like(lens)
        # [256, 142, 4] or [256, 140, 4]
        start, end = seqs[:, 0], seqs[:, 1]
        start, end = start.view(-1, self.seq_length, 4), end.view(-1, self.seq_length, 4)
        embedding = F.relu(self.embedding(start))
        # output = [256, 140, 2*50]  // 256, 4, 50???
        output1, (h_n, c_n) = self.lstm1(embedding)

        embedding2 = F.relu(self.embedding(end))
        output2, (h_n, c_n) = self.lstm2(embedding2)

        feats = torch.cat((output1, output2), dim=1)
        feats = self.bn(feats.view(-1, self.LSTM_dim, self.seq_length*2)).view(-1, self.seq_length*2, self.LSTM_dim)

        # (batch, attn_dimension)
        attn_seq, ws = self.attention(feats)
        class_feats = torch.cat((attn_seq, lens), dim=1)
        y = self.drop_fc(F.relu(self.fc1(class_feats)))
        y = torch.sigmoid(self.fc2(y))
        return y, ws

class AttentionBlockWithHeads(BaseModel):
    def __init__(self, in_dim, out_dim, n_heads, head_dim, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.keys = clones(torch.nn.Linear(in_dim, head_dim), n_heads)
        self.values = clones(torch.nn.Linear(in_dim, head_dim), n_heads)
        self.queries = clones(torch.nn.Linear(1, head_dim), n_heads)
        self.heads_unifier = torch.nn.Linear(head_dim*n_heads, out_dim)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        outputs, attn_ws = [], []
        for i in range(self.n_heads):
            values = self.values[i](input_seq)
            keys = self.keys[i](input_seq)
            # since same query for each element in batch
            queries = self.queries[i].weight.repeat(batch_size, 1, 1)
            unnorm_weights = torch.bmm(keys, queries)
            attn_w = torch.softmax(unnorm_weights, dim=1)
            drop_attn_w = self.drop(attn_w)
            weighted_vals = values * drop_attn_w
            output = torch.sum(weighted_vals, dim=1)
            outputs.append(output)
            attn_ws.append(attn_w)

        zs = torch.cat(outputs, dim=1)
        attn_ws = torch.cat(attn_ws, dim=1)
        z = self.heads_unifier(zs)
        return z, attn_ws


class AttentionBlockWithoutQueryWithHeads(BaseModel):
    def __init__(self, in_dim, out_dim, n_heads, head_dim, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.keys = clones(torch.nn.Linear(in_dim, 1), n_heads)
        self.values = clones(torch.nn.Linear(in_dim, head_dim), n_heads)
        self.heads_unifier = torch.nn.Linear(head_dim*n_heads, out_dim)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, input_seq):
        outputs, attn_ws = [], []
        for i in range(self.n_heads):
            values = self.values[i](input_seq)
            keys = self.keys[i](input_seq)

            unnorm_weights = keys
            attn_w = torch.softmax(unnorm_weights, dim=1)
            drop_attn_w = self.drop(attn_w)
            weighted_vals = values * drop_attn_w
            output = torch.sum(weighted_vals, dim=1)
            outputs.append(output)
            attn_ws.append(attn_w)

        zs = torch.cat(outputs, dim=1)
        attn_ws = torch.cat(attn_ws, dim=1)
        z = self.heads_unifier(zs)
        return z, attn_ws

def clones(module, n):
    """Produce n identical layers. (Taken from the annotated transformer) """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class AttentionBlock(BaseModel):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.key = torch.nn.Linear(in_dim, out_dim)
        self.value = torch.nn.Linear(in_dim, out_dim)
        # I just have one query independent of sequence length
        self.drop = nn.Dropout(dropout)
        self.query = torch.nn.Linear(1, out_dim) # perhaps other way to express this

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        values = self.value(input_seq)
        keys = self.key(input_seq)
        # since same query for each element in batch
        queries = self.query.weight.repeat(batch_size, 1, 1)
        unnorm_weights = torch.bmm(keys, queries)
        attn_weights = torch.softmax(unnorm_weights, dim=1)
        attn_weights_drop = self.drop(attn_weights)

        weighted_vals = values * attn_weights_drop
        output = torch.sum(weighted_vals, dim=1)
        return output, attn_weights

class AttentionBlockWithoutQuery(BaseModel):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.key = torch.nn.Linear(in_dim, 1)
        self.value = torch.nn.Linear(in_dim, out_dim)

    def forward(self, input_seq):
        values = self.value(input_seq)
        unnorm_weights = self.key(input_seq)

        attn_weights = torch.softmax(unnorm_weights, dim=1)
        weighted_vals = values * attn_weights
        output = torch.sum(weighted_vals, dim=1)
        return output, attn_weights


class AttentionBlockWithConv(BaseModel):
    def __init__(self, seq_len, in_dim, out_dim, kernel_size, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.key = torch.nn.Linear(in_dim, out_dim)
        self.value = torch.nn.Linear(in_dim, out_dim)
        # I just have one query independent of sequence length
        self.query = torch.nn.Linear(1, out_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.drop_conv = torch.nn.Dropout2d(0)
        self.bn_values = torch.nn.BatchNorm1d(out_dim)
        self.bn_keys = torch.nn.BatchNorm1d(out_dim)
        self.conv = torch.nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size//2)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        values = self.drop(self.value(input_seq))
        values = self.bn_values(values.view(batch_size, self.out_dim, 2*self.seq_len))
        # apply conv to start sequence
        values_start = values[:, :, :self.seq_len]
        values_conv_start = self.conv(values_start)
        values_conv_start = values_conv_start.view(batch_size, self.seq_len, self.out_dim)
        # apply conv to end sequence
        values_end = values[:, :, self.seq_len:]
        values_conv_end = self.conv(values_end)
        values_conv_end = values_conv_end.view(batch_size, self.seq_len, self.out_dim)
        # combine both start and end sequence again
        values_conv = torch.cat((values_conv_start, values_conv_end), dim=1)
        values_conv = self.drop_conv(values_conv)

        keys = self.drop(self.key(input_seq))
        keys = self.bn_keys(keys.view(batch_size, self.out_dim, 2*self.seq_len))
        # apply conv to start sequence
        keys_start = keys[:, :, :self.seq_len]
        keys_conv_start = self.conv(keys_start)
        keys_conv_start = keys_conv_start.view(batch_size, self.seq_len, self.out_dim)
        # apply conv to end sequence
        keys_end = keys[:, :, self.seq_len:]
        keys_conv_end = self.conv(keys_end)
        keys_conv_end = keys_conv_end.view(batch_size, self.seq_len, self.out_dim)
        # combine both start and end sequence again
        keys_conv = torch.cat((keys_conv_start, keys_conv_end), dim=1)
        keys_conv = self.drop_conv(keys_conv)

        # since same query for each element in batch
        queries = self.query.weight.repeat(batch_size, 1, 1)
        unnorm_weights = torch.bmm(keys_conv, queries)
        attn_weights = torch.softmax(unnorm_weights, dim=1)
        weighted_vals = values_conv * attn_weights
        output = torch.sum(weighted_vals, dim=1)
        return output, attn_weights

class AttentionBlockWithoutQueryWithConv(BaseModel):
    def __init__(self, seq_len, in_dim, out_dim, kernel_size, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.key = torch.nn.Linear(in_dim, 1)
        self.value = torch.nn.Linear(in_dim, out_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.drop_conv = torch.nn.Dropout2d(0.3)
        self.bn_values = torch.nn.BatchNorm1d(out_dim)
        self.conv = torch.nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size//2)

    def forward(self, input_seq):
        values = self.drop(self.value(input_seq))
        values = self.bn_values(values.view(-1, self.out_dim, self.seq_len*2))
        # apply conv to start sequence
        values_start = values[:, :, :self.seq_len]
        values_conv_start = self.conv(values_start)
        values_conv_start = values_conv_start.view(-1, self.seq_len, self.out_dim)
        # apply conv to end sequence
        values_end = values[:, :, self.seq_len:]
        values_conv_end = self.conv(values_end)
        values_conv_end = values_conv_end.view(-1, self.seq_len, self.out_dim)
        # combine both start and end sequence again
        values_conv = torch.cat((values_conv_start, values_conv_end), dim=1)
        values_conv = self.drop_conv(values_conv)

        keys = self.drop(self.key(input_seq))

        unnorm_weights = keys
        attn_weights = torch.softmax(unnorm_weights, dim=1)
        weighted_vals = values_conv * attn_weights
        output = torch.sum(weighted_vals, dim=1)
        return output, attn_weights

# overfitting AS FUCK
class MLP(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(200, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.drop_fc1 = nn.Dropout(0.2)
        self.drop_fc2 = nn.Dropout(0.2)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [128, 142, 4] or [128, 140, 4]
        x = F.relu(self.drop_fc1(self.fc1(d2v_feats)))
        x  = F.relu(self.drop_fc2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

# 512 neurons in hidden layers still overfits
# 256 neurons with dropout 0.2/0.5 still overfits
# 128/64 with dropout=0.5 overfits
# 64/16 d=0.5 just bad; d=0.2 overfits
# lens as input are again a fucking game changer...
class MLP2(BaseModel):
    def __init__(self, use_lens=True):
        super().__init__()
        self.use_lens = use_lens
        self.fc1 = nn.Linear(200+3, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.drop_fc1 = nn.Dropout(0.2)
        self.drop_fc2 = nn.Dropout(0.2)

    def forward(self, d2v_feats, lens):
        if not self.use_lens: lens = torch.zeros_like(lens)
        # [B, 200]
        feats = torch.cat((d2v_feats, lens), dim=1)
        x = F.relu(self.drop_fc1(self.fc1(feats)))
        # x = F.relu(self.drop_fc1(self.fc1(d2v_feats)))
        x  = F.relu(self.drop_fc2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

# ok, still a bit of overfitting, but just worse than MLP2
# AUC ~83.5 with 64, d=0.2
class MLP3(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(200+3, 16)
        self.fc2 = nn.Linear(16, 1)
        self.drop_fc1 = nn.Dropout(0.2)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        # lens = torch.zeros_like(lens)
        #d2v_feats = torch.zeros_like(d2v_feats)
        feats = torch.cat((d2v_feats, lens), dim=1)
        x = F.relu(self.drop_fc1(self.fc1(feats)))
        x = torch.sigmoid(self.fc2(x))
        return x

class MLP4(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(200+3, 1)
        self.drop_fc1 = nn.Dropout(0.35)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        feats = torch.cat((d2v_feats, lens), dim=1)
        x = torch.sigmoid(self.fc1(feats))
        # x = torch.sigmoid(self.drop_fc1(self.fc1(feats)))
        # x = (self.fc2(x))
        return x

class MLP_4_SEQ(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(400+3, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.drop_fc1 = nn.Dropout(0.2)
        self.drop_fc2 = nn.Dropout(0.2)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        feats = torch.cat((d2v_feats, lens), dim=1)
        x = F.relu(self.drop_fc1(self.fc1(feats)))
        # x = F.relu(self.drop_fc1(self.fc1(d2v_feats)))
        x  = F.relu(self.drop_fc2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x