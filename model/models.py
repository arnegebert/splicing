import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import copy
import torch
import numpy as np
import math
from torch.autograd import Variable


class MNISTModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class NaivePSIModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=(1,3), padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1,3), padding=0)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(156*16, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, 4, 1, 2*80)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = x.view(-1, 156*16)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


class DeepSplicingCodeSmoll(BaseModel):

    def __init__(self):
        super().__init__()

        self.fc_in = 18*8*2
        self.conv1_start = nn.Conv1d(4, 32, kernel_size=3)
        self.conv1_drop_start = nn.Dropout2d(0.2)

        self.conv2_start = nn.Conv1d(32, 8, kernel_size=3)
        self.conv2_drop_start = nn.Dropout2d(0.2)

        self.conv3_start = nn.Conv1d(8, 8, kernel_size=3)
        self.conv3_drop_start = nn.Dropout2d(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv1d(4, 32, kernel_size=3)
        self.conv1_drop_end = nn.Dropout2d(0.2)

        self.conv2_end = nn.Conv1d(32, 8, kernel_size=3)
        self.conv2_drop_end = nn.Dropout2d(0.2)

        self.conv3_end = nn.Conv1d(8, 8, kernel_size=3)
        self.conv3_drop_end = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(self.fc_in, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, data):
        # [128, 80, 4]
        start, end = data[:, 0], data[:, 1]
        start, end = start.view(-1, 4, 80), end.view(-1, 4, 80)
        x = F.relu(self.conv1_drop_start(self.conv1_start(start)))

#        x = F.max_pool1d(F.relu(self.conv1_drop_start(self.conv1_start(start))), 2)
        x = F.max_pool1d(F.relu(self.conv2_drop_start(self.conv2_start(x))), 2)
        x = F.max_pool1d(F.relu(self.conv3_drop_start(self.conv3_start(x))), 2)

        # xx = F.max_pool1d(F.relu(self.conv1_drop_end(self.conv1_end(end))), 2)
        xx = F.relu(self.conv1_drop_end(self.conv1_end(end)))
        xx = F.max_pool1d(F.relu(self.conv2_drop_end(self.conv2_end(xx))), 2)
        xx = F.max_pool1d(F.relu(self.conv3_drop_end(self.conv3_end(xx))), 2)

        feats = torch.cat((x, xx), dim=1)
        feats = feats.view(-1, self.fc_in)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))
        return y


class DSC(BaseModel):

    def __init__(self):
        super().__init__()

        self.fc_in = 15*8*2 + 3
        self.conv1_start = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_start.weight)
        self.conv1_drop_start = nn.Dropout2d(0.2)

        self.conv2_start = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_start.weight)
        self.conv2_drop_start = nn.Dropout2d(0.2)

        self.conv3_start = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_start.weight)
        self.conv3_drop_start = nn.Dropout2d(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_end.weight)
        self.conv1_drop_end = nn.Dropout2d(0.2)

        self.conv2_end = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_end.weight)
        self.conv2_drop_end = nn.Dropout2d(0.2)

        self.conv3_end = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_end.weight)
        self.conv3_drop_end = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(self.fc_in, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        # [128, 2, 142, 4] or [128, 2, 140, 4]
        start, end = seqs[:, 0], seqs[:, 1]
        if start.shape[1] == 140:
            start, end = start.view(-1, 4, 140), end.view(-1, 4, 140)
        else:
            start, end = start.view(-1, 4, 142), end.view(-1, 4, 142)

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


class GTEx_DSC(BaseModel):

    def __init__(self):
        super().__init__()

        self.fc_in = 15*8*2 + 1
        self.conv1_start = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_start.weight)
        self.conv1_drop_start = nn.Dropout2d(0.2)

        self.conv2_start = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_start.weight)
        self.conv2_drop_start = nn.Dropout2d(0.2)

        self.conv3_start = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_start.weight)
        self.conv3_drop_start = nn.Dropout2d(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_end.weight)
        self.conv1_drop_end = nn.Dropout2d(0.2)

        self.conv2_end = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_end.weight)
        self.conv2_drop_end = nn.Dropout2d(0.2)

        self.conv3_end = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_end.weight)
        self.conv3_drop_end = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(self.fc_in, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        # [128, 142, 4] or [128, 140, 4]
        start, end = seqs[:, 0], seqs[:, 1]
        if start.shape[1] == 140:
            start, end = start.view(-1, 4, 140), end.view(-1, 4, 140)
        else:
            start, end = start.view(-1, 4, 142), end.view(-1, 4, 142)

        x = F.max_pool1d(F.relu(self.conv1_drop_start(self.conv1_start(start))), 2, stride=2)
        x = F.max_pool1d(F.relu(self.conv2_drop_start(self.conv2_start(x))), 2, stride=2)
        x = F.max_pool1d(F.relu(self.conv3_drop_start(self.conv3_start(x))), 2, stride=2)

        xx = F.max_pool1d(F.relu(self.conv1_drop_end(self.conv1_end(end))), 2, stride=2)
        xx = F.max_pool1d(F.relu(self.conv2_drop_end(self.conv2_end(xx))), 2, stride=2)
        xx = F.max_pool1d(F.relu(self.conv3_drop_end(self.conv3_end(xx))), 2, stride=2)

        feats = torch.cat((x, xx), dim=1)
        feats = feats.view(-1, self.fc_in-1)
        feats = torch.cat((feats, lens.view(-1, 1)), dim=1)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))
        return y

class DSCKiller(BaseModel):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)

    def forward(self, seqs, lens):
        y = torch.sigmoid(self.lin(lens))
        return y

class DSCKillerKiller(BaseModel):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1, bias=True)

    def forward(self, seqs, lens):
        y = torch.sigmoid(self.lin(lens[:, 1].view(-1, 1)))
        return y


# v bad, 140k parameters, AUC barely getting above 0.5
class BiLSTM1(BaseModel):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(4, 4, bias=True)

        self.fc_in = 15*8*2 + 3
        self.conv1_start = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_start.weight)
        self.conv1_drop_start = nn.Dropout2d(0.2)

        self.conv2_start = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_start.weight)
        self.conv2_drop_start = nn.Dropout2d(0.2)

        self.conv3_start = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_start.weight)
        self.conv3_drop_start = nn.Dropout2d(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_end.weight)
        self.conv1_drop_end = nn.Dropout2d(0.2)

        self.conv2_end = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_end.weight)
        self.conv2_drop_end = nn.Dropout2d(0.2)

        self.conv3_end = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_end.weight)
        self.conv3_drop_end = nn.Dropout2d(0.2)
        self.lstm = nn.LSTM(input_size=self.fc_in, hidden_size=100//2, num_layers=1, bidirectional=True,
                            batch_first=True)
        self.lin_1 = nn.Linear(100, 64)
        self.lin_2 = nn.Linear(64, 1)

        self.fc1 = nn.Linear(100, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        # [128, 142, 4] or [128, 140, 4]
        start, end = seqs[:, 0], seqs[:, 1]
        start, end = start.view(-1, 4, 140), end.view(-1, 4, 140)

        x = F.max_pool1d(F.relu(self.conv1_drop_start(self.conv1_start(start))), 2, stride=2)
        x = F.max_pool1d(F.relu(self.conv2_drop_start(self.conv2_start(x))), 2, stride=2)
        x = F.max_pool1d(F.relu(self.conv3_drop_start(self.conv3_start(x))), 2, stride=2)

        xx = F.max_pool1d(F.relu(self.conv1_drop_end(self.conv1_end(end))), 2, stride=2)
        xx = F.max_pool1d(F.relu(self.conv2_drop_end(self.conv2_end(xx))), 2, stride=2)
        xx = F.max_pool1d(F.relu(self.conv3_drop_end(self.conv3_end(xx))), 2, stride=2)

        feats = torch.cat((x, xx), dim=1)
        feats = feats.view(-1, self.fc_in-3)
        feats = torch.cat((feats, lens), dim=1)
        feats = feats.view(-1, 1, self.fc_in)
        output, (h_n, c_n) = self.lstm(feats)
        y = h_n.view(-1, 100)
        y = self.drop_fc(F.relu(self.fc1(y)))
        y = torch.sigmoid(self.fc2(y))
        return y

# ok, but 113 k parameters and doesn't amaze
class BiLSTM2(BaseModel):
    def __init__(self, three_len_feats):
        super().__init__()
        self.three_feats = three_len_feats
        if self.three_feats:
            self.in_dim = 140
            self.in_fc = 103
        else:
            self.in_dim = 140
            self.in_fc = 101
        self.embedding = nn.Linear(self.in_dim, self.in_dim, bias=True)
        self.embedding2 = nn.Linear(self.in_dim, self.in_dim, bias=True)

        self.lstm1 = nn.LSTM(input_size=self.in_dim, hidden_size=50//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=self.in_dim, hidden_size=50//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.in_fc, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        # [128, 142, 4] or [128, 140, 4]
        start, end = seqs[:, 0], seqs[:, 1]
        start, end = start.view(-1, 4, self.in_dim), end.view(-1, 4, self.in_dim)

        embedding = F.relu(self.embedding(start))
        output, (h_n, c_n) = self.lstm1(embedding)
        x = c_n.view(-1, 50)

        embedding2 = F.relu(self.embedding(end))
        output, (h_n, c_n) = self.lstm2(embedding2)
        xx = c_n.view(-1, 50)

        feats = torch.cat((x, xx), dim=1)
        if self.three_feats:
            feats = torch.cat((feats, lens.view(-1, 3)), dim=1)
        else:
            feats = torch.cat((feats, lens.view(-1, 1)), dim=1)
        feats = feats.view(-1, self.in_fc)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))
        return y



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
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(200+3, 32)
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


class CancellationOfDSC(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(0+3, 16)
        self.fc2 = nn.Linear(16, 1)
        self.drop_fc1 = nn.Dropout(0.2)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        # feats = torch.cat((d2v_feats, lens), dim=1)
        x = F.relu(self.drop_fc1(self.fc1(lens)))
        x = torch.sigmoid(self.fc2(x))
        return x

class CancellationOfDSC2(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(0+3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        # feats = torch.cat((d2v_feats, lens), dim=1)
        x = F.relu(self.fc1(lens))
        x = torch.sigmoid(self.fc2(x))
        return x

class CancellationOfDSC3(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(0+3, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        # feats = torch.cat((d2v_feats, lens), dim=1)
        x = F.relu(self.fc1(lens))
        x = torch.sigmoid(self.fc2(x))
        return x


class CancellationOfDSC4(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(0+3, 1)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        x = torch.sigmoid(self.fc1(lens))
        # x = torch.sigmoid(self.drop_fc1(self.fc1(feats)))
        # x = (self.fc2(x))
        return x

class CancellationOfDSC5(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(0+3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        # feats = torch.cat((d2v_feats, lens), dim=1)
        x = F.relu(self.fc1(lens))
        x = torch.sigmoid(self.fc2(x))
        return x

class CancellationOfDSC6(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(0+3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        # feats = torch.cat((d2v_feats, lens), dim=1)
        x = (self.fc1(lens))
        x = torch.sigmoid(self.fc2(x))
        return x

class CancellationOfDSC7(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(0+3, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        # feats = torch.cat((d2v_feats, lens), dim=1)
        x = F.relu(self.fc1(lens))
        x = torch.sigmoid(self.fc2(x))
        return x

class CancellationOfDSC8(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(0+3, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, d2v_feats, lens):
        # [B, 100] input

        # [B, 200]
        # feats = torch.cat((d2v_feats, lens), dim=1)
        x = (self.fc1(lens))
        x = (self.fc2(x))
        return x


















