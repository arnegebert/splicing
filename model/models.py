import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


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
        # lens = torch.zeros_like(lens)
        start, end = seqs[:, 0], seqs[:, 1]
        if start.shape[1] == 140:
            start, end = start.view(-1, 4, 140), end.view(-1, 4, 140)
        else:
            start, end = start.view(-1, 4, 142), end.view(-1, 4, 142)

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


class BadIdea(BaseModel):

    def __init__(self):
        super().__init__()

        self.fc_in = 129*8*2 + 3
        self.conv1_start = nn.Conv2d(1, 32, kernel_size=(4, 7))
        torch.nn.init.xavier_uniform_(self.conv1_start.weight)
        self.conv1_drop_start = nn.Dropout2d(0.2)

        self.conv2_start = nn.Conv2d(32, 8, kernel_size=(1, 4))
        torch.nn.init.xavier_uniform_(self.conv2_start.weight)
        self.conv2_drop_start = nn.Dropout2d(0.2)

        self.conv3_start = nn.Conv2d(8, 8, kernel_size=(1, 3))
        torch.nn.init.xavier_uniform_(self.conv3_start.weight)
        self.conv3_drop_start = nn.Dropout2d(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv2d(1, 32, kernel_size=(4,7))
        torch.nn.init.xavier_uniform_(self.conv1_end.weight)
        self.conv1_drop_end = nn.Dropout2d(0.2)

        self.conv2_end = nn.Conv2d(32, 8, kernel_size=(1, 4))
        torch.nn.init.xavier_uniform_(self.conv2_end.weight)
        self.conv2_drop_end = nn.Dropout2d(0.2)

        self.conv3_end = nn.Conv2d(8, 8, kernel_size=(1, 3))
        torch.nn.init.xavier_uniform_(self.conv3_end.weight)
        self.conv3_drop_end = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(self.fc_in, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        # [128, 2, 142, 4] or [128, 2, 140, 4]
        # lens = torch.zeros_like(lens)
        start, end = seqs[:, 0], seqs[:, 1]
        if start.shape[1] == 140:
            start, end = start.view(-1, 1, 4, 140), end.view(-1, 1, 4, 140)
        else:
            start, end = start.view(-1, 1, 4, 142), end.view(-1, 1, 4, 142)


        x = (F.relu(self.conv1_drop_start(self.conv1_start(start))))
        x = (F.relu(self.conv2_drop_start(self.conv2_start(x))))
        x = (F.relu(self.conv3_drop_start(self.conv3_start(x))))

        xx = (F.relu(self.conv1_drop_end(self.conv1_end(end))))
        xx = (F.relu(self.conv2_drop_end(self.conv2_end(xx))))
        xx = (F.relu(self.conv3_drop_end(self.conv3_end(xx))))

        # x = F.max_pool2d(F.relu(self.conv1_drop_start(self.conv1_start(start))), 2, stride=2)
        # x = F.max_pool2d(F.relu(self.conv2_drop_start(self.conv2_start(x))), 2, stride=2)
        # x = F.max_pool2d(F.relu(self.conv3_drop_start(self.conv3_start(x))), 2, stride=2)
        #
        # xx = F.max_pool2d(F.relu(self.conv1_drop_end(self.conv1_end(end))), 2, stride=2)
        # xx = F.max_pool2d(F.relu(self.conv2_drop_end(self.conv2_end(xx))), 2, stride=2)
        # xx = F.max_pool2d(F.relu(self.conv3_drop_end(self.conv3_end(xx))), 2, stride=2)

        feats = torch.cat((x, xx), dim=1)
        feats = feats.view(-1, self.fc_in-3)
        feats = torch.cat((feats, lens), dim=1)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))

        return y


class BadIdeaWithPool(BaseModel):

    def __init__(self):
        super().__init__()

        self.fc_in = 15*8*2 + 3
        self.conv1_start = nn.Conv2d(1, 32, kernel_size=(4, 7))
        torch.nn.init.xavier_uniform_(self.conv1_start.weight)
        self.conv1_drop_start = nn.Dropout2d(0.2)

        self.conv2_start = nn.Conv2d(32, 8, kernel_size=(1, 4))
        torch.nn.init.xavier_uniform_(self.conv2_start.weight)
        self.conv2_drop_start = nn.Dropout2d(0.2)

        self.conv3_start = nn.Conv2d(8, 8, kernel_size=(1, 3))
        torch.nn.init.xavier_uniform_(self.conv3_start.weight)
        self.conv3_drop_start = nn.Dropout2d(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv2d(1, 32, kernel_size=(4,7))
        torch.nn.init.xavier_uniform_(self.conv1_end.weight)
        self.conv1_drop_end = nn.Dropout2d(0.2)

        self.conv2_end = nn.Conv2d(32, 8, kernel_size=(1, 4))
        torch.nn.init.xavier_uniform_(self.conv2_end.weight)
        self.conv2_drop_end = nn.Dropout2d(0.2)

        self.conv3_end = nn.Conv2d(8, 8, kernel_size=(1, 3))
        torch.nn.init.xavier_uniform_(self.conv3_end.weight)
        self.conv3_drop_end = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(self.fc_in, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        # [128, 2, 142, 4] or [128, 2, 140, 4]
        # lens = torch.zeros_like(lens)
        start, end = seqs[:, 0], seqs[:, 1]
        if start.shape[1] == 140:
            start, end = start.view(-1, 1, 4, 140), end.view(-1, 1, 4, 140)
        else:
            start, end = start.view(-1, 1, 4, 142), end.view(-1, 1, 4, 142)


        # x = (F.relu(self.conv1_drop_start(self.conv1_start(start))))
        # x = (F.relu(self.conv2_drop_start(self.conv2_start(x))))
        # x = (F.relu(self.conv3_drop_start(self.conv3_start(x))))
        #
        # xx = (F.relu(self.conv1_drop_end(self.conv1_end(end))))
        # xx = (F.relu(self.conv2_drop_end(self.conv2_end(xx))))
        # xx = (F.relu(self.conv3_drop_end(self.conv3_end(xx))))

        x = F.max_pool2d(F.relu(self.conv1_drop_start(self.conv1_start(start))), (1,2), stride=2)
        x = F.max_pool2d(F.relu(self.conv2_drop_start(self.conv2_start(x))), (1,2), stride=2)
        x = F.max_pool2d(F.relu(self.conv3_drop_start(self.conv3_start(x))), (1,2), stride=2)

        xx = F.max_pool2d(F.relu(self.conv1_drop_end(self.conv1_end(end))), (1,2), stride=2)
        xx = F.max_pool2d(F.relu(self.conv2_drop_end(self.conv2_end(xx))), (1,2), stride=2)
        xx = F.max_pool2d(F.relu(self.conv3_drop_end(self.conv3_end(xx))), (1,2), stride=2)

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
        self.conv1_drop_bef_start = nn.Dropout2d(0.2)

        self.conv2_bef_start = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_bef_start.weight)
        self.conv2_drop_bef_start = nn.Dropout2d(0.2)

        self.conv3_bef_start = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_bef_start.weight)
        self.conv3_drop_bef_start = nn.Dropout2d(0.2)

        # start sequence
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

        # end conv blocks here...
        self.conv1_after_end = nn.Conv1d(4, 32, kernel_size=7)
        torch.nn.init.xavier_uniform_(self.conv1_after_end.weight)
        self.conv1_drop_after_end = nn.Dropout2d(0.2)

        self.conv2_after_end = nn.Conv1d(32, 8, kernel_size=4)
        torch.nn.init.xavier_uniform_(self.conv2_after_end.weight)
        self.conv2_drop_after_end = nn.Dropout2d(0.2)

        self.conv3_after_end = nn.Conv1d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3_after_end.weight)
        self.conv3_drop_after_end = nn.Dropout2d(0.2)

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
        self.embedding_dim = 140
        self.LSTM_dim = 50
        self.in_dim = 140
        self.dim_fc = 64
        self.dropout_prob = 0.5
        self.lstm_layer = 2
        self.lstm_dropout = 0.2
        self.in_fc = 2 * self.LSTM_dim * self.lstm_layer + (3 if self.three_feats else 1)


        self.embedding = nn.Linear(self.in_dim, self.embedding_dim, bias=True)
        # self.embedding2 = nn.Linear(self.in_dim, self.in_dim, bias=True)

        self.lstm1 = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.lstm2 = nn.LSTM(input_size=self.in_dim, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.fc1 = nn.Linear(self.in_fc, self.dim_fc)
        self.drop_fc = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.dim_fc, 1)

    def forward(self, seqs, lens):
        # [128, 142, 4] or [128, 140, 4]
        # lens = torch.zeros_like(lens)
        start, end = seqs[:, 0], seqs[:, 1]
        start, end = start.view(-1, 4, self.in_dim), end.view(-1, 4, self.in_dim)

        embedding = F.relu(self.embedding(start))
        output, (h_n, c_n) = self.lstm1(embedding)
        # todo -- might need to add lstm layers for grid search here
        x = c_n.view(-1, self.LSTM_dim*self.lstm_layer)

        embedding2 = F.relu(self.embedding(end))
        output, (h_n, c_n) = self.lstm2(embedding2)
        xx = c_n.view(-1, self.LSTM_dim)

        feats = torch.cat((x, xx), dim=1)
        if self.three_feats:
            feats = torch.cat((feats, lens.view(-1, 3)), dim=1)
        else:
            feats = torch.cat((feats, lens.view(-1, 1)), dim=1)
        feats = feats.view(-1, self.in_fc*self.lstm_layer)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))
        return y

# 85.6
class BiLSTM3(BaseModel):
    def __init__(self, three_len_feats):
        super().__init__()
        self.three_feats = three_len_feats
        self.embedding_dim = 140
        self.LSTM_dim = 50
        self.in_dim = 140
        self.dim_fc = 64
        self.dropout_prob = 0.5
        self.lstm_layer = 1
        self.lstm_dropout = 0.2
        self.in_fc = 2 * self.LSTM_dim * self.lstm_layer + (3 if self.three_feats else 1)


        self.embedding = nn.Linear(self.in_dim, self.embedding_dim, bias=True)
        # self.embedding2 = nn.Linear(self.in_dim, self.in_dim, bias=True)

        self.lstm1 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.lstm2 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.fc1 = nn.Linear(self.in_fc, self.dim_fc)
        self.drop_fc = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.dim_fc, 1)

    def forward(self, seqs, lens):
        # [128, 142, 4] or [128, 140, 4]
        # lens = torch.zeros_like(lens)
        start, end = seqs[:, 0], seqs[:, 1]
        start, end = start.view(-1, 4, self.in_dim), end.view(-1, 4, self.in_dim)

        # embedding = F.relu(self.embedding(start))
        embedding = F.relu(self.embedding(start))
        # currently treat the 140-input as dimension, but shouldn't
        # just want a dense mapping from sparse 4-d to dense 4-d
        output, (h_n, c_n) = self.lstm1(embedding.view(-1, 140, 4))
        # todo -- might need to add lstm layers for grid search here
        # output = [256, 140, 2*50]  // 256, 4, 50???
        x = h_n.view(-1, self.LSTM_dim*self.lstm_layer)
        # want: a 50-dimensional output for the complete sequence (140x4)
        # currently have: a 50-dimensional output for each element in the sequence

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

# 1 layers = seems best, as high as 85.8, fastest to train
# 2 layers = slow to train, stop at 85 after 63 epochs
# 3 layers = stopped after 20 epochs took like 20 min and it was still only at 82
class BiLSTM4(BaseModel):
    def __init__(self, three_len_feats):
        super().__init__()
        self.three_feats = three_len_feats
        self.LSTM_dim = 50
        self.seq_length = 140
        self.dim_fc = 64
        self.dropout_prob = 0.5
        self.lstm_layer = 1
        self.lstm_dropout = 0.2
        self.in_fc = 2 * self.LSTM_dim * self.lstm_layer + (3 if self.three_feats else 1)

        self.embedding = nn.Linear(4, 4, bias=True)
        # self.embedding2 = nn.Linear(self.in_dim, self.in_dim, bias=True)

        self.lstm1 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.lstm2 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.fc1 = nn.Linear(self.in_fc, self.dim_fc)
        self.drop_fc = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.dim_fc, 1)

    def forward(self, seqs, lens):
        # [256, 142, 4] or [256, 140, 4]
        # lens = torch.zeros_like(lens)
        start, end = seqs[:, 0], seqs[:, 1]
        start, end = start.view(-1, self.seq_length, 4), end.view(-1, self.seq_length, 4)
        embedding = F.relu(self.embedding(start))
        # currently treat the 140-input as dimension, but shouldn't
        # just want a dense mapping from sparse 4-d to dense 4-d
        output, (h_n, c_n) = self.lstm1(embedding)
        # todo -- might need to add lstm layers for grid search here
        # output = [256, 140, 2*50]  // 256, 4, 50???
        x = h_n.view(-1, self.LSTM_dim*self.lstm_layer)
        # want: a 50-dimensional output for the complete sequence (140x4)
        # currently have: a 50-dimensional output for each element in the sequence

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
    def __init__(self, LSTM_dim=50, fc_dim=64, attn_dim=50, conv_size=5, n_heads=1, three_len_feats=True):
        super().__init__()
        assert conv_size % 2 == 1, "Only uneven convolution sizes allowed for reasons of padding"
        self.three_feats = three_len_feats
        self.conv_size = conv_size
        self.n_heads = n_heads
        self.LSTM_dim = LSTM_dim
        self.seq_length = 140
        self.dim_fc = fc_dim
        self.dropout_prob = 0.5
        self.lstm_layer = 1 # not changing this; too long training times already, really need to put my foot down here
        # if I want to finish this thesis in time
        self.lstm_dropout = 0.2
        self.attn_dim = attn_dim
        self.in_fc = attn_dim * n_heads + (3 if self.three_feats else 1)

        self.embedding = nn.Linear(4, 4, bias=True)

        self.lstm1 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.lstm2 = nn.LSTM(input_size=4, hidden_size=self.LSTM_dim//2, num_layers=self.lstm_layer,
                             bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        # self.attention = AttentionBlock(self.LSTM_dim, self.attn_dim)
        # self.attention = AttentionBlockWithoutQuery(self.LSTM_dim, self.attn_dim)
        # self.attention = AttentionBlockWithHeads(self.LSTM_dim, self.attn_dim, self.n_heads)
        # self.attention = AttentionBlockWithConv(self.seq_length, self.LSTM_dim, self.attn_dim, self.conv_size)
        self.attention = AttentionBlockWithConvAndSequenceSeparation(self.seq_length, self.LSTM_dim, self.attn_dim, self.conv_size)

        self.fc1 = nn.Linear(self.in_fc, self.dim_fc)
        self.drop_fc = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.dim_fc, 1)


    def forward(self, seqs, lens):
        # [256, 142, 4] or [256, 140, 4]
        # lens = torch.zeros_like(lens)
        start, end = seqs[:, 0], seqs[:, 1]
        start, end = start.view(-1, self.seq_length, 4), end.view(-1, self.seq_length, 4)
        embedding = F.relu(self.embedding(start))
        # currently treat the 140-input as dimension, but shouldn't
        # just want a dense mapping from sparse 4-d to dense 4-d
        output1, (h_n, c_n) = self.lstm1(embedding)
        # output = [256, 140, 2*50]  // 256, 4, 50???

        embedding2 = F.relu(self.embedding(end))
        output2, (h_n, c_n) = self.lstm2(embedding2)

        feats = torch.cat((output1, output2), dim=1)

        # (batch, attn_dimension)
        attn_seq, ws = self.attention(feats)
        class_feats = torch.cat((attn_seq, lens), dim=1)
        y = self.drop_fc(F.relu(self.fc1(class_feats)))
        y = torch.sigmoid(self.fc2(y))
        return y

class AttentionBlock(BaseModel):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.key = torch.nn.Linear(in_dim, out_dim)
        self.value = torch.nn.Linear(in_dim, out_dim)
        # I just have one query independent of sequence length
        self.query = torch.nn.Linear(1, out_dim, bias=False) # perhaps other way to express this

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        values = self.value(input_seq)
        keys = self.key(input_seq)
        # since same query for each element in batch
        queries = self.query.weight.repeat(batch_size, 1, 1) # attn_weights[0, 5] * values[0, 5] = weighted_vals[0, 5]
        unnorm_weights = torch.bmm(keys, queries)
        attn_weights = torch.softmax(unnorm_weights, dim=1)
        weighted_vals = values * attn_weights
        output = torch.sum(weighted_vals, dim=1)
        return output, attn_weights

class AttentionBlockWithoutQuery(BaseModel):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.key = torch.nn.Linear(in_dim, 1)
        self.value = torch.nn.Linear(in_dim, out_dim)
        # I just have one query independent of sequence length

    def forward(self, input_seq):
        values = self.value(input_seq)
        unnorm_weights = self.key(input_seq)

        attn_weights = torch.softmax(unnorm_weights, dim=1)
        weighted_vals = values * attn_weights
        output = torch.sum(weighted_vals, dim=1)
        return output, attn_weights

class AttentionBlockWithHeads(BaseModel):
    def __init__(self, in_dim, out_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.keys = clones(torch.nn.Linear(in_dim, out_dim), n_heads)
        self.values = clones(torch.nn.Linear(in_dim, out_dim), n_heads)
        self.queries = clones(torch.nn.Linear(1, out_dim, bias=False), n_heads)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        outputs, attn_ws = [], []

        for i in range(self.n_heads):
            values = self.values[i](input_seq)
            keys = self.keys[i](input_seq)
            # since same query for each element in batch
            queries = self.queries[i].weight.repeat(batch_size, 1, 1) # attn_weights[0, 5] * values[0, 5] = weighted_vals[0, 5]
            unnorm_weights = torch.bmm(keys, queries)
            attn_w = torch.softmax(unnorm_weights, dim=1)
            weighted_vals = values * attn_w
            output = torch.sum(weighted_vals, dim=1)
            outputs.append(output)
            attn_ws.append(attn_w)

        output = torch.cat(outputs, dim=1)
        return output, attn_ws

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class AttentionBlockWithoutQueryWithHeads(BaseModel):
    def __init__(self, in_dim, out_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.keys, self.values = [], []
        for _ in range(n_heads):
            self.keys.append(torch.nn.Linear(in_dim, 1))
            self.values.append(torch.nn.Linear(in_dim, out_dim))

    def forward(self, input):
        outputs, attn_ws = [], []
        for i in range(self.n_heads):
            values = self.values[i](input)
            unnorm_weights = self.keys[i](input)

            attn_weight = torch.softmax(unnorm_weights, dim=1)
            weighted_vals = values * attn_weight
            output = torch.sum(weighted_vals, dim=1)
            outputs.append(output)
            attn_ws.append(attn_weight)
        return outputs, attn_ws

class AttentionBlockWithConv(BaseModel):
    def __init__(self, seq_len, in_dim, out_dim, kernel_size):
        super().__init__()
        self.comb_seq_len = 2* seq_len
        self.out_dim = out_dim

        self.key = torch.nn.Linear(in_dim, out_dim)
        self.value = torch.nn.Linear(in_dim, out_dim)
        # I just have one query independent of sequence length
        self.query = torch.nn.Linear(1, out_dim, bias=False)
        self.conv = torch.nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size//2)

    # todo: compare between only concatenating sequences after and before convolution
    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        values = self.value(input_seq)
        values_conv = self.conv(values.view(-1, self.out_dim, self.comb_seq_len))
        values_conv = values_conv.view(-1, self.comb_seq_len, self.out_dim)

        keys = self.key(input_seq)
        keys_conv = self.conv(keys.view(-1, self.out_dim, self.comb_seq_len))
        keys_conv = keys_conv.view(-1, self.comb_seq_len, self.out_dim)

        # since same query for each element in batch
        queries = self.query.weight.repeat(batch_size, 1, 1)
        unnorm_weights = torch.bmm(keys_conv, queries)
        attn_weights = torch.softmax(unnorm_weights, dim=1)
        weighted_vals = values_conv * attn_weights
        output = torch.sum(weighted_vals, dim=1)
        return output, attn_weights

class AttentionBlockWithConvAndSequenceSeparation(BaseModel):
    def __init__(self, seq_len, in_dim, out_dim, kernel_size):
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim

        self.key = torch.nn.Linear(in_dim, out_dim)
        self.value = torch.nn.Linear(in_dim, out_dim)
        # I just have one query independent of sequence length
        self.query = torch.nn.Linear(1, out_dim, bias=False)
        self.drop = torch.nn.Dropout(0.5)
        self.drop_conv = torch.nn.Dropout2d(0.5)
        self.bn_values = torch.nn.BatchNorm1d(out_dim)
        self.bn_keys = torch.nn.BatchNorm1d(out_dim)
        self.conv = torch.nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size//2)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        values = self.drop(self.value(input_seq))
        values_start = values[:, :self.seq_len]
        values_start = self.bn_values(values_start.view(-1, self.out_dim, self.seq_len))
        values_conv_start = self.conv(values_start)
        values_conv_start = values_conv_start.view(-1, self.seq_len, self.out_dim)

        values_end = values[:, self.seq_len:]
        values_end = self.bn_values(values_end.view(-1, self.out_dim, self.seq_len))
        values_conv_end = self.conv(values_end)
        values_conv_end = values_conv_end.view(-1, self.seq_len, self.out_dim)
        values_conv = torch.cat((values_conv_start, values_conv_end), dim=1)
        values_conv = self.drop_conv(values_conv)


        keys = self.drop(self.key(input_seq))
        keys_start = keys[:, :self.seq_len]
        keys_start = self.bn_keys(keys_start.view(-1, self.out_dim, self.seq_len))
        keys_conv_start = self.conv(keys_start)
        keys_conv_start = keys_conv_start.view(-1, self.seq_len, self.out_dim)
        keys_end = keys[:, self.seq_len:]

        keys_end = self.bn_keys(keys_end.view(-1, self.out_dim, self.seq_len))
        keys_conv_end = self.conv(keys_end)
        keys_conv_end = keys_conv_end.view(-1, self.seq_len, self.out_dim)
        keys_conv = torch.cat((keys_conv_start, keys_conv_end), dim=1)
        keys_conv = self.drop_conv(keys_conv)

        # since same query for each element in batch
        queries = self.query.weight.repeat(batch_size, 1, 1)
        unnorm_weights = torch.bmm(keys_conv, queries)
        attn_weights = torch.softmax(unnorm_weights, dim=1)
        weighted_vals = values_conv * attn_weights
        output = torch.sum(weighted_vals, dim=1)
        return output, attn_weights

# ok, but 113 k parameters and doesn't amaze
class BiLSTM2_4_SEQ(BaseModel):
    def __init__(self, three_len_feats):
        super().__init__()
        self.three_feats = three_len_feats
        if self.three_feats:
            self.in_dim = 140
            self.in_fc = 4 * 50 + 3
        else:
            self.in_dim = 140
            self.in_fc = 4 * 50 + 1
        self.embedding = nn.Linear(self.in_dim, self.in_dim, bias=True)
        # self.embedding2 = nn.Linear(self.in_dim, self.in_dim, bias=True)

        self.lstm0 = nn.LSTM(input_size=self.in_dim, hidden_size=50//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.lstm1 = nn.LSTM(input_size=self.in_dim, hidden_size=50//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=self.in_dim, hidden_size=50//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.lstm3 = nn.LSTM(input_size=self.in_dim, hidden_size=50//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.in_fc, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        # [128, 142, 4] or [128, 140, 4]
        # lens = torch.zeros_like(lens)
        before_start, start, end, after_end = seqs[:, 0], seqs[:, 1], seqs[:, 2], seqs[:, 3]
        start, end = start.view(-1, 4, self.in_dim), end.view(-1, 4, self.in_dim)
        before_start, after_end = before_start.view(-1, 4, self.in_dim), after_end.view(-1, 4, self.in_dim)

        embedding0 = F.relu(self.embedding(before_start))
        output, (h_n, c_n) = self.lstm0(embedding0)
        x = c_n.view(-1, 50)

        embedding = F.relu(self.embedding(start))
        output, (h_n, c_n) = self.lstm1(embedding)
        xx = c_n.view(-1, 50)

        embedding2 = F.relu(self.embedding(end))
        output, (h_n, c_n) = self.lstm2(embedding2)
        xxx = c_n.view(-1, 50)

        embedding3 = F.relu(self.embedding(after_end))
        output, (h_n, c_n) = self.lstm3(embedding3)
        xxxx = c_n.view(-1, 50)

        feats = torch.cat((x, xx, xxx, xxxx), dim=1)
        if self.three_feats:
            feats = torch.cat((feats, lens.view(-1, 3)), dim=1)
        else:
            feats = torch.cat((feats, lens.view(-1, 1)), dim=1)
        feats = feats.view(-1, self.in_fc)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))
        return y

# ok, but 113 k parameters and doesn't amaze
class BiLSTM3_4_SEQ(BaseModel):
    def __init__(self, three_len_feats):
        super().__init__()
        self.three_feats = three_len_feats
        if self.three_feats:
            self.in_dim = 140
            self.in_fc = 4 * 24 + 3
        else:
            self.in_dim = 140
            self.in_fc = 4 * 24 + 1
        self.embedding_dim = 35
        self.embedding = nn.Linear(self.in_dim, self.embedding_dim, bias=True)

        self.lstm0 = nn.LSTM(input_size=self.embedding_dim, hidden_size=24//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.lstm1 = nn.LSTM(input_size=self.embedding_dim, hidden_size=24//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=self.embedding_dim, hidden_size=24//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.lstm3 = nn.LSTM(input_size=self.embedding_dim, hidden_size=24//2, num_layers=1, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.in_fc, 16)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, seqs, lens):
        # [128, 142, 4] or [128, 140, 4]
        # lens = torch.zeros_like(lens)
        before_start, start, end, after_end = seqs[:, 0], seqs[:, 1], seqs[:, 2], seqs[:, 3]
        start, end = start.view(-1, 4, self.in_dim), end.view(-1, 4, self.in_dim)
        before_start, after_end = before_start.view(-1, 4, self.in_dim), after_end.view(-1, 4, self.in_dim)

        embedding0 = F.relu(self.embedding(before_start))
        output, (h_n, c_n) = self.lstm0(embedding0)
        x = c_n.view(-1, 24)

        embedding = F.relu(self.embedding(start))
        output, (h_n, c_n) = self.lstm1(embedding)
        xx = c_n.view(-1, 24)

        embedding2 = F.relu(self.embedding(end))
        output, (h_n, c_n) = self.lstm2(embedding2)
        xxx = c_n.view(-1, 24)

        embedding3 = F.relu(self.embedding(after_end))
        output, (h_n, c_n) = self.lstm3(embedding3)
        xxxx = c_n.view(-1, 24)

        feats = torch.cat((x, xx, xxx, xxxx), dim=1)
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