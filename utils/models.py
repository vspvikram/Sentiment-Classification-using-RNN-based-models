"""
Sentiment Classification with RNN based models - Models File
Vikram Singh
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings, num_class):
    # def __init__(self, input_size, EMBEDDING_DIM, hidden1_dim, output_size):
        super(RecurrentNetwork, self).__init__()

        # creating layers and attributes our network needs.
        # (self, input_size, EMBEDDING_DIM, hidden1_dim, output_size)
        self.input_size = 17756#17288#17752 #17288 #input_size
        self.EMBEDDING_DIM = 100 #EMBEDDING_DIM
        self.output_size = num_class
        self.hidden1_dim = 100 #hidden1_dim

        self.embedding = nn.Embedding(self.input_size, self.EMBEDDING_DIM)

        self.rnn1 = nn.GRU(self.EMBEDDING_DIM, self.hidden1_dim)
        self.rnn2 = nn.GRU(self.hidden1_dim, self.hidden1_dim)
        # self.hid1_hid2 = nn.Linear(EMBEDDING_DIM + EMBEDDING_DIM, EMBEDDING_DIM)
        self.fc = nn.Linear(self.hidden1_dim, num_class)
        # self.softmax = nn.Softmax(dim=2)

        # raise NotImplementedError
        self.embedding.weight.data = embeddings

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        embedded = self.embedding(x.permute((1,0)))
        output, hidden = self.rnn1(embedded.float())
        # print("[Printing Hidden]")
        # print(hidden[0].shape, hidden[1].shape, output.shape)
        output2, hidden2 = self.rnn2(hidden.float())

        # print("Output shape: {0}, hidden shape: {1}".format(output.shape, hidden.shape))
        # Output shape: torch.Size([91, 128, 102]), hidden shape: torch.Size([1, 128, 102])

        out = self.fc(hidden2.float())
        # out = self.softmax(out)
        # out_soft = self.softmax(out)
        # print("Out shape: {0}".format(out.shape)) # Out shape: torch.Size([1, 128, 4])
        return out.squeeze(0)



class LSTM(nn.Module):
    def __init__(self, input_size, EMBEDDING_DIM, hidden1_dim, output_size):
        super(LSTM, self).__init__()

         # creating layers and attributes our network needs
        self.input_size = input_size
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.output_size = output_size
        self.hidden1_dim = hidden1_dim

        self.embedding = nn.Embedding(input_size, EMBEDDING_DIM)

        self.rnn1 = nn.LSTM(self.EMBEDDING_DIM, self.hidden1_dim)
        self.rnn2 = nn.LSTM(self.hidden1_dim, self.hidden1_dim)
        # self.hid1_hid2 = nn.Linear(EMBEDDING_DIM + EMBEDDING_DIM, EMBEDDING_DIM)
        self.fc = nn.Linear(self.hidden1_dim, output_size)
        # self.softmax = nn.Softmax(dim=2)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        embedded = self.embedding(x.permute((1,0)))
        # print("starting embedding here:")
        # print(x.permute((1,0)).shape) # torch.Size([91, 128])
        # print(embedded.shape) # torch.Size([91, 128, 100])

        output, hidden = self.rnn1(embedded.float())
        # print("[Printing Hidden]")
        # print(hidden[0].shape, hidden[1].shape, output.shape)
        output2, hidden2 = self.rnn2(hidden[0].float())

        # print("Output shape: {0}, hidden shape: {1}".format(output.shape, hidden.shape))
        # Output shape: torch.Size([91, 128, 102]), hidden shape: torch.Size([1, 128, 102])

        out = self.fc(hidden2[0].float())
        # out = self.softmax(out)
        # out_soft = self.softmax(out)
        # print("Out shape: {0}".format(out.shape)) # Out shape: torch.Size([1, 128, 4])
        return out.squeeze(0)



class RecurrentNetwork_old(nn.Module):
    def __init__(self, input_size, EMBEDDING_DIM, hidden1_dim, output_size):
        super(RecurrentNetwork_old, self).__init__()

        self.input_size = input_size
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.output_size = output_size
        self.hidden1_dim = hidden1_dim

        self.embedding = nn.Embedding(input_size, EMBEDDING_DIM)

        self.rnn1 = nn.RNN(self.EMBEDDING_DIM, self.hidden1_dim)
        self.rnn2 = nn.RNN(self.hidden1_dim, self.hidden1_dim)
        # self.hid1_hid2 = nn.Linear(EMBEDDING_DIM + EMBEDDING_DIM, EMBEDDING_DIM)
        self.fc = nn.Linear(self.hidden1_dim, output_size)
        # self.softmax = nn.Softmax(dim=2)

        # raise NotImplementedError

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        # print("x_shape is: {0}".format(x.shape))
        # print("hidden state1 is: {0}".format(hidden_state1.shape))

        embedded = self.embedding(x.permute((1,0)))
        # print("starting embedding here:")
        # print(x.permute((1,0)).shape) # torch.Size([91, 128])
        # print(embedded.shape) # torch.Size([91, 128, 100])

        output, hidden = self.rnn1(embedded.float())
        output2, hidden2 = self.rnn2(hidden.float())

        # print("Output shape: {0}, hidden shape: {1}".format(output.shape, hidden.shape))
        # Output shape: torch.Size([91, 128, 102]), hidden shape: torch.Size([1, 128, 102])

        out = self.fc(hidden2.float())
        # out = self.softmax(out)
        # print("Out shape: {0}".format(out.shape)) # Out shape: torch.Size([1, 128, 4])
        return out.squeeze(0)
        

    def init_hidden(self):
        hid1_init = nn.init.kaiming_uniform_(torch.empty(1, self.EMBEDDING_DIM))
        hid2_init = nn.init.kaiming_uniform_(torch.empty(1, self.EMBEDDING_DIM))
        return hid1_init, hid2_init
        # return hid1_init