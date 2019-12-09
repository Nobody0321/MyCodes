import pandas as pd
import jieba
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def padding(sequences, pad = 0, maxlen = 50):
    for i in range(len(sequences)):
        if len(sequences[i]) > maxlen:
            sequences[i] = sequences[i][:maxlen]
        else:
            sequences[i] =  [pad] * (maxlen-len(sequences[i])) + sequences[i]
    return sequences


class ClassfierModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size = 53039, tagset_size=2):
        super(ClassfierModel, self).__init__()
        print('Build model...')
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, inputs):
        embeddings = self.word_embeddings(inputs)
        # print("embeddings", embeddings)
        lstm_out, _ = self.lstm(embeddings.view(len(inputs),1,-1))
        # print("lstm_out",lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(inputs),-1))
        # print("tag_space", tag_space)
        tag_scores = F.log_softmax(tag_space, dim=1)
        # print("tag_scores",tag_scores)
        return tag_scores

    def init_hidden(self, hidden_dim):
        return(torch.zeros(1, 1, hidden_dim),
               torch.zeros(1, 1, hidden_dim))
    

def main():


    x = np.array(list(pn['wordId']))[::2] #训练集
    y = np.array(list(pn['sentiment']))[::2]

    xt = np.array(list(pn['wordId']))[1::4] #测试集
    yt = np.array(list(pn['sentiment']))[1::4]
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ClassfierModel(embedding_dim= 50, hidden_dim =10 ,vocab_size = 53039, tagset_size = 2)
    model.to(device)
    loss_function = nn.BCELoss()  # nagetive log likehood
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print("Start Training")
    for epoch in range(300):
        for wordids, target in zip(x,y) :
            # cause pytorch accumulates gradients
            # we clear grads every time before training
            model.zero_grad()

            wordids = wordids[1:-1].split(',')
            wordids = [int(wordid) for wordid in wordids]
            # input and forward
            tag_scores = model(torch.Tensor(wordids).long().cuda())
            _, tag_score = torch.max(tag_scores)
            print(tag_score)
            print(tag_score.shape)
            # caculate loss,gradients, 
            loss = loss_function(tag_score.data, torch.Tensor(target).cuda())
            loss.backward()

            optimizer.step()  # update optimizers's parameters
    print("Training Done")
    
    print("Start Testing")
    eval_loss = 0.
    eval_acc = 0.
    with torch.no_grad():

        for words, target in zip(xt,yt):
            batch_x, batch_y = torch.Tensor(words).long().cuda(), torch.Tensor(target).long().cuda()
            out = model(batch_x)
            loss = loss_function(out, batch_y)
            eval_loss += loss.data[0]
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.data[0]

        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            xt)), eval_acc / (len(xt))))


if  __name__ == "__main__":
    main()