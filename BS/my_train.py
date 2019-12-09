import numpy as np
import os
import torch
from .models import BLSTM_ATT


data_path = ".mini_dataset-300"
use_bag = True

net = BLSTM_ATT()
batch_size = 5

def to_var(x):
    return torch.from_numpy(x).cuda()




    # load_all_train_data(self):
print("Reading training data...")
data_word_vec = np.load(os.path.join(data_path, 'vec.npy'))
data_train_word = np.load(os.path.join(data_path, 'train_word.npy'))
data_train_pos1 = np.load(os.path.join(data_path, 'train_pos1.npy'))
data_train_pos2 = np.load(os.path.join(data_path, 'train_pos2.npy'))
data_train_mask = np.load(os.path.join(data_path, 'train_mask.npy'))
if use_bag:
    data_query_label = np.load(os.path.join(data_path, 'train_ins_label.npy'))
    data_train_label = np.load(os.path.join(data_path, 'train_bag_label.npy'))
    data_train_scope = np.load(os.path.join(data_path, 'train_bag_scope.npy'))

print("Finish reading")
train_order = list(range(len(data_train_label)))
train_batches = len(data_train_label) / batch_size
if len(data_train_label) % batch_size != 0:
    train_batches += 1

net.embedding.word = to_var(batch_word)  # assign batch word2vec to embedding.word
net.embedding.pos1 = to_var(batch_pos1)
net.embedding.pos2 = to_var(batch_pos2)
# trainModel.encoder.mask = to_var(batch_mask)
trainModel.selector.scope = batch_scope
# trainModel.selector.attention_query = to_var(batch_attention_query)
trainModel.selector.label = to_var(batch_label)
trainModel.classifier.label = to_var(batch_label)
optimizer.zero_grad()
loss, _output = trainModel()
loss.backward()
optimizer.step()
for i, prediction in enumerate(_output):
    if batch_label[i] == 0:
        acc_NA.add(prediction == batch_label[i])
    else:
        acc_not_NA.add(prediction == batch_label[i])
    acc_total.add(prediction == batch_label[i])
