import datetime
import sys

import sklearn
from tqdm import tqdm

from Attention import TransformerEncoder
import numpy as np
import os
import torch
import torch.optim as optim


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


def to_tensor(x, use_gpu):
    if use_gpu:
        return torch.from_numpy(x).cuda()
    else:
        return torch.from_numpy(x)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.trainModel = None
        self.testModel = None
        self.optimizer = None

    def load_train_data(self):
        """
        将数据载入当前类保存
        :return:
        """
        print("Reading training data...")
        self.data_word_vec = np.load(os.path.join(self.config.data_path, 'vec.npy'))  # word vec mapping
        self.data_train_word = np.load(os.path.join(self.config.data_path, 'train_word.npy'))  # one hot for each word
        self.data_train_pos1 = np.load(os.path.join(self.config.data_path, 'train_pos1.npy'))
        self.data_train_pos2 = np.load(os.path.join(self.config.data_path, 'train_pos2.npy'))
        # self.data_train_mask = np.load(os.path.join(self.data_path, 'train_mask.npy'))
        if self.config.use_bag:
            self.data_query_label = np.load(
                os.path.join(self.config.data_path, 'train_ins_label.npy'))  # 每一句话的real label
            self.data_train_label = np.load(
                os.path.join(self.config.data_path, 'train_bag_label.npy'))  # label for each bag
            self.data_train_scope = np.load(
                os.path.join(self.config.data_path, 'train_bag_scope.npy'))  # bag range in training data
        else:
            self.data_train_label = np.load(os.path.join(self.config.data_path, 'train_ins_label.npy'))
            self.data_train_scope = np.load(os.path.join(self.config.data_path, 'train_ins_scope.npy'))
        print("Finish reading")
        self.data_train_order = list(range(len(self.data_train_label)))  # bag label number, according to iteration size
        self.train_batches_num = len(self.data_train_label) // self.config.batch_size
        if len(self.data_train_label) % self.config.batch_size != 0:
            self.train_batches_num += 1

    def load_test_data(self):
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.config.data_path, 'vec.npy'))  # word id - vec mapping
        self.data_test_word = np.load(os.path.join(self.config.data_path, 'test_word.npy'))  # word idx
        self.data_test_pos1 = np.load(os.path.join(self.config.data_path, 'test_pos1.npy'))  #
        self.data_test_pos2 = np.load(os.path.join(self.config.data_path, 'test_pos2.npy'))
        # self.data_test_mask = np.load(os.path.join(self.data_path, 'test_mask.npy'))
        if self.config.use_bag:
            self.data_test_label = np.load(os.path.join(self.config.data_path, 'test_bag_label.npy'))
            self.data_test_scope = np.load(os.path.join(self.config.data_path, 'test_bag_scope.npy'))
        else:
            self.data_test_label = np.load(os.path.join(self.config.data_path, 'test_ins_label.npy'))
            self.data_test_scope = np.load(os.path.join(self.config.data_path, 'test_ins_scope.npy'))
        print("Finish reading")
        self.test_batches = len(self.data_test_label) / self.config.batch_size
        if len(self.data_test_label) % self.config.batch_size != 0:
            self.test_batches += 1

        self.total_recall = self.data_test_label[:, 1:].sum()

    def set_train_model(self, model):
        print("Initializing training model...")
        self.model = model
        self.trainModel = self.model(config=self.config)
        if self.config.pretrain_model is not None:
            self.trainModel.load_state_dict(torch.load(self.config.pretrain_model))
        if self.config.use_gpu:
            self.trainModel.cuda()
        if self.optimizer is not None:
            pass
        elif self.config.opt_method == "Adagrad" or self.config.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr=self.config.learning_rate, lr_decay=self.lr_decay,
                                           weight_decay=self.config.weight_decay)
        elif self.config.opt_method == "Adadelta" or self.config.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr=self.config.learning_rate,
                                            weight_decay=self.config.weight_decay)
        elif self.config.opt_method == "Adam" or self.config.opt_method == "adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr=self.config.learning_rate,
                                        weight_decay=self.config.weight_decay)
        else:
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr=self.config.learning_rate,
                                       weight_decay=self.config.weight_decay)
        print("Finish initializing")

    def get_train_batch(self, batch_number):
        """
        one bag is one iteration
        :param batch_number: batch_number
        """
        # self.data_train_order ids shuffled, so the batch is shuffled
        input_scope = np.take(self.data_train_scope,
                              self.data_train_order[
                              batch_number * self.config.batch_size: (batch_number + 1) * self.config.batch_size],
                              axis=0)
        index = []
        scope = [0]  # bag scopes
        for num in input_scope:
            index += list(range(num[0], num[1] + 1))  # extend one bag scope
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.batch_word = self.data_train_word[index, :]
        self.batch_pos1 = self.data_train_pos1[index, :]
        self.batch_pos2 = self.data_train_pos2[index, :]
        # self.batch_mask = self.data_train_mask[index, :]

        # bags' labels in one batch
        self.batch_label = np.take(self.data_train_label,
                                   self.data_train_order[
                                   batch_number * self.config.batch_size: (batch_number + 1) * self.config.batch_size],
                                   axis=0)
        # self.batch_attention_query = self.data_query_label[index]

        self.batch_scope = scope  # all bag scope in one batch

    def train_one_epoch(self):
        self.trainModel.embedding.word = to_tensor(self.batch_word,
                                                   self.config.use_gpu)  # assign batch word2vec to embedding.word
        self.trainModel.embedding.pos1 = to_tensor(self.batch_pos1, self.config.use_gpu)
        self.trainModel.embedding.pos2 = to_tensor(self.batch_pos2, self.config.use_gpu)
        # self.trainModel.encoder.mask = to_tensor(self.batch_mask), self.config.use_gpu
        self.trainModel.selector.scope = self.batch_scope
        # self.trainModel.selector.attention_query = to_tensor(self.batch_attention_query, self.config.use_gpu)
        self.trainModel.selector.label = to_tensor(self.batch_label, self.config.use_gpu)
        self.trainModel.classifier.label = to_tensor(self.batch_label, self.config.use_gpu)
        self.optimizer.zero_grad()
        loss, _output = self.trainModel()
        loss.backward()
        self.optimizer.step()
        for i, prediction in enumerate(_output):
            if self.batch_label[i] == 0:
                self.acc_NA.add(prediction == self.batch_label[i])
            else:
                self.acc_not_NA.add(prediction == self.batch_label[i])
            self.acc_total.add(prediction == self.batch_label[i])
        return loss.data[0]

    def train(self):
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        best_auc = 0.0
        best_p = None
        best_r = None
        best_epoch = 0
        for epoch in range(self.config.ax_epoch):
            print(('Epoch ' + str(epoch) + ' starts...'))
            # 清除上轮计数
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            # 打乱bag
            np.random.shuffle(self.data_train_order)  # shuffle the bags labels' idx
            for batch_num in range(self.train_batches_num):
                self.get_train_batch(batch_num)
                loss = self.train_one_epoch()
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write(
                    "epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (
                        epoch, batch_num, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(),
                        self.acc_total.get()))
                sys.stdout.flush()
            if (epoch + 1) % self.config.save_epoch == 0:
                print(('Epoch ' + str(epoch + 1) + ' has finished'))
                print('Saving model...')
                path = os.path.join(self.config.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
                torch.save(self.trainModel.state_dict(), path)
                print(('Have saved model to ' + path))
            if (epoch + 1) % self.config.test_epoch == 0:
                self.config.testModel = self.trainModel
                auc, pr_x, pr_y = self.test_one_epoch()
                if auc > best_auc:
                    best_auc = auc
                    best_p = pr_x
                    best_r = pr_y
                    best_epoch = epoch
        print("Finish training")
        print(("Best epoch = %d | auc = %f" % (best_epoch, best_auc)))
        print("Storing best result...")
        if not os.path.isdir(self.config.test_result_dir):
            os.mkdir(self.config.test_result_dir)
        np.save(os.path.join(self.config.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
        np.save(os.path.join(self.config.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
        print("Finish storing")

    def get_test_batch(self, batch):
        """
        :param batch: batch number
        """
        input_scope = self.config.data_test_scope[batch * self.batch_size: (batch + 1) * self.batch_size]
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.batch_word = self.config.data_test_word[index, :]
        self.batch_pos1 = self.config.data_test_pos1[index, :]
        self.batch_pos2 = self.config.data_test_pos2[index, :]
        # self.batch_mask = self.data_test_mask[index, :]
        self.batch_scope = scope

    def test_one_step(self):
        self.testModel.embedding.word = self.to_var(self.batch_word)
        self.testModel.embedding.pos1 = self.to_var(self.batch_pos1)
        self.testModel.embedding.pos2 = self.to_var(self.batch_pos2)
        # self.testModel.encoder.mask= self.to_var(self.batch_mask)
        self.testModel.selector.scope = self.batch_scope
        return self.testModel.test()

    def test_one_epoch(self):
        test_score = []
        for batch in tqdm(list(range(self.test_batches))):
            self.get_test_batch(batch)
            batch_score = self.test_one_step()
            test_score = test_score + batch_score
        test_result = []
        for i in range(len(test_score)):
            for j in range(1, len(test_score[i])):
                test_result.append([self.data_test_label[i][j], test_score[i][j]])
        test_result = sorted(test_result, key=lambda x: x[1])
        test_result = test_result[::-1]
        pr_x = []
        pr_y = []
        correct = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / self.total_recall)
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print(("auc: ", auc))
        return auc, pr_x, pr_y

    def test(self):
        best_epoch = None
        best_auc = 0.0
        best_p = None
        best_r = None
        for epoch in self.epoch_range:
            path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
            if not os.path.exists(path):
                continue
            print(("Start testing epoch %d" % (epoch)))
            self.testModel.load_state_dict(torch.load(path))
            auc, p, r = self.test_one_epoch()
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_p = p
                best_r = r
            print(("Finish testing epoch %d" % (epoch)))
        print(("Best epoch = %d | auc = %f" % (best_epoch, best_auc)))
        print("Storing best result...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
        print("Finish storing")
