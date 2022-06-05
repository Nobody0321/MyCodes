import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import math


class Data_Loader():

    def __init__(self, data_dir, config):
        super().__init__()
        self.config.data_dir = data_dir
        self.config = config

    def load_train_data(self):
        self.data_word_vec = np.load(os.path.join(self.config.data_dir, 'vec.npy'))
        self.data_train_word = np.load(os.path.join(self.config.data_dir, 'train_word.npy'))

        self.data_train_pos0 = np.load(os.path.join(self.config.data_dir, 'train_pos0.npy'))
        self.data_train_pos1 = np.load(os.path.join(self.config.data_dir, 'train_pos1.npy'))
        self.data_train_pos2 = np.load(os.path.join(self.config.data_dir, 'train_pos2.npy'))
        self.data_train_mask = np.load(os.path.join(self.config.data_dir, 'train_mask.npy'))

        self.data_query_label = np.load(os.path.join(self.config.data_dir, 'train_ins_label.npy'))
        self.data_train_label = np.load(os.path.join(self.config.data_dir, 'train_bag_label.npy'))
        self.data_train_scope = np.load(os.path.join(self.config.data_dir, 'train_bag_scope.npy'))


    
    def load_test_data(self):
        self.data_word_vec = np.load(os.path.join(self.config.data_dir, 'vec.npy'))
        self.data_test_word = np.load(os.path.join(self.config.data_dir, 'test_word.npy'))
        self.data_test_pos0 = np.load(os.path.join(self.config.data_dir, 'test_pos0.npy'))

        self.data_test_pos1 = np.load(os.path.join(self.config.data_dir, 'test_pos1.npy'))
        self.data_test_pos2 = np.load(os.path.join(self.config.data_dir, 'test_pos2.npy'))
        self.data_test_mask = np.load(os.path.join(self.config.data_dir, 'test_mask.npy'))
        self.data_test_label = np.load(os.path.join(self.config.data_dir, 'test_ins_label.npy'))
        self.data_test_scope = np.load(os.path.join(self.config.data_dir, 'test_ins_scope.npy'))

        self.test_order = list(range(len(self.data_test_label)))
        self.num_test_batches = math.ceil(len(self.data_test_label) / self.batch_size)

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass
        

class Model_Trainer():
    def __init__(self, train_config, model, data_loader):
        super().__init__()
        self.config = train_config
        self.model = model
        self.data_loader = data_loader
        self.batch_size = self.config.batch_size

        self.train_order = list(range(len(self.data_loader.data_train_label)))
        self.num_train_batches = math.ceil(len(self.data_loader.data_train_label) / self.batch_size)
        self.test_order = list(range(len(self.data_loader.data_test_label)))
        self.num_test_batches = math.ceil(len(self.data_loader.data_test_label) / self.batch_size)

    def load_train_data_one_batch(self, batch_num):
        start_ix = batch_num * self.batch_size
        # input sen scope list for this batch
        # sen scope means continous sens' range of the same label
        input_scope = np.take(self.data_loader.data_train_scope, self.train_order[start_ix: start_ix + self.batch_size], axis=0)
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        batch_word = self.data_loader.data_train_word[index, :]
        batch_pos0 = self.data_loader.data_train_pos0[index, :]
        batch_pos1 = self.data_loader.data_train_pos1[index, :]
        batch_pos2 = self.data_loader.data_train_pos2[index, :]
        batch_mask = self.data_loader.data_train_mask[index, :]
        batch_label = np.take(self.data_loader.data_train_label,
                                   self.data_loader.train_order[start_ix: start_ix + self.batch_size], axis=0)
        batch_attention_query = self.data_loader.data_query_label[index]
        batch_scope = scope
    
        return [batch_word, batch_pos0, batch_pos1, batch_pos2, batch_mask, batch_attention_query, batch_scope], batch_label
    
    def load_train_data_one_batch(self, batch_num):
        start_ix = batch_num * self.batch_size
        # input sen scope list for this batch
        # sen scope means continous sens' range of the same label
        input_scope = np.take(self.data_loader.data_test_scope, self.test_order[start_ix: start_ix + self.batch_size], axis=0)
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        batch_word = self.data_loader.data_test_word[index, :]
        batch_pos0 = self.data_loader.data_test_pos0[index, :]
        batch_pos1 = self.data_loader.data_test_pos1[index, :]
        batch_pos2 = self.data_loader.data_test_pos2[index, :]
        batch_mask = self.data_loader.data_test_mask[index, :]
        batch_label = np.take(self.data_loader.data_test_label,
                                   self.data_loader.test_order[start_ix: start_ix + self.batch_size], axis=0)
        batch_attention_query = self.data_loader.data_query_label[index]
        batch_scope = scope
    
        return [batch_word, batch_pos0, batch_pos1, batch_pos2, batch_mask, batch_attention_query, batch_scope], batch_label
    
    def train_one_batch(self):
        for epoch in range(self.config.train_start_epoch, self.config.train_max_epoch):
            train_data, train_label = self.load_train_data_one_batch()
            loss, logits = model.train_on_batch(x=train_data, y=train_label)
            for i, prediction in enumerate(logits):
                if batch_label[i] == 0:
                    acc_NA.add(prediction == batch_label[i])
                else:
                    acc_not_NA.add(prediction == batch_label[i])
                acc_total.add(prediction == batch_label[i])
        return loss.eval()

    def test_one_batch(self):
        test_data, test_label = self.load_test_data_one_batch()
        loss, logits = model.test_on_batch(x=test_data, y=test_label)
        return loss, logits
       
    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_auc = 0.0
        best_p = None
        best_r = None
        best_epoch = 0
        self.init_logger("train-" + self.model.__name__)
        for epoch in range(self.start_epoch, self.max_epoch):
            print('Epoch ' + str(epoch) + ' starts...')
            self.logger.info('Epoch ' + str(epoch) + ' starts...')
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            np.random.shuffle(self.train_order)
            for batch in range(self.train_batches):
                loss = self.train_one_batch()
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write(
                    "epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (
                        epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                sys.stdout.flush()
                self.logger.info(
                    "epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (
                        epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
            if (epoch + 1) % self.save_epoch == 0:
                print('Epoch ' + str(epoch) + ' has finished')

                auc, pr_x, pr_y = self.test_one_epoch()
                np.save(os.path.join(self.test_result_dir, self.model.__name__ + "{}-{}".format(epoch, auc) + '_x.npy'),
                        pr_x)
                np.save(os.path.join(self.test_result_dir, self.model.__name__ + "{}-{}".format(epoch, auc) + '_y.npy'),
                        pr_y)
                print('Saving model...')
                self.logger.info('Epoch ' + str(epoch) + ' has finished')
                self.logger.info('Saving model...')
                path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch) + "-" + str(auc))
                torch.save(self.trainModel.state_dict(), path)
                print('Have saved model to ' + path)
                self.logger.info('Have saved model to ' + path)