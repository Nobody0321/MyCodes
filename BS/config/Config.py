import os
import datetime
import logging
import numpy as np
import torch
import torch.optim as optim
import sklearn.metrics
from tqdm import tqdm


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


class Config(object):
    def __init__(self):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.data_dir = "./data"
        self.log_dir = "./logs"
        self.use_bag = True
        self.use_gpu = True
        self.is_training = True
        self.max_sen_length = 120  # max sentnece lem
        self.pos_num = 2 * self.max_sen_length  # num of positions = 240
        self.num_classes = 53  # num of relations
        self.hidden_dim = 230  # output dim encoder
        self.word_embedding_dim = 50  # word embedding dim
        self.pos_embedding_dim = 5
        self.max_epoch = 15
        self.opt_method = "SGD"
        self.optimizer = None
        self.learning_rate = 0.5
        self.weight_decay = 1e-5  # for Adadelta
        self.dropout = 0.1
        self.checkpoint_dir = "./checkpoint"
        self.test_result_dir = "./test_result"
        self.save_epoch = 1
        self.test_epoch = 1
        self.pretrain_model = None
        self.trainModel = None
        self.testModel = None
        self.batch_size = 160
        self.sentence_len = 120
        self.window_size = 3
        self.epoch_range = None
        self.input_dim = self.word_embedding_dim + self.pos_embedding_dim * 2  # input dim
        self.save_iter = 1000
        self.attn_dropout = 0.1
        self.train_start_epoch = 1

    def init_logger(self, log_name):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.DEBUG)
        log_file_name = log_name + ".log"
        log_handler = logging.FileHandler(os.path.join(self.log_dir, log_file_name), "w")
        log_format = logging.Formatter("%(asctime)s: %(message)s")
        log_handler.setFormatter(log_format)
        logger.addHandler(log_handler)
        self.logger = logger

    def set_data_path(self, data_path):
        self.data_dir = data_path

    def set_max_length(self, max_length):
        self.max_sen_length = max_length
        self.pos_num = 2 * self.max_sen_length

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_hidden_size(self, hidden_size):
        self.hidden_dim = hidden_size

    def set_window_size(self, window_size):
        self.window_size = window_size

    def set_pos_size(self, pos_size):
        self.pos_embedding_dim = pos_size

    def set_word_size(self, word_size):
        self.word_embedding_dim = word_size

    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_drop_prob(self, drop_prob):
        self.dropout = drop_prob

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch

    def set_save_epoch(self, save_epoch):
        self.save_epoch = save_epoch

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def set_is_training(self, is_training):
        self.is_training = is_training

    def set_use_bag(self, use_bag):
        self.use_bag = use_bag

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range

    def to_tensor(self, x):
        if self.use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)

    def load_train_data(self):
        """
        load pre processed training and allocate to this class
        """
        print("Reading training data...")
        self.data_word_vec = np.load(os.path.join(self.data_dir, "vec.npy"))  # word vec mapping
        self.data_train_word = np.load(os.path.join(self.data_dir, "train_word.npy"))  # one hot for each word
        self.data_train_pos1 = np.load(os.path.join(self.data_dir, "train_pos1.npy"))
        self.data_train_pos2 = np.load(os.path.join(self.data_dir, "train_pos2.npy"))
        # self.data_train_mask = np.load(os.path.join(self.data_path, "train_mask.npy"))
        if self.use_bag:
            self.data_query_label = np.load(os.path.join(self.data_dir, "train_ins_label.npy"))  # 每一句话的 real label
            self.data_train_label = np.load(os.path.join(self.data_dir, "train_bag_label.npy"))  # label for each bag
            self.data_train_scope = np.load(
                os.path.join(self.data_dir, "train_bag_scope.npy"))  # bag range in training data
        else:
            self.data_train_label = np.load(os.path.join(self.data_dir, "train_ins_label.npy"))
            self.data_train_scope = np.load(os.path.join(self.data_dir, "train_ins_scope.npy"))
        print("Finish reading")
        self.data_train_order = list(range(len(self.data_train_label)))  # bag label number, according to iteration size
        self.train_batches_num = len(self.data_train_label) // self.batch_size
        if len(self.data_train_label) % self.batch_size != 0:
            self.train_batches_num += 1

    def load_test_data(self):
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.data_dir, "vec.npy"))  # word id - vec mapping
        self.data_test_word = np.load(os.path.join(self.data_dir, "test_word.npy"))  # word idx
        self.data_test_pos1 = np.load(os.path.join(self.data_dir, "test_pos1.npy"))  #
        self.data_test_pos2 = np.load(os.path.join(self.data_dir, "test_pos2.npy"))
        # self.data_test_mask = np.load(os.path.join(self.data_path, "test_mask.npy"))
        if self.use_bag:
            self.data_test_label = np.load(os.path.join(self.data_dir, "test_bag_label.npy"))
            self.data_test_scope = np.load(os.path.join(self.data_dir, "test_bag_scope.npy"))
        else:
            self.data_test_label = np.load(os.path.join(self.data_dir, "test_ins_label.npy"))
            self.data_test_scope = np.load(os.path.join(self.data_dir, "test_ins_scope.npy"))
        print("Finish reading")
        self.test_batches = len(self.data_test_label) // self.batch_size
        if len(self.data_test_label) % self.batch_size != 0:
            self.test_batches += 1

        # all positive
        self.total_recall = self.data_test_label[:, 1:].sum()

    def set_train_model(self, model):
        print("Initializing training model...")
        self.model = model  # model class
        self.trainModel = self.model(config=self)  # model instance
        if self.pretrain_model is not None:
            self.trainModel.load_state_dict(torch.load(self.pretrain_model))
        if self.use_gpu:
            self.trainModel.cuda()
        if self.optimizer is not None:
            pass
        elif self.opt_method.lower() == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr=self.learning_rate, lr_decay=self.lr_decay,
                                           weight_decay=self.weight_decay)
        elif self.opt_method.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr=self.learning_rate,
                                            weight_decay=self.weight_decay)
        elif self.opt_method.lower() == "adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        else:
            print("optim {} not found, using SGD instead".format(self.opt_method))
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        print("Finish initializing")

    def set_test_model(self, model):
        print("Initializing test model...")
        self.model = model
        self.testModel = self.model(config=self)
        if self.use_gpu:
            self.testModel.cuda()
        self.testModel.eval()
        print("Finish initializing")

    def get_train_batch(self, batch_number):
        """
        a bag is an iteration
        :param batch_number: batch_number
        """
        # self.data_train_order ids shuffled, so the batch is shuffled
        input_scope = np.take(self.data_train_scope,
                              self.data_train_order[
                              batch_number * self.batch_size: (batch_number + 1) * self.batch_size], axis=0)
        index = []
        bag_scope = [0]  # store bag scopes
        for num in input_scope:
            # meaning from num[0] to num[1] in training set are sentences in one bag
            index += list(range(num[0], num[1] + 1))
            bag_scope.append(bag_scope[len(bag_scope) - 1] + (num[1] - num[0]) + 1)
        self.word_embedding_in_batch = self.data_train_word[index, :]
        self.postition_embedding1_in_batch = self.data_train_pos1[index, :]
        self.postition_embedding2_in_batch = self.data_train_pos2[index, :]
        # self.batch_mask = self.data_train_mask[index, :]

        # bags" label ids in one batch, like [12, 15] means relation 12 for bag 0 ans relation 15 for bag 1
        self.batch_label = np.take(self.data_train_label,
                                   self.data_train_order[
                                   batch_number * self.batch_size: (batch_number + 1) * self.batch_size], axis=0)
        self.batch_attention_query = self.data_query_label[index]  # get all rel ids for sens in one batch

        # all bag scope in one batch
        self.batch_scope = bag_scope

    def get_test_batch(self, batch):
        """
        :param batch: batch number
        """
        input_scope = self.data_test_scope[batch * self.batch_size: (batch + 1) * self.batch_size]
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.word_embedding_in_batch = self.data_test_word[index, :]
        self.postition_embedding1_in_batch = self.data_test_pos1[index, :]
        self.postition_embedding2_in_batch = self.data_test_pos2[index, :]
        # self.batch_mask = self.data_test_mask[index, :]
        self.batch_scope = scope

    def train_one_step(self):
        self.trainModel.selector.scope = self.batch_scope
        # assign batch word2vec to embedding.word
        self.trainModel.embedding.word = self.to_tensor(self.word_embedding_in_batch)
        self.trainModel.embedding.pos1 = self.to_tensor(self.postition_embedding1_in_batch)
        self.trainModel.embedding.pos2 = self.to_tensor(self.postition_embedding2_in_batch)
        # self.trainModel.encoder.mask = self.to_var(self.batch_mask)
        self.trainModel.selector.attention_query = self.to_tensor(self.batch_attention_query)
        self.trainModel.selector.label = self.to_tensor(self.batch_label)
        self.trainModel.classifier.label = self.to_tensor(self.batch_label)
        self.optimizer.zero_grad()  # clear gradient from last step
        try:
            loss, _output = self.trainModel()  # loss and prediction result
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    print("cleaning empty cache")
                loss, _output = self.trainModel()  # loss and prediction result
            else:
                raise e
        _output = _output.cpu().numpy().tolist()
        loss.backward()
        self.optimizer.step()
        loss = loss.cpu().detach().numpy()
        print("prediction: ", _output)
        print("gt label:   ", self.batch_label.tolist())
        self.logger.info("prediction: " + str(_output))
        self.logger.info("gt label: " + str(self.batch_label.tolist()))
        for i, prediction in enumerate(_output):
            if self.batch_label[i] == 0:
                self.acc_NA.add(prediction == self.batch_label[i])
            else:
                self.acc_not_NA.add(prediction == self.batch_label[i])
            self.acc_total.add(prediction == self.batch_label[i])
        return loss

    def test_one_step(self):
        self.testModel.selector.scope = self.batch_scope
        self.testModel.embedding.word = self.to_tensor(self.word_embedding_in_batch)
        self.testModel.embedding.pos1 = self.to_tensor(self.postition_embedding1_in_batch)
        self.testModel.embedding.pos2 = self.to_tensor(self.postition_embedding2_in_batch)
        # self.testModel.encoder.mask = self.to_var(self.batch_mask)
        # no label in test, we do not need labels to calculate loss
        try:
            score = self.testModel.test()  # only returns classification result, no need for loss
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    score = self.testModel.test()  # only returns classification result, no need for loss
            else:
                raise e
        return score

    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_auc = 0.0
        best_p = None
        best_r = None
        best_epoch = 0
        self.init_logger("train-" + self.model.__name__)
        for epoch in range(self.train_start_epoch, self.max_epoch):
            print("Epoch " + str(epoch) + " starts...")
            self.logger.info("Epoch " + str(epoch) + " starts...")
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            # shuffle bag
            np.random.shuffle(self.data_train_order)  # shuffle the bags labels" idx
            for batch_num in range(self.train_batches_num):
                self.get_train_batch(batch_num)
                loss = self.train_one_step()
                # if np.isnan(loss):
                #     np.save("./embedding.npy", self.word_embedding_in_batch)
                #     np.save("./scope", np.array(self.batch_scope))
                #     return
                time_str = datetime.datetime.now().isoformat()
                info_massage = "epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, " \
                               "total accuracy: %f\r" % (
                                   epoch, batch_num + 1, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(),
                                   self.acc_total.get())
                print(info_massage)
                self.logger.info(info_massage)
                if (batch_num + 1) % self.save_iter == 0:
                    print("Saving model at Epoch: {0}, iteration: {1}.".format(epoch, batch_num + 1))
                    path = os.path.join(self.checkpoint_dir,
                                        self.model.__name__ + "-{0}_{1}-{2}:{3}".format(epoch, batch_num + 1, loss, self.acc_not_NA.get()))
                    torch.save(self.trainModel.state_dict(), path)
            if epoch % self.test_epoch == 0:
                self.testModel = self.trainModel
                auc, pr_x, pr_y = self.test_one_epoch()
                np.save(os.path.join(self.test_result_dir, self.model.__name__ + str(epoch+1) + "_x.npy"), best_p)
                np.save(os.path.join(self.test_result_dir, self.model.__name__ + str(epoch+1) + "_y.npy"), best_r)
                if auc > best_auc:
                    best_auc = auc
                    best_p = pr_x
                    best_r = pr_y
                    best_epoch = epoch
            if epoch % self.save_epoch == 0:
                print("Epoch {} has finished".format(epoch))
                print("Saving model...")
                self.logger.info("Epoch {} has finished".format(epoch))
                self.logger.info("Saving model...")
                path = os.path.join(self.checkpoint_dir, self.model.__name__ + "-{}-auc-{}".format(epoch, auc))
                torch.save(self.trainModel.state_dict(), path)
                print("Have saved model to " + path)
                self.logger.info("Have saved model to " + path)

        info_massage = "Finish training\n" + "Best epoch = {0} | auc = {1}\n".format(best_epoch, best_auc) + "Storing best result..."
        print(info_massage)
        self.logger.info(info_massage)
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + "_x.npy"), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + "_y.npy"), best_r)
        print("Finish storing")
        self.logger.info("Finish storing")

    def test_one_epoch(self):
        """
        when testing, a bag consisits of sens of the same entity pair but different relation
        :return:
        """
        test_score = []
        for batch in tqdm(list(range(self.test_batches))):
            self.get_test_batch(batch)
            # batch_score [batch_size, 53], a bag and its multi relation scores
            # batch_score 返回一个batch中每一个bag的multi hot relation score
            batch_score = self.test_one_step()
            test_score += batch_score
        # test_score [epoch_size*[batch_size, 53}] stores each bag's multi-hot label
        test_result = []
        # for each epoch
        for i in range(len(test_score)):
            # for each batch
            for j in range(1, len(test_score[i])):
                # skip relation 0: NA
                # get each bag's gt and predicted  multi-hot label
                test_result.append([self.data_test_label[i][j], test_score[i][j]])
        # test_result (epoch_size_of_sen, 53) stores each instance's label and predicted label score
        # sort test_result by predicted score
        test_result = sorted(test_result, key=lambda x: x[1], reverse=True)  # decrease
        pr_x = []  # x for pr curve
        pr_y = []  # y for pr curve
        correct = 0  # tp
        for i, item in enumerate(test_result):
            correct += item[0]  # tp
            pr_x.append(float(correct) / self.total_recall)  # recall
            pr_y.append(float(correct) / (i + 1))  # precision
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)  # get area under pr curve
        print("auc: ", auc)
        self.logger.info("auc: ", auc)
        return auc, pr_x, pr_y

    def test(self):
        best_epoch = None
        best_auc = 0.0
        best_p = None
        best_r = None
        self.init_logger("test-" + self.model.__name__)
        for epoch in self.epoch_range:
            path = os.path.join(self.checkpoint_dir, self.model.__name__ + "-" + str(epoch))
            if not os.path.exists(path):
                continue
            print("Start testing epoch %d" % epoch)
            self.logger.info("Start testing epoch %d" % epoch)
            self.testModel.load_state_dict(torch.load(path))
            auc, p, r = self.test_one_epoch()
            self.logger.info("For Epoch {}, auc: {}".format(epoch, auc))
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_p = p
                best_r = r
            print("Finish testing epoch %d" % epoch)
            self.logger.info("Finish testing epoch %d" % epoch)

        print(("Best epoch = %d | auc = %f" % (best_epoch, best_auc)))
        self.logger.info("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        self.logger.info("Storing best result...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + "_x.npy"), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + "_y.npy"), best_r)
        print("Finish storing")
        self.logger.info("Finish storing")
