from config.Config import Config
import models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
con = Config()
con.set_pretrain_model("checkpoint/self_self-10-auc-0.20328220590112753")
con.use_gpu = True
con.data_dir = "./data"
con.set_max_epoch(20)
con.set_learning_rate(0.0001)
con.set_weight_decay(0.00001)
con.train_start_epoch = 11
con.save_iter = 3000
con.opt_method = "Adam"
con.load_train_data()
con.load_test_data()
con.set_train_model(models.PCNN_self)
con.train()
