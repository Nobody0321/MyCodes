from config.Config import Config
import models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
con = Config()
con.use_gpu = True
con.data_dir = "./data"
con.set_max_epoch(20)
# con.opt_method = "SGD"
# con.set_learning_rate(0.01)
con.set_opt_method("adam")
con.set_learning_rate(0.001)
con.set_weight_decay(0.0001)
# con.train_start_epoch = 11
con.save_iter = 3000
con.load_train_data()
con.load_test_data()
# con.set_pretrain_model("")
con.set_train_model(models.self_self)
con.train()
