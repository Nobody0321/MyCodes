from config.Config import Config
import models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
con = Config()
con.use_gpu = True
con.data_dir = "./data"
# con.set_max_epoch(20)
# con.opt_method = "SGD"
# con.set_learning_rate(0.01)
con.set_opt_method("adam")
<<<<<<< HEAD
con.set_learning_rate(0.0001)
# con.set_weight_decay(0.000001)
# con.train_start_epoch = 11
=======
con.set_learning_rate(0.00001)
# con.set_weight_decay(0.000001)
con.train_start_epoch = 7
>>>>>>> 651c319435849811baf16606ea5ea81d1048c89d
# con.d_ff = 2048
con.save_iter = 3000
con.load_train_data()
con.load_test_data()
con.set_pretrain_model("./checkpoint/PCNN_self-6-auc-0.283806512101348")
con.set_train_model(models.PCNN_self)
con.train()
