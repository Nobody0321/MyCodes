from config.Config import Config
import models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
con = Config()
con.set_drop_prob(0.5)
# con.set_pretrain_model("checkpoint/LayerAttention-9_5000-0.34088969230651855:0.7250814732829122")
con.use_gpu = True
con.data_dir = "./data"
# con.data_dir = "./mini_dataset-1000"
con.set_max_epoch(15)
con.batch_size = 20
con.output_dim = 200
con.d_ff = 1024  # feed forward dim
con.n_heads = 5  # num of scale product attention heads
con.n_blocks = 1  # encoder blocks
con.learning_rate = 0.001
con.seve_iter = 3000
con.opt_method = "Adam"
con.load_train_data()
con.load_test_data()
con.set_train_model(models.self_self)
con.train()
