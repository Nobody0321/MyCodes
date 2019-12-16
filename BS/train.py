from config.Config import Config
import models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
con = Config()
con.use_gpu = True
con.data_dir = "./data"
con.set_max_epoch(15)
con.batch_size = 80
# con.hidden_size = 256  # output dim for blstm
con.d_model = 256  # input dim for attention layer
con.d_ff = 1024  # feed forward dim
con.d_feature = 32  # dim for each attention head
con.n_heads = 4  # num of scale product attention heads
con.n_blocks = 3  # encoder blocks
con.learning_rate = 0.01
# con.opt_method = "Adadelta"
con.load_train_data()
con.load_test_data()
con.set_train_model(models.BiGruAtt)
con.train()
