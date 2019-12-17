from config.Config import Config
import models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
con = Config()
# con.set_pretrain_model("BiGruAtt-8_3000")
con.set_drop_prob(0.5)
con.use_gpu = True
con.data_dir = "./data"
con.set_max_epoch(15)
con.batch_size = 50
# con.hidden_size = 256  # output dim for blstm
con.d_model = 230  # input dim for attention layer
con.d_ff = 1024  # feed forward dim
con.d_feature = 46  # dim for each attention head
con.n_heads = 5  # num of scale product attention heads
con.n_blocks = 1  # encoder blocks
con.learning_rate = 0.001
con.seve_iter = 800
con.opt_method = "Adam"
con.load_train_data()
con.load_test_data()
con.set_train_model(models.BiGruAtt)
con.train()
