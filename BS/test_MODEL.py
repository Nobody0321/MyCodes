import config
import models
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="pcnn_att", help="name of the model")
args = parser.parse_args()
model = {
    "pcnn_att": models.PCNN_ATT,
    "self_self": models.self_self
}
con = config.Config()
con.set_drop_prob(0.5)
con.set_pretrain_model("checkpoint/self_self-6")
con.use_gpu = True
con.output_dim = 200
con.d_ff = 1024  # feed forward dim
con.n_heads = 5  # num of scale product attention heads
con.n_blocks = 1  # encoder blocks
con.set_max_epoch(15)
con.load_test_data()
con.set_test_model(model[args.model_name])
con.set_epoch_range([1, 7])
con.test()
