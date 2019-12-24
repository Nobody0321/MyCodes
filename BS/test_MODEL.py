import config
import models
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="pcnn_ave", help="name of the model")
args = parser.parse_args()
model = {
    "pcnn_att": models.PCNN_ATT,
    "pcnn_ave": models.PCNN_AVE
}
con = config.Config()
con.set_max_epoch(15)
con.load_test_data()
con.set_test_model(model[args.model_name])
con.set_epoch_range(list(range(0, 20)))
con.test()
