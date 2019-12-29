import config
import models
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="msnet_layer_att2", help="name of the model")
args = parser.parse_args()
model = {
    'pcnn_att': models.PCNN_ATT,
    'pcnn_one': models.PCNN_ONE,
    'pcnn_ave': models.PCNN_AVE,
    'pcnn_self': models.PCNN_Self,
    'cnn_att': models.CNN_ATT,
    'cnn_one': models.CNN_ONE,
    'cnn_ave': models.CNN_AVE,
    'msnet_att': models.MSNET_ATT,
    "msnet_layer_att": models.MSNET_Layer_ATT,
    "msnet_layer_att2": models.MSNet_Layer_att2,
    "bigru": models.BiGru_ATT,
    "att_pcnn": models.SelfPCNN_ATT
}
con = config.Config()
# con.set_max_epoch(15)
con.set_batch_size(50)
con.set_use_bag(False)
con.load_test_data()
con.set_test_model(model[args.model_name])
con.set_epoch_range([3])
con.test()
