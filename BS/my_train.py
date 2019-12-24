import config
import models
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='msnet_layer_att', help='name of the model')
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
    "msnet_layer_att": models.MSNET_Layer_ATT
}

con = config.Config()
con.set_opt_method("Adam")
con.set_learning_rate(0.001)
# con.set_pretrain_model("checkpoint/PCNN_Self-0-0.2997476786396999")
con.set_max_epoch(30)
# con.start_epoch = 4
con.set_batch_size(50)
con.load_train_data()
con.load_test_data()
con.set_train_model(model[args.model_name])
con.train()
