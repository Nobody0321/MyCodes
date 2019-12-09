from config.Config import Config
import models
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='blstm_att', help='name of the model')
args = parser.parse_args()
con = Config()
con.set_max_epoch(15)
con.load_train_data()
con.load_test_data()
con.set_train_model(models.BLSTM_ATT)
con.train()
