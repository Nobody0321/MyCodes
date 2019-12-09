from Config import Config
from myTrainer import Trainer
import os
import argparse
from BlstmAtt import BLstmAtt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='blstm_att', help='name of the model')
args = parser.parse_args()
con = Config()
con.max_epoch = 15
trainer = Trainer(con)
trainer.load_train_data()
trainer.load_test_data()
trainer.set_train_model(BLstmAtt)
trainer.train()
