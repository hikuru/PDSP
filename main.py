import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pdsp
import utils
import argparse

parser = argparse.ArgumentParser(description='REQUEST REQUIRED PARAMETERS OF MatchMaker v2')

parser.add_argument('--comb-data-name', default='input_data/DrugComb_processed_all_metrics_ic50_binary.csv',
                    help="Name of the drug combination data")

parser.add_argument('--cell_line-gex', default='input_data/cell_line_features_ready_qnorm.json',
                    help="Name of the cell line gene expression data")

parser.add_argument('--drug-chemicals', default='input_data/drug_features.json',
                    help="Name of the chemical features data for drug 1")

parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--train-test-mode', default=1, type = int,
                    help="Test of train mode (0: test, 1: train)")

parser.add_argument('--gpu-support', default=True,
                    help='Use GPU support or not')

parser.add_argument('--reversed', default=1, type = int,
                    help="Reversed train data added (0: not, 1: reversed)")

parser.add_argument('--config', default=2, type = int,
                    help="Config mode (2: conf2, 3: conf3, 4:conf4)")

parser.add_argument('--add-model', default=0, type = int,
                    help="Merge drugs model mode (0: concat model, 1: add model)")

args = parser.parse_args()

# ---------------------------------------------------------- #
num_cores = 8
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
GPU = True
if args.gpu_support:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 2
    num_GPU = 0


config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU})

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

train_data, test_data, val_data = utils.prepare_data(drugCom_processed_path='input_data/DrugComb_processed_all_metrics_ic50_binary.csv')

modelName = 'best_model.h5'
layers = {}
layers['encoder_layers'] = [4096, 2048, 1024] # layers of Drug Encoder Network
layers['ic50_layers'] = [2048, 1024] # layers of Sensitivity Network
layers['synergy_layers'] = [2048, 1024] # layers of Synergy Prediction Network

input_feat_size = train_data['drug_row'].shape[1]
model = pdsp.generate_model(input_feat_size, layers, args.config, args.add_model)

model = pdsp.trainer(model=model, l_rate=0.0001, train=train_data, val=val_data,
                                    epo=1000, batch_size=128, earlyStop = 100,
                                    alpha=alpha, modelName=modelName)

sens_drug1, synergy, sens_drug2 = pdsp.predict(model, [test_data['drug_row'],test_data['drug_col']])
sens_drug12, synergy2, sens_drug22 = pdsp.predict(model, [test_data['drug_col'],test_data['drug_row']])

if args.reversed == 1:
    synergy = (synergy + synergy2) / 2.0

mse_synergy = utils.mse(test_data['loewe'], synergy)
spearman_synergy = utils.spearman(test_data['loewe'], synergy)
pearson_synergy = utils.pearson(test_data['loewe'], synergy)

msg = msg + '[TEST] Synergy MSE: ' + str(mse_synergy) + ' Synergy Pearson: ' + str(pearson_synergy) + ' Synergy Spearman: ' + str(spearman_synergy) + '\n\n'

