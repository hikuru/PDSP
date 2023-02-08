import pandas as pd
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization, Activation, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K

def drug_cell_line_encoder(input_feat_size, encoder_layers, inDrop=0.2, drop=0.5):
    encoder_input = Input(shape=(input_feat_size,))

    for l in range(len(encoder_layers)):
        if l == 0:
            middle_layer = Dense(int(encoder_layers[l]), activation='relu', kernel_initializer='he_normal')(encoder_input)
            middle_layer = Dropout(float(inDrop))(middle_layer)
        elif l == (len(encoder_layers)-1):
            encoder_output = Dense(int(encoder_layers[l]), activation='linear', kernel_initializer='he_normal', name='drug_out')(middle_layer)
        else:
            middle_layer = Dense(int(encoder_layers[l]), activation='relu', kernel_initializer='he_normal')(middle_layer)
            middle_layer = Dropout(float(drop))(middle_layer)

    model = Model(encoder_input, encoder_output)

    return model

def ic50_predictor(input_feat_size, ic50_layers, drop=0.5, name='ic50_predictor'):

    ic50_input = Input(shape=(input_feat_size,))

    for ic50_layer in range(len(ic50_layers)):
        if len(ic50_layers) == 1:
            ic50_FC = Dense(int(ic50_layers[ic50_layer]), activation='relu', kernel_initializer='he_normal')(ic50_input)
            ic50_output = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='ic50_output')(ic50_FC)
        else:
            # more than one FC layer at concat
            if ic50_layer == 0:
                ic50_FC = Dense(int(ic50_layers[ic50_layer]), activation='relu', kernel_initializer='he_normal')(ic50_input)
                ic50_FC = Dropout(float(drop))(ic50_FC)
            elif ic50_layer == (len(ic50_layers)-1):
                ic50_FC = Dense(int(ic50_layers[ic50_layer]), activation='relu', kernel_initializer='he_normal')(ic50_FC)
                ic50_output = Dense(1, activation='sigmoid', name='ic50_output')(ic50_FC)
            else:
                ic50_FC = Dense(int(ic50_layers[ic50_layer]), activation='relu', kernel_initializer='he_normal')(ic50_FC)
                ic50_FC = Dropout(float(drop))(ic50_FC)

    model = Model(ic50_input, ic50_output, name=name)

    return model

def synergy_predictor(input_feat_size, synergy_layers, drop=0.5):

    synergy_input = Input(shape=(input_feat_size,))

    for synergy_layer in range(len(synergy_layers)):
        if len(synergy_layers) == 1:
            synergy_FC = Dense(int(synergy_layers[synergy_layer]), activation='relu', kernel_initializer='he_normal')(synergy_input)
            synergy_output = Dense(1, activation='linear', kernel_initializer='he_normal', name='synergy_output')(synergy_FC)
        else:
            # more than one FC layer at concat
            if synergy_layer == 0:
                synergy_FC = Dense(int(synergy_layers[synergy_layer]), activation='relu', kernel_initializer='he_normal')(synergy_input)
                synergy_FC = Dropout(float(drop))(synergy_FC)
            elif synergy_layer == (len(synergy_layers)-1):
                synergy_FC = Dense(int(synergy_layers[synergy_layer]), activation='relu', kernel_initializer='he_normal')(synergy_FC)
                synergy_output = Dense(1, activation='linear', name='synergy_output')(synergy_FC)
            else:
                synergy_FC = Dense(int(synergy_layers[synergy_layer]), activation='relu', kernel_initializer='he_normal')(synergy_FC)
                synergy_FC = Dropout(float(drop))(synergy_FC)

    model = Model(synergy_input, synergy_output, name='synergy_predictor')

    return model

def generate_model(input_feat_size, layers, config, add_model, fine_tune=False):

    drug_row = Input(shape=(input_feat_size,))
    drug_col = Input(shape=(input_feat_size,))

    # initialize encoder
    encoder1 = drug_cell_line_encoder(input_feat_size, layers['encoder_layers'])
    if config == 4:
        encoder2 = drug_cell_line_encoder(input_feat_size, layers['encoder_layers'])

    # drugs are processed with encoder
    drug_col_feats = encoder1(drug_col)
    if config == 2 or config == 3:
        drug_row_feats = encoder1(drug_row)
    elif config == 4:
        drug_row_feats = encoder2(drug_col)

    encoder_out_size = int(layers['encoder_layers'][-1])

    single_response_predictior1 = ic50_predictor(encoder_out_size, layers["ic50_layers"],name='ic50_predictor')

    if config == 3 or config == 4:
        single_response_predictior2 = ic50_predictor(encoder_out_size, layers["ic50_layers"],name='ic50_predictor_1')

    drug_row_ic50 = single_response_predictior1(drug_row_feats)

    if config == 3 or config == 4:
        drug_col_ic50 = single_response_predictior2(drug_col_feats)
    elif config == 2:
        drug_col_ic50 = single_response_predictior1(drug_col_feats)

    synergy_in_size = 2 * encoder_out_size
    if add_model == 0:
        concatModel = concatenate([drug_row_feats, drug_col_feats])
        synergy_in_size = 2 * encoder_out_size
    elif add_model == 1:
        concatModel = Add()([drug_row_feats, drug_col_feats])
        synergy_in_size = encoder_out_size

    combination_predictor = synergy_predictor(synergy_in_size, layers["synergy_layers"])

    if(fine_tune):
        for layer in combination_predictor.layers:
            layer.trainable = False


    synergy_loewe = combination_predictor(concatModel)

    model = Model(inputs=[drug_row, drug_col], outputs=[drug_row_ic50, synergy_loewe, drug_col_ic50])

    return model


def weighted_binary_crossentropy(y_true, y_pred):
    zero_weight=1.0
    one_weight=3.0
    b_ce = K.binary_crossentropy(y_true, y_pred)

    # weighted calc
    weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce

    return K.mean(weighted_b_ce)


def trainer(model, l_rate, train, val, epo=1000, batch_size=128, earlyStop = 100, modelName="best_model.h5"):

    #losses = ['mean_squared_error', 'mean_squared_error', 'mean_squared_error']
    losses = {
               'ic50_predictor' : 'weighted_binary_crossentropy',
               'synergy_predictor' : 'mean_squared_error',
               'ic50_predictor_1' : 'weighted_binary_crossentropy'
             }
    lossWeights = {
               'ic50_predictor' : 10, 
               'synergy_predictor' : 1, 
               'ic50_predictor_1' : 10,
                  }
    get_custom_objects().update({'weighted_binary_crossentropy': weighted_binary_crossentropy})

    loss_str = 'val_synergy_predictor_loss'
    cb_check = ModelCheckpoint((modelName), verbose=1, monitor=loss_str,save_best_only=True, mode='auto')
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=keras.optimizers.Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, amsgrad=False))
    model.fit([train["drug_row"], train["drug_col"]], [train["ic50_row"], train["loewe"], train["ic50_col"]], 
                    epochs=epo, shuffle=True, batch_size=batch_size,verbose=1, 
                   validation_data=([val["drug_row"], val["drug_col"]], [val["ic50_row"], val["loewe"], val["ic50_col"]]),
                   callbacks=[EarlyStopping(monitor=loss_str, mode='auto', patience = earlyStop),cb_check])
    return model

def predict(model, data):
    ic50_drug1, synergy, ic50_drug2 = model.predict(data)
    return ic50_drug1.flatten(), synergy.flatten(), ic50_drug2.flatten()