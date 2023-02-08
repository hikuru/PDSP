import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import mean_squared_error

def write_pickle(data,path):
    pickle.dump(data, open(path, 'wb'))

def read_pickle(path):
    data = pickle.load(open(path, 'rb'))
    return data

def write_json(json_string, path):
    json.dump(json_string, open(path, 'w'))

def read_json(path):
    data = json.load(open(path))
    return data

def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1!=0
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X-means1)/std1[feat_filt]
    if norm == 'norm':
        return(X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        X[:,std2==0]=0
        return(X, means1, std1, means2, std2, feat_filt)

def read_train_files(drugCom_processed_path = 'input_data/DrugComb_processed_all_metrics_ic50_binary.csv',
                    drug_features_path = 'input_data/drug_features.json',
                    cell_line_features_path = 'input_data/cell_line_features_ready_qnorm.json'):
    summary_final = pd.read_csv(drugCom_processed_path)
    drug_features = read_json(drug_features_path)
    cell_line_features = read_json(cell_line_features_path)
    print("Dataset loading is done...")
    return summary_final, drug_features, cell_line_features

def prepare_data(drugCom_processed_path = 'input_data/DrugComb_processed_all_metrics_ic50_binary.csv',
                drug_features_path = 'input_data/drug_features.json',
                cell_line_features_path = 'input_data/cell_line_features_ready_qnorm.json',
                norm='tanh_norm',
                reversed = 1):

    summary_final, drug_features, cell_line_features = read_train_files(drugCom_processed_path,
                                                                        drug_features_path,
                                                                        cell_line_features_path)

    drug_features =  {k.upper(): v for k, v in drug_features.items()}
    n_feat = len(drug_features[list(drug_features.keys())[0]])
    n_feat_cl = len(cell_line_features[list(cell_line_features.keys())[0]])
    n_sample = summary_final.shape[0]

    X_drug_row = np.zeros((n_sample, n_feat))
    X_drug_col = np.zeros((n_sample, n_feat))
    X_cell_line = np.zeros((n_sample, n_feat_cl))
    loewe = np.zeros((n_sample,))
    ic50_row = np.zeros((n_sample,))
    ic50_col = np.zeros((n_sample,))

    print("Dataset preparation is STARTED...")
    for i in range(n_sample):
        X_drug_row[i,:] = drug_features[summary_final['drug_row'].iloc[i]]
        X_drug_col[i,:] = drug_features[summary_final['drug_col'].iloc[i]]
        X_cell_line[i,:] = cell_line_features[summary_final['cell_line_name'].iloc[i]]
        loewe[i] = summary_final['synergy_bliss'].iloc[i]
        ic50_row[i] = summary_final['ic50_row'].iloc[i]
        ic50_col[i] = summary_final['ic50_col'].iloc[i]

    train_data = {}
    test_data = {}
    val_data = {}

    train_inds = (summary_final['split'] == 1).values
    val_inds = (summary_final['split'] == 2).values
    test_inds = (summary_final['split'] == 3).values

    # Concatenate drug and cell line features
    train_drug_row = np.concatenate((X_drug_row[train_inds,], X_cell_line[train_inds,]), axis=1)
    train_drug_col = np.concatenate((X_drug_col[train_inds,], X_cell_line[train_inds,]), axis=1)
    if reversed == 1:
        tmp_col = train_drug_col
        train_drug_col = np.concatenate((train_drug_col, train_drug_row), axis=0)
        train_drug_row = np.concatenate((train_drug_row, tmp_col), axis=0)

    val_drug_row = np.concatenate((X_drug_row[val_inds,], X_cell_line[val_inds,]), axis=1)
    val_drug_col = np.concatenate((X_drug_col[val_inds,], X_cell_line[val_inds,]), axis=1)

    test_drug_row = np.concatenate((X_drug_row[test_inds,], X_cell_line[test_inds,]), axis=1)
    test_drug_col = np.concatenate((X_drug_col[test_inds,], X_cell_line[test_inds,]), axis=1)

    all_data = np.concatenate((train_drug_row, train_drug_col), axis=0)
    # Normalize data
    drugs_all, mean1, std1, mean2, std2, feat_filt = normalize(all_data, norm=norm)

    train_data['drug_row'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(train_drug_row, mean1, std1, mean2, std2,
                                                                                feat_filt=feat_filt, norm=norm)

    val_data['drug_row'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(val_drug_row, mean1, std1, mean2, std2,
                                                                                feat_filt=feat_filt, norm=norm)

    test_data['drug_row'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(test_drug_row, mean1, std1, mean2, std2,
                                                                                feat_filt=feat_filt, norm=norm)

    train_data['drug_col'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(train_drug_col, mean1, std1, mean2, std2,
                                                                                feat_filt=feat_filt, norm=norm)

    val_data['drug_col'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(val_drug_col, mean1, std1, mean2, std2,
                                                                                feat_filt=feat_filt, norm=norm)

    test_data['drug_col'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(test_drug_col, mean1, std1, mean2, std2,
                                                                                feat_filt=feat_filt, norm=norm)

    train_data['loewe'] = np.concatenate((loewe[train_inds,], loewe[train_inds,]), axis = 0)
    train_data['ic50_row'] = np.concatenate((ic50_row[train_inds,], ic50_row[train_inds,]), axis = 0)
    train_data['ic50_col'] = np.concatenate((ic50_col[train_inds,], ic50_col[train_inds,]), axis = 0)

    test_data['loewe'] = loewe[test_inds,]
    test_data['ic50_row'] = ic50_row[test_inds,]
    test_data['ic50_col'] = ic50_col[test_inds,]

    val_data['loewe'] = loewe[val_inds,]
    val_data['ic50_row'] = ic50_row[val_inds,]
    val_data['ic50_col'] = ic50_col[val_inds,]

    print("Dataset preparation is DONE...")

    return train_data, test_data, val_data


def pearson(y, pred):
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]
    print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
    return pear_value, pear_p_val

def spearman(y, pred):
    spear = stats.spearmanr(y, pred)
    spear_value = spear[0]
    spear_p_val = spear[1]
    print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
    return spear_value, spear_p_val

def mse(y, pred):
    err = mean_squared_error(y, pred)
    print("Mean squared error is {}".format(err))
    return err

