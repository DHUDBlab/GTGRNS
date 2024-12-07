"""
-*- coding: utf-8 -*-
@Author : JeikoLiu
@Institution : DHU/DBLab
@UpdateTime : 2024/12/07 19:50
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score,average_precision_score, precision_recall_curve, roc_curve, auc

class scRNADataset(Dataset):
    def __init__(self, dataPath, expression_data):
        data = pd.read_csv(dataPath, index_col = 0, header = 0)

        self.dataset = np.array(data.iloc[:, :])
        # self.trueEdge = np.array( data[data.iloc[:, -1] == 1].iloc[:, :-1] )
        label = np.array(data.iloc[:, -1])
        self.label = np.eye(2)[label]
        self.label = label
        self.expression_data = expression_data

    def __getitem__(self, idx):
        linkIndex = self.dataset[idx, :-1]
        info_edge = self.dataset[idx]
        # 在指定的 axis 上增加一个维度，转换成一个（1，N）的二维数组
        gene1_expr = np.expand_dims(self.expression_data[linkIndex[0]], axis=0)
        gene2_expr = np.expand_dims(self.expression_data[linkIndex[1]], axis=0)
        embeddings = np.concatenate((gene1_expr, gene2_expr), axis=0)
        label = self.label[idx]
        # expression_data = self.expression_data
        return linkIndex,info_edge, embeddings, label

    def __len__(self):
        return len(self.dataset)

def computeScore(TrueEdgeDict, PredEdgeDict):
    """
    :param TrueEdgeDict: 真实调控边字典
    :param PredEdgeDict: 预测调控边字典
    :return: 计算评测分数 AUROC、AUPRC
    """
    outDF = pd.DataFrame([TrueEdgeDict, PredEdgeDict]).T
    outDF.columns = ['TrueEdges', 'PredEdges']
    # print(outDF)
    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)

    return auc(recall, prec), auc(fpr, tpr)

def evaluation(y_true, y_pre, flag = False):
    if flag:
        y_p = y_pre[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pre.cpu().detach().numpy()
        y_p = y_p.flatten()
    y_t = y_true.cpu().numpy().flatten().astype(int)
    AUROC = roc_auc_score(y_true=y_t, y_score=y_p)
    AUPRC = average_precision_score(y_true=y_t,y_score=y_p)
    return AUROC, AUPRC

"""
Refer to the GEENLink source code for details on this section
https://github.com/zpliulab/GENELink
"""
def Network_Statistic(data_type,net_scale,net_type):

    if net_type =='STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
               'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError