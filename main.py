"""
-*- coding: utf-8 -*-
@Author : JeikoLiu
@Institution : DHU/DBLab
@UpdateTime : 2024/12/07 20:35
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from util.utils import scRNADataset
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR
from model.models import GTGRNS
from train import train, validate, inference_grn

def parse_args():
    parser = argparse.ArgumentParser(description="请选择你的参数")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--embed_size', type=int, default=768, help='Embedding size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.999, help='Gamma value for learning rate scheduler')
    parser.add_argument('--scheduler_flag', type=bool, default=True)
    parser.add_argument('--normalize', help="请选择编码器是否要标准化", type=bool, default=True)
    parser.add_argument('--aggregator', help="请选择聚合函数", choices=['mean', 'lstm', 'max', 'sofmax'], default='mean')
    parser.add_argument('--hidden_dim', help="The dimension of hidden layer", type=int, default=256)
    parser.add_argument('--att_head', help="Number of head attentions", type=int, default=3)
    args = parser.parse_args()
    return args

def datasetLoader(train_dataset, test_dataset,val_dataset, all_dataset, args):
    trainLoader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.Batch_size,
                                              drop_last = False, num_workers = 0, shuffle = True)
    testLoader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.Batch_size,
                                              drop_last = False, num_workers = 0, shuffle = True)
    valLoader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = args.Batch_size,
                                              drop_last = False, num_workers = 0, shuffle = True)
    allLoader = torch.utils.data.DataLoader(dataset = all_dataset, batch_size = args.Batch_size,
                                              drop_last = False, num_workers = 0, shuffle = True)
    return trainLoader, testLoader, valLoader, allLoader

def main(data_dir, args):
    # 模型输入
    expression_data_path = data_dir + '/BL--ExpressionData.csv'
    train_data_path = data_dir + '/train_set.csv'
    val_data_path = data_dir + '/val_set.csv'
    test_data_path = data_dir + '/test_set.csv'
    all_grn_path = data_dir + '/allGRNs.csv'
    exp_data = np.array(pd.read_csv(expression_data_path, index_col=0, header=0))
    result_file = data_dir + '/inferred_result_500.csv'

    std = StandardScaler().fit_transform(exp_data.T)
    expression_data_shape = std.T.shape

    train_dataset = scRNADataset(train_data_path, exp_data)
    val_dataset = scRNADataset(val_data_path, exp_data)
    test_dataset = scRNADataset(test_data_path, exp_data)
    all_dataset = scRNADataset(all_grn_path, exp_data)

    expr_data = torch.from_numpy(exp_data).to(torch.float32)
    edge_data = np.array(pd.read_csv(train_data_path, index_col=0, header=0))
    true_edge = torch.from_numpy(edge_data[np.where(edge_data[:, -1] == 1)[0], :-1].T)
    data = Data(x=expr_data, edge_index=true_edge)

    # hidden_dim1 = args.hidden_dim
    # hidden_dim2 = args.hidden_dim[1]
    # att_head1 = args.att_head
    # att_head2 = args.att_head[1]

    # 初始化模型的参数
    embDim = args.embed_size
    numOfLayer = args.num_layers
    numOfHead = args.num_head
    lRate = args.lr
    epochs = args.epochs
    stepSize = args.step_size
    gamma = args.gamma
    normalize = args.normalize
    aggregator = args.aggregator
    global schedulerFlag
    schedulerFlag = args.scheduler_flag
    trainLoader, testLoader, valLoader, allGrnLoader = datasetLoader(train_dataset, test_dataset,val_dataset, all_dataset, args)

    model = GTGRNS(expression_data_shape, embDim, numOfLayer, numOfHead, aggregator, normalize)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lRate)
    scheduler = StepLR(optimizer, step_size=stepSize, gamma=gamma)
    lossFuntion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        train(model, trainLoader, lossFuntion, optimizer, epoch, scheduler, args, data)
        AUROC_val,AUPRC_val = validate(model,valLoader,lossFuntion, data)
        print('epoch {:4d} valid AUROC {:6.4f} valid AUPRC {:6.4f}'.format(epoch, AUROC_val, AUPRC_val))
        print("/******************************/")
        AUROC_test, AUPRC_test = validate(model, testLoader, lossFuntion, data)
        print('epoch {:4d} valid AUROC {:6.4f} valid AUPRC {:6.4f}'.format(epoch, AUROC_test, AUPRC_test))

    # 推断 grn
    inference_grn(model,allGrnLoader,lossFuntion, data, result_file)

if __name__ == '__main__':
    args = parse_args
    dataset_gene = 'Benchmark Dataset'
    dataset_type = 'Specific Dataset'
    data_type = 'mHSC-L'
    num = 500
    data_dir = 'Dataset/' + dataset_gene + '/' + dataset_type + '/' + data_type + '/TFs+' + str(num)
    sample_dir = 'Ablation/Sample Data/' + data_type + '/' + 'train_set_sample_80.csv'
    main(data_dir, args)