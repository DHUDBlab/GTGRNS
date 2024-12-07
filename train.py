"""
-*- coding: utf-8 -*-
@Author : JeikoLiu
@Institution : DHU/DBLab
@UpdateTime : 2024/12/07 19:38
"""

import torch
import pandas as pd
import util.utils as ut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(model, dataloader, lossFunction, optimizer, epoch, scheduler, args, data):
    model.train()
    total_loss = 0
    data = data.to(device)

    for enums, (linkIndex, info_edge, embeddings, linkLabels) in enumerate(dataloader):
        linkLabels = linkLabels.to(device)
        linkIndex = linkIndex.to(device)
        info_edge = info_edge.to(device)
        embeddings = embeddings.to(torch.float32)

        '''
        trueList = []
        for row_index in range(info_edge.shape[0]):
            if info_edge[row_index, -1] == 1:
                trueList.append(info_edge[row_index, :2])
        trueList = torch.stack(trueList)
        true_edge = trueList.transpose(0, 1)
        '''
        # true_edge = info_edge[ np.where(info_edge[:, -1] == 1)[0], :-1 ].T
        # true_edge = true_edge.to(device)
        '''
        non_zero_indexes = torch.nonzero(true_edge)
        row_indexes = non_zero_indexes[:, 0]
        col_indexes = non_zero_indexes[:, 1]
        values = true_edge[row_indexes, col_indexes]
        coo_tensor = torch.sparse_coo_tensor(indices=torch.stack((row_indexes, col_indexes)),
                                             values = values,
                                             size=true_edge.shape).coalesce()
        # coo_matrix = sp.coo_matrix((coo_tensor.values(), (coo_tensor.indices()[0], coo_tensor.indices()[1])),
        #                         shape=coo_tensor.size())
        # csc_matrix = coo_matrix.tocsc()
        coo_tensor = coo_tensor.coalesce()
        '''
        optimizer.zero_grad()

        sage_embedding = model.Sage_forward(data.x, data.edge_index)
        # att_embedding = model.GAT_forward(data.x, data.edge_index)
        linkPredict = model(linkIndex, sage_embedding)
        # linkPredict = model(linkIndex, expr_embedding)
        loss = lossFunction(linkPredict.squeeze(), linkLabels.float())
        total_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        if args.scheduler_flag:
            scheduler.step()

    print('| epoch {:4d} | total_loss {:6.4f}'.format(epoch, total_loss))


def validate(model, dataloader, lossFunction, data):
    model.eval()
    # expression_data = torch.from_numpy(expression_data).to(torch.float32).to(device)
    data = data.to(device)
    ps = []
    ls = []
    for enums, (linkIndex, info_edge, embeddings, linkLabels) in enumerate(dataloader):
        embeddings = embeddings.to(torch.float32)
        linkIndex.to(device)
        info_edge = info_edge.to(device)
        embeddings.to(device)
        '''
        trueList = []
        for row_index in range(info_edge.shape[0]):
            if info_edge[row_index, -1] == 1:
                trueList.append(info_edge[row_index, :2])
        trueList = torch.stack(trueList)
        true_edge = trueList.transpose(0, 1)
        '''

        # true_edge = info_edge[np.where(info_edge[:, -1] == 1)[0], :-1].T
        '''
        non_zero_indexes = torch.nonzero(true_edge)
        row_indexes = non_zero_indexes[:, 0]
        col_indexes = non_zero_indexes[:, 1]
        values = true_edge[row_indexes, col_indexes]
        coo_tensor = torch.sparse_coo_tensor(indices=torch.stack((row_indexes, col_indexes)),
                                             values=values,
                                             size=true_edge.shape).coalesce()
        # coo_matrix = sp.coo_matrix((coo_tensor.values(), (coo_tensor.indices()[0], coo_tensor.indices()[1])),
        #                            shape=coo_tensor.size())
        # csc_matrix = coo_matrix.tocsc()
        coo_tensor = coo_tensor.coalesce()
        '''

        sage_embedding =model.Sage_forward(data.x, data.edge_index)
        # predicted_label = model(linkIndex, embeddings)
        predicted_label = model(linkIndex, sage_embedding)
        ps.extend(predicted_label)
        linkLabels.to(device)
        ls.extend(linkLabels)

    ps = torch.vstack(ps)
    ls = torch.vstack(ls)

    AUROC, AUPRC = ut.evaluation(y_pre = ps, y_true = ls)
    return AUROC, AUPRC

def inference_grn(model, dataloader, lossFunction, data, result_file):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        tf = []
        target = []
        pre = []

        for enums, (linkIndex, info_edge, embeddings, label) in enumerate(dataloader):
            for i in range(linkIndex.shape[0]):
                tf.append(linkIndex[i][0].item())
                target.append(linkIndex[i][1].item())

            embeddings = embeddings.to(torch.float32)
            linkIndex.to(device)
            info_edge = info_edge.to(device)
            embeddings.to(device)

            sage_embedding = model.Sage_forward(data.x, data.edge_index)
            linkPredict = model(linkIndex, sage_embedding)
            for pre_label in linkPredict:
                tmp = pre_label.item()
                pre.append(tmp)
            # pre.extend(linkPredict)
            label.to(device)

        infResult =pd.DataFrame()
        infResult['Gene1'] = tf
        infResult['Gene2'] = target
        infResult['Score'] = pre
        infResult.to_csv(result_file)