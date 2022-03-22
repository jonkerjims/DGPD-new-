import warnings

warnings.filterwarnings("ignore")
import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
# import torch as th
BASE_DIR = os.getcwd()

# import dgl
# from dgl.data.utils import download, extract_archive, get_download_dir

from itertools import product
from collections import Counter
from copy import deepcopy
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph

import random

random.seed(1234)
np.random.seed(1234)


# ======================================
def load_data(directory):
    IP = pd.read_csv(os.path.join(directory,'feature_all.csv'))
    IP = pd.DataFrame(IP).reset_index()
    IP.rename(columns={'index': 'id'}, inplace=True)
    IP['id'] = IP['id'] + 1
    # print('===============IP.shape:',IP.shape)   (1952, 401)
    return IP


def sample(directory, random_seed):
    all_associations = pd.read_csv(os.path.join(directory,'label_all.csv'))  # 加一行三列的名称，pd.read_csv读取csv时默认第一行作为header读取
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)
    # print('==============================sample_df.shape:', sample_df.shape)
    return sample_df  # (492 rows × 3 columns)



def obtain_data(directory, isbalance):
    IP = load_data(directory)

    # isbalance = False 执行else后的程序

    if isbalance:
        dtp = pd.read_csv(os.path.join(directory,'label_all.csv'))  # [ 1952 rows x 3 columns]
    else:
        dtp = sample(directory)  # [10860 rows x 3 columns]
        # 保存成csv文件查看所有数据


    protein_ids = list(set(dtp['ID']))
    random.shuffle(protein_ids)
    # print('# protein = {}'.format(len(protein_ids)))

    protein_test_num = int(len(protein_ids) / 5)
    # print('# Test: protein = {}'.format(protein_test_num))
    # print(dtp)
    # print(IP)
    knn_x = pd.merge(dtp, IP, left_on='ID', right_on='id')

    # ========================================

    X = np.array(knn_x)
    # print(X.shape)
    pd.DataFrame(X).to_csv(os.path.join(BASE_DIR,'Predict','deeplearn','dataSet','knn_x_all.csv'), index=None, encoding='utf-8')

    # ========================================

    label = dtp['label']
    # print(knn_x)
    knn_x.drop(labels=['ID', '#', 'label', 'id'], axis=1, inplace=True)  # 删除label的这四列
    # print(knn_x)


    return IP, dtp, protein_ids, protein_test_num, knn_x, label



def generate_task_Tp_train_test_idx(knn_x, dtp):
    kf = KFold(n_splits=5, shuffle=True)

    train_index_all, test_index_all, n = [], [], 0
    train_id_all, test_id_all = [], []
    fold = 0
    train_id = []
    train_index = []
    test_id = []
    test_index = []
    for i in range(1, 1935):
        lt = []
        lt.append(i+1)
        train_index.append(i)
        train_id.append(lt)
        # train_id_all.append(np.array(dtp.iloc[i][['ID']]))
        # print(train_id_all)
    for j in range(1935, 1935+num):
        lt = []
        lt.append(j+1)
        test_index.append(j)
        test_id.append(lt)
        # test_id_all.append(np.array(dtp.iloc[j][['ID']]))
        # print(test_id)
    train_index_all.append(np.array(train_index))
    train_id_all.append(np.array(train_id, dtype=int))
    test_index_all.append(np.array(test_index))
    test_id_all.append(np.array(test_id, dtype=int))
    # print('============train_id_all\n', train_id_all)
    # print('============train_index_all\n', train_index_all)
    # print('============test_index_all\n', test_index_all)

    return train_index_all, test_index_all, train_id_all, test_id_all
    # for train_idx, test_idx in tqdm(kf.split(knn_x)):  # train_index与test_index为下标
    #     # print('-------Fold ', fold)
    #     # print('============================\n', train_idx)
    #     # print(test_idx)
    #     train_index_all.append(train_idx)
    #     test_index_all.append(test_idx)
    #
    #     train_id_all.append(np.array(dtp.iloc[train_idx][['ID']]))
    #     test_id_all.append(np.array(dtp.iloc[test_idx][['ID']]))
    #
    #     # print('# Pairs: Train = {} | Test = {}'.format(len(train_idx), len(test_idx)))
    #     # print(train_index_all)
    #
    #
    #
    #     fold += 1
    #
    #     return train_index_all, test_index_all, train_id_all, test_id_all



'''
KNN
'''
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report


def generate_knn_graph_save(knn_x, label, n_neigh, train_index_all, test_index_all, pwd, task, balance):
    fold = 0
    for train_idx, test_idx in zip(train_index_all, test_index_all):
        # print('-------Fold ', fold)

        knn_y = deepcopy(label)  # 深层复制数据 label 0 或 1 的标签
        # print(knn_y)
        # knn_y[test_idx] = 0  # 把测试集标签设为0
        # print(knn_y)
        # print('Label: ', Counter(label))  # 标签 为1 或0（5430,5430）
        # print('knn_y: ', Counter(knn_y))  # 测试集标签设为0 后的总数据（）

        knn = KNeighborsClassifier(n_neighbors=n_neigh)  # n_neigh就是选取最近的点的个数：n_neigh in [1, 3, 5, 7, 10, 15]
        knn.fit(knn_x, knn_y)  # knn_x 为训练数据   knn_y为训练的标签 有关联为1 没有关联为0   训练模型


        knn_y_pred = knn.predict(knn_x)  # 预测出各个分类的结果(496)
        # print('=======================knn_y_pred:', knn_y_pred)
        knn_y_prob = knn.predict_proba(knn_x)  # 测属每个测试集样本对应各个分类结果的概率(496)
        # print('-------------knn_y_proba:', knn_y_prob)
        knn_neighbors_graph = knn.kneighbors_graph(knn_x, n_neighbors=n_neigh)  # 用kneighbors_graph查找的近邻数((0, 0)	1.0)
        # print('+++++++++++++++++\n', knn_neighbors_graph)
        # cf=len(knn_y_prob)
        # print('knnx',knn_x)
        # print(knn_neighbors_graph)
        # knn_neighbors_graph.toarray ()
        prec_reca_f1_supp_report = classification_report(knn_y, knn_y_pred, target_names=['label_0', 'label_1'])
        tn, fp, fn, tp = confusion_matrix(knn_y, knn_y_pred).ravel()

        pos_acc = tp / sum(knn_y)
        neg_acc = tn / (len(knn_y_pred) - sum(knn_y_pred))  # [y_true=0 & y_pred=0] / y_pred=0
        accuracy = (tp + tn) / (tn + fp + fn + tp)

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)

        roc_auc = roc_auc_score(knn_y, knn_y_prob[:, 1])
        prec, reca, _ = precision_recall_curve(knn_y, knn_y_prob[:, 1])
        aupr = auc(reca, prec)

        # print(
        #     'acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(
        #         accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc))
        # print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        # print('y_pred: ', Counter(knn_y_pred))
        # print('y_true: ', Counter(knn_y))

        # print('knn_score = {:.4f}'.format(knn.score(knn_x, knn_y)))

        sp.save_npz(pwd + 'task_' + task + balance + '__testlabel0_knn' + str(n_neigh) + 'neighbors_edge__fold' + str(
            fold) + '.npz', knn_neighbors_graph)
        fold += 1
    return knn_x, knn_y, knn, knn_neighbors_graph


def construct(predict_num):
    global num
    num = predict_num
    pwd = os.path.join(BASE_DIR, 'Predict','deeplearn','graph')  # 图结构路径
    # directory = "./data/"  # 数据路径
    directory = os.path.join(BASE_DIR,'Predict','deeplearn','iLearnResultCSVFile','output_file','feature')  # 数据路径


    # for isbalance in [True, False]:
    # for isbalance in [False, True]:  # 只保留False，只跑平衡数据
    # for isbalance in [False]:
    for isbalance in [True]:  # 不平衡
        # print('************isbalance = ', isbalance)

        for task in ['Tp']:
            # print('=================task = ', task)

            IP, dtp, protein_ids, protein_test_num, knn_x, label = obtain_data(directory,
                                                                               isbalance)

            if task == 'Tp':
                train_index_all, test_index_all, train_id_all, test_id_all = generate_task_Tp_train_test_idx(knn_x, dtp)

            if isbalance:
                balance = '__imbalanced'

            else:
                balance = '__balanced'
            path = os.path.join(pwd,task+balance+'__testlabel0_knn_edge_train_test_index_all.npz')
            np.savez_compressed(
                path,
                train_index_all=train_index_all,
                test_index_all=test_index_all,
                train_id_all=train_id_all,
                test_id_all=test_id_all)

            # for n_neigh in [1, 3, 5, 7, 10, 15]:
            for n_neigh in [3]:

                # print('--------------------------n_neighbors = ', n_neigh)
                knn_x, knn_y, knn, knn_neighbors_graph = generate_knn_graph_save(knn_x, label, n_neigh, train_index_all,
                                                                                 test_index_all, pwd, task, balance)
