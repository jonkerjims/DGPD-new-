#!/usr/bin/env python
# _*_coding:utf-8_*_
import os

import numpy
import numpy as np
import pandas as pd

# BASE_DIR = os.path.abspath(os.path.join(os.getcwd(__file__), ""))
BASE_DIR = os.path.abspath(os.path.join(os.getcwd()))
print(BASE_DIR)
# InputFilePath = os.path.join(BASE_DIR, r'iLearnResultCSVFile\input_file\protein_sequences.txt')
OutputFilePath = os.path.join(BASE_DIR, r'Predict\deeplearn\iLearnResultCSVFile\output_file\result.csv')
protein_label_filepath = os.path.join(BASE_DIR, r'Predict\deeplearn\iLearnResultCSVFile\output_file\feature\label.csv')
feature_filepath = os.path.join(BASE_DIR, r'Predict\deeplearn\iLearnResultCSVFile\output_file\feature\feature.csv')
# print(InputFilePath)
import argparse
import re

from descproteins import *
from pubscripts import *


def iLearnStart(InputFilePath):
    # parser = argparse.ArgumentParser(usage="it's usage tprotein_label.",
    #                                  descrprotein_labeltion="Generating various numerical representation schemes for protein sequences")
    # parser.add_argument("--file", required=True, help="input fasta file")
    # parser.add_argument("--method", required=True,
    #                     choices=['AAC', 'EAAC', 'CKSAAP', 'DPC', 'DDE', 'TPC', 'binary',
    #                              'GAAC', 'EGAAC', 'CKSAAGP', 'GDPC', 'GTPC',
    #                              'AAINDEX', 'ZSCALE', 'BLOSUM62',
    #                              'NMBroto', 'Moran', 'Geary',
    #                              'CTDC', 'CTDT', 'CTDD',
    #                              'CTriad', 'KSCTriad',
    #                              'SOCNumber', 'QSOrder',
    #                              'PAAC', 'APAAC',
    #                              'KNNprotein', 'KNNpeptide',
    #                              'PSSM', 'SSEC', 'SSEB', 'Disorder', 'DisorderC', 'DisorderB', 'ASA', 'TA'
    #                              ],
    #                     help="the encoding type")
    # parser.add_argument("--path", dest='filePath',
    #                     help="data file path used for 'PSSM', 'SSEB(C)', 'Disorder(BC)', 'ASA' and 'TA' encodings")
    # parser.add_argument("--order", dest='order',
    #                     choices=['alphabetically', 'polarity', 'sideChainVolume', 'userDefined'],
    #                     help="output order for of Amino Acid Composition (i.e. AAC, EAAC, CKSAAP, DPC, DDE, TPC) descrprotein_labeltors")
    # parser.add_argument("--userDefinedOrder", dest='userDefinedOrder',
    #                     help="user defined output order for of Amino Acid Composition (i.e. AAC, EAAC, CKSAAP, DPC, DDE, TPC) descrprotein_labeltors")
    # parser.add_argument("--format", choices=['csv', 'tsv', 'svm', 'weka', 'tsv_1'], default='svm',
    #                     help="the encoding type")
    # parser.add_argument("--out", help="the generated descrprotein_labeltor file")
    # args = parser.parse_args()
    # print(args)

    ############################定义参数####################################
    argsFile = InputFilePath
    argsFilePath = None
    argsFormat = 'svm'
    argsMethod = 'CKSAAP'
    argsUserDefinedOrder = None
    argsOrder = None
    argsOut = None
    ################################################################
    # fastas = read_fasta_sequences.read_protein_sequences(args.file)
    fastas = read_fasta_sequences.read_protein_sequences(argsFile)
    # userDefinedOrder = args.userDefinedOrder if args.userDefinedOrder != None else 'ACDEFGHIKLMNPQRSTVWY'
    userDefinedOrder = argsUserDefinedOrder if argsUserDefinedOrder != None else 'ACDEFGHIKLMNPQRSTVWY'
    userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
    if len(userDefinedOrder) != 20:
        userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
    myAAorder = {
        'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
        'polarity': 'DENKRQHSGTAPYVMCWIFL',
        'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
        'userDefined': userDefinedOrder
    }
    # myOrder = myAAorder[args.order] if args.order != None else 'ACDEFGHIKLMNPQRSTVWY'
    myOrder = myAAorder[argsOrder] if argsOrder != None else 'ACDEFGHIKLMNPQRSTVWY'
    # kw = {'path': args.filePath, 'order': myOrder, 'type': 'Protein'}
    kw = {'path': argsFilePath, 'order': myOrder, 'type': 'Protein'}
    # cmd = args.method + '.' + args.method + '(fastas, **kw)'
    cmd = argsMethod + '.' + argsMethod + '(fastas, **kw)'
    # print('Descrprotein_labeltor type: ' + args.method)
    # print('Descrprotein_labeltor type: ' + argsMethod)
    encodings = eval(cmd)
    # out_file = args.out if args.out != None else 'encoding.txt'
    out_file = argsOut if argsOut != None else OutputFilePath
    # print(out_file)
    # save_file.save_file(encodings, args.format, out_file)
    save_file.save_file(encodings, argsFormat, out_file)

    X = numpy.array(encodings)
    header = X[0, 2:]
    protein_label = X[1:, :2]
    id_ = np.ones(protein_label.shape[0])
    protein_label = np.insert(protein_label, 0, values=id_, axis=1)  # 在第0列前插入一列
    # protein_label = pd.DataFrame(protein_label).reset_index()  # 重置索引，索引从0开始
    # protein_label.rename(columns={'index': 'id'}, inplace=True)  # 加一列id
    # protein_label['id'] = protein_label['id'] + 1953  # 原始id从0开始，现在从1953开始，从我们的数据后开始加
    feature = X[1:, 2:]
    # feature = np.insert(feature, 0, values=header, axis=0)
    # feature = np.insert(feature, 0, values=header, axis=0)
    # print("X.shape:", X.shape)
    # print("protein_label.shape:", protein_label.shape)
    # print(feature)
    # print(type(feature))
    # print("feature.shape:", feature.shape)

    pd.DataFrame(X).to_csv(OutputFilePath, index=False, encoding='utf-8')
    pd.DataFrame(protein_label).to_csv(protein_label_filepath, index=False, header=['ID', '#', 'label'], encoding='utf-8')
    pd.DataFrame(feature).to_csv(feature_filepath, index=False, header=header, encoding='utf-8')
    return header

