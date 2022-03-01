# -*- coding: utf-8 -*-
import queue
import sys
import os
import time

import numpy as np
import pandas as pd
from numpy import array
from dbworm.settings import BASE_DIR as SYS_BASE_DIR
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SYS_BASE_DIR, r'DBA'))
sys.path.append(os.path.join(BASE_DIR, r'iLearn-master'))

# sys.path.append(r'C:\Users\Shi Haoyuan\Desktop\deeplearn\deeplearn\iLearn-master')

# print(os.path.exists(r'C:\Users\Shi Haoyuan\Desktop\deeplearn\iLearn-master'))
# os.path.exists(r'C:\Users\80934\Desktop\iLearnResultCSVFile\input_file\protein_sequences.txt')
# print(os.path.abspath(__file__))
from iLearn_protein_basic import *
from .GraphConstruction import *
from .model_GCN import *
from GlobeUtils import send_email


def pro_entry(email, name, File_path, q: queue.Queue):

    try:
        # print('算法线程已启动', email, name, File_path)
        n = 1 / 0
        header = iLearnStart(File_path)

        # 读取生成特征
        path = os.path.join(BASE_DIR, r'Predict\deeplearn\iLearnResultCSVFile\output_file\feature')
        feature = pd.read_csv(os.path.join(path, r'feature.csv'))
        BasicFeature = pd.read_csv(os.path.join(path, r'data.csv'))
        feature_all = BasicFeature.append(feature)
        pd.DataFrame(feature_all).to_csv(os.path.join(path, r'feature_all.csv'), index=False, header=header,
                                         encoding='utf-8')

        # 读取生成标签
        BasicLabel = pd.read_csv(os.path.join(path, r'protein_label.csv'), names=['ID', '#', 'label'])
        base_num = BasicLabel.shape[0]
        label = pd.read_csv(os.path.join(path, r'label.csv'))
        predict_num = label.shape[0]
        label_all = BasicLabel.append(label)
        id_ = []
        for i in range(1, base_num + predict_num + 1):
            id_.append(i)
        label_all['ID'] = id_
        pd.DataFrame(label_all).to_csv(os.path.join(path, r'label_all.csv'), index=False, encoding='utf-8')

        # 生成图
        construct()

        # 图卷积
        result = convolution(predict_num)
        print(result)
        # except BaseException as err:
        #     print(err)
        #     print('邮箱：序列处理失败提醒')
        # 取出管道的内容
        q.get()

    except:
        time.sleep(15)
        massage = '错误信息'
        subject = '66666666'
        backcall = send_email(email,massage,subject,name)
        print(backcall)
        # 取出管道的内容
        q.get()
        print('算法出错')
